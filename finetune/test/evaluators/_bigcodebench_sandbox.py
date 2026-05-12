"""Standalone sandbox runner for BigCodeBench tests.
Invoked via subprocess to avoid inheriting model memory from the training process.
All validation logic is identical to the original BigCodeBench evaluation.
"""
import contextlib
import faulthandler
import io
import json
import multiprocessing
import multiprocessing.queues
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from multiprocessing import Pipe


TIMEOUT_LIMIT = 240.0
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"
_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3
_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


@contextlib.contextmanager
def swallow_subprocess_output():
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if kwargs.get("capture_output", False):
            kwargs.pop("stdout", None)
            kwargs.pop("stderr", None)
        else:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if kwargs.get("capture_output", False):
            kwargs.pop("stdout", None)
            kwargs.pop("stderr", None)
        else:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


@contextlib.contextmanager
def swallow_io():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with swallow_subprocess_output():
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        cwd = os.getcwd()
        os.chdir(dirname)
        try:
            yield dirname
        finally:
            os.chdir(cwd)


def reliability_guard(max_as_limit=30 * 1024 * 1024 * 1024,
                      max_data_limit=30 * 1024 * 1024 * 1024,
                      max_stack_limit=10 * 1024 * 1024):
    """Set resource limits for sandboxed execution."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (max_as_limit, max_as_limit))
    resource.setrlimit(resource.RLIMIT_DATA, (max_data_limit, max_data_limit))
    resource.setrlimit(resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit))
    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None


@contextlib.contextmanager
def safe_environment():
    original_kill = os.kill
    original_killpg = os.killpg
    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid:
            original_killpg(pgid, sig)

    os.kill = safe_kill
    os.killpg = safe_killpg
    try:
        yield
    finally:
        os.kill = original_kill
        os.killpg = original_killpg


def unsafe_execute(entry_point, code, test_code, timeout,
                   max_as_limit, max_data_limit, max_stack_limit,
                   result_conn):
    """Run code + tests in sandbox, send (stat, details) back via Pipe."""
    details = {}
    stat = _UNKNOWN
    with safe_environment(), create_tempdir():
        import os
        import shutil
        import builtins
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({
            "__builtins__": builtins,
            "__file__": f"{module_name}.py",
            "__package__": None,
            "__doc__": None,
            "sys": sys,
            "os": os,
            "environ": os.environ,
        })

        try:
            full_code = code + "\n" + test_code
            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", "exec"), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, "TestCases")
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace
            stat = _SUCCESS
        except BaseException as e:
            details["ALL"] = str(e)
            stat = _FAILED

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

    try:
        result_conn.send((stat, details))
    except Exception:
        pass
    finally:
        result_conn.close()


def untrusted_check(code, test_code, entry_point,
                    max_as_limit=30 * 1024 * 1024 * 1024,
                    max_data_limit=30 * 1024 * 1024 * 1024,
                    max_stack_limit=10 * 1024 * 1024,
                    min_time_limit=10, gt_time_limit=60):
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = max(TIMEOUT_LIMIT, min_time_limit) + 1

    parent_conn, child_conn = Pipe(duplex=False)

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(entry_point, code, test_code, timeout,
              max_as_limit, max_data_limit, max_stack_limit,
              child_conn),
    )
    p.start()
    child_conn.close()

    stat = _UNKNOWN
    details = {}
    if parent_conn.poll(timeout + 1):
        try:
            stat, details = parent_conn.recv()
        except Exception:
            stat = _FAILED
            details = {"ALL": "Failed to receive result from sandbox"}
    parent_conn.close()

    p.join(timeout=1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping.get(stat, TIMEOUT)
    if not stat:
        stat = TIMEOUT
    if stat == PASS and details:
        stat = FAIL
    return stat, details


if __name__ == "__main__":
    input_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(input_path, "r") as f:
        data = json.load(f)

    stat, details = untrusted_check(
        code=data["code"],
        test_code=data["test_code"],
        entry_point=data["entry_point"],
    )

    with open(result_path, "w") as f:
        json.dump({"stat": stat, "details": details}, f)
