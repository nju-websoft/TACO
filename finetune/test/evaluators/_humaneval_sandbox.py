"""Standalone sandbox runner for HumanEval tests.
Invoked via subprocess to avoid inheriting model memory from the training process.
All validation logic is identical to the original HumanEval evaluation.
"""
import contextlib
import faulthandler
import io
import json
import multiprocessing
import os
import platform
import signal
import sys
import tempfile
import time
from multiprocessing import Pipe


TIMEOUT_LIMIT = 10.0


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        cwd = os.getcwd()
        os.chdir(dirname)
        try:
            yield dirname
        finally:
            os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """Lightweight sandbox: only disable builtins that could halt the process."""
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"


def _unsafe_execute(problem, completion, timeout, result_conn):
    """Run prompt+completion+test in sandbox, send result via Pipe."""
    with create_tempdir():
        reliability_guard()

        check_program = (
            problem["prompt"]
            + completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result_conn.send("passed")
        except TimeoutException:
            result_conn.send("timed out")
        except BaseException as e:
            result_conn.send(f"failed: {e}")
        finally:
            result_conn.close()


def check_correctness(problem, completion, timeout):
    """Evaluate functional correctness using Pipe."""
    parent_conn, child_conn = Pipe(duplex=False)

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(problem, completion, timeout, child_conn),
        daemon=False,
    )
    p.start()
    child_conn.close()

    result = "timed out"
    if parent_conn.poll(timeout + 2):
        try:
            result = parent_conn.recv()
        except Exception:
            result = "failed: recv error"
    parent_conn.close()

    p.join(timeout=1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    return {
        "task_id": problem["task_id"],
        "passed": result == "passed",
        "result": result,
    }


if __name__ == "__main__":
    input_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(input_path, "r") as f:
        data = json.load(f)

    r = check_correctness(
        problem=data["problem"],
        completion=data["completion"],
        timeout=TIMEOUT_LIMIT,
    )

    with open(result_path, "w") as f:
        json.dump(r, f)
