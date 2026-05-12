"""Standalone sandbox runner for CRUXEval tests (Input & Output prediction).
Invoked via subprocess to avoid inheriting model memory from the training process.
"""
import contextlib
import io
import json
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Pipe


TIMEOUT_LIMIT = 5.0


class TimeoutException(Exception):
    pass


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


def _execute(check_program, timeout, result_conn):
    """Run check_program with timeout, send result via Pipe."""
    try:
        exec_globals = {}
        with time_limit(timeout):
            exec(check_program, exec_globals)
        result_conn.send("passed")
    except TimeoutException:
        result_conn.send("timed out")
    except BaseException as e:
        try:
            result_conn.send(f"failed: {e}")
        except Exception:
            pass
    finally:
        try:
            result_conn.close()
        except Exception:
            pass


def check_correctness(check_program, timeout=3.0):
    """Evaluate correctness using Pipe."""
    parent_conn, child_conn = Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_execute,
        args=(check_program, timeout, child_conn),
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

    return result == "passed"


if __name__ == "__main__":
    input_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(input_path, "r") as f:
        data = json.load(f)

    passed = check_correctness(
        check_program=data["check_program"],
        timeout=TIMEOUT_LIMIT,
    )

    with open(result_path, "w") as f:
        json.dump({"passed": passed}, f)
