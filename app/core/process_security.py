"""Linux process hardening for the long-lived evaluator service."""

from __future__ import annotations

import ctypes
import os
import sys

from app.core.logging import get_logger

logger = get_logger("process_security")

_PR_SET_DUMPABLE = 4


def protect_service_process_memory() -> None:
    """Prevent same-UID evaluator children from inspecting the parent process.

    Participant code receives a deliberately minimal environment, but the
    broker credentials necessarily remain in the long-lived service process.
    Marking that process non-dumpable closes Linux ``/proc/<pid>`` and ptrace
    access to those credentials without changing normal child execution.
    """

    if not sys.platform.startswith("linux"):
        return

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int

    if prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise RuntimeError(
            "Could not protect evaluator service process memory: "
            f"{os.strerror(error_number)}"
        )

    logger.info("Evaluator service process memory is protected")
