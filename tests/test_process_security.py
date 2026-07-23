import sys

from app.core.process_security import protect_service_process_memory
from app.evaluator.engines.custom_evaluator_worker import (
    _install_linux_syscall_filter,
)


def test_linux_specific_hardening_is_a_noop_on_other_platforms(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    protect_service_process_memory()
    _install_linux_syscall_filter()
