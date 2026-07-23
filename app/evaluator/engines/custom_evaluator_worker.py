"""Disposable process entry point for participant-provided evaluator scripts."""

from __future__ import annotations

import ast
import ctypes
import ctypes.util
import errno
import json
import math
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_PR_SET_NO_NEW_PRIVS = 38
_MAX_REAL_UID_TASKS = 128
_DENIED_NATIVE_ROOTS = {
    "_cffi_backend",
    "_ctypes",
    "_multiprocessing",
    "_posixsubprocess",
    "cffi",
    "ctypes",
    "multiprocessing",
    "resource",
}
_DENIED_FFI_ROOTS = {"_cffi_backend", "_ctypes", "cffi", "ctypes"}

# These syscalls are never needed by an evaluator. Blocking them in the kernel
# means a native extension or a Python sandbox escape still cannot reach the
# network, inspect another process, mount filesystems, or open a second kernel
# execution surface through io_uring/eBPF.
_BLOCKED_LINUX_SYSCALLS = {
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "shutdown",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "pidfd_getfd",
    "pidfd_send_signal",
    "kill",
    "tkill",
    "tgkill",
    "mount",
    "umount2",
    "pivot_root",
    "setns",
    "unshare",
    "open_by_handle_at",
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "keyctl",
    "add_key",
    "request_key",
}


def _install_linux_syscall_filter() -> None:
    """Install a fail-closed libseccomp denylist for participant execution."""

    if not sys.platform.startswith("linux"):
        return

    library_name = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    try:
        seccomp = ctypes.CDLL(library_name, use_errno=True)
    except OSError as exc:
        raise RuntimeError("libseccomp is required for custom evaluators") from exc

    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise RuntimeError(
            f"Could not enable no-new-privileges: {os.strerror(error_number)}"
        )

    context = seccomp.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("Could not initialize custom evaluator seccomp policy")

    installed: set[str] = set()
    try:
        deny_action = _SCMP_ACT_ERRNO | errno.EPERM
        for name in sorted(_BLOCKED_LINUX_SYSCALLS):
            syscall_number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if syscall_number < 0:
                continue
            result = seccomp.seccomp_rule_add(
                context,
                deny_action,
                syscall_number,
                0,
            )
            if result != 0:
                raise RuntimeError(
                    f"Could not restrict custom evaluator syscall {name}"
                )
            installed.add(name)

        # These are present on every supported Linux deployment. Treat a
        # missing rule as a broken sandbox instead of silently weakening it.
        missing_required = {"socket", "connect", "ptrace"}.difference(installed)
        if missing_required:
            raise RuntimeError(
                "Custom evaluator seccomp policy is incomplete: "
                + ", ".join(sorted(missing_required))
            )

        if seccomp.seccomp_load(context) != 0:
            error_number = ctypes.get_errno()
            raise RuntimeError(
                "Could not load custom evaluator seccomp policy: "
                f"{os.strerror(error_number)}"
            )
    finally:
        seccomp.seccomp_release(context)


def _is_denied_native_reference(value: Any) -> bool:
    if type(value) is types.ModuleType:
        module_name = getattr(value, "__name__", "")
    elif type(value) is types.FunctionType:
        module_name = getattr(value, "__module__", "")
    else:
        module_name = getattr(type(value), "__module__", "")
    return (
        isinstance(module_name, str)
        and module_name.split(".", 1)[0] in _DENIED_FFI_ROOTS
    )


def _purge_native_escape_modules() -> None:
    """Remove retained native/process handles before participant code runs."""

    # Trusted scientific imports may retain aliases such as torch.ctypes even
    # after their source module is removed from sys.modules. Scrub those direct
    # references as well so the import denylist cannot be bypassed through an
    # already-loaded package namespace.
    for module in tuple(sys.modules.values()):
        if type(module) is not types.ModuleType:
            continue
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for attribute, value in tuple(namespace.items()):
            if _is_denied_native_reference(value):
                namespace.pop(attribute, None)

    for module_name in tuple(sys.modules):
        if module_name.split(".", 1)[0] in _DENIED_NATIVE_ROOTS:
            sys.modules.pop(module_name, None)

    globals().pop("ctypes", None)


def _set_limit(resource_module, name: str, soft: int, hard: int) -> None:
    limit = getattr(resource_module, name, None)
    if limit is None:
        return

    try:
        _, current_hard = resource_module.getrlimit(limit)
        if current_hard != resource_module.RLIM_INFINITY:
            hard = min(hard, current_hard)
            soft = min(soft, hard)
        resource_module.setrlimit(limit, (soft, hard))
    except (OSError, ValueError):
        # Containers may prohibit lowering individual limits. Other limits and
        # the parent's wall-clock deadline still remain in force.
        pass


def _apply_resource_limits(timeout_seconds: float, memory_limit_mb: int) -> None:
    if os.name != "posix":
        return

    import resource

    # Keep the CPU limit just beyond the parent's wall deadline so timeout
    # errors remain deterministic while still bounding runaway descendants.
    cpu_soft = max(1, int(math.ceil(timeout_seconds)) + 1)
    _set_limit(resource, "RLIMIT_CPU", cpu_soft, cpu_soft + 1)
    _set_limit(resource, "RLIMIT_CORE", 0, 0)
    _set_limit(resource, "RLIMIT_NOFILE", 256, 256)
    _set_limit(
        resource,
        "RLIMIT_FSIZE",
        512 * 1024 * 1024,
        512 * 1024 * 1024,
    )

    # RLIMIT_NPROC counts every process and thread with this real UID, including
    # other containers and rolling replicas on the node. A ceiling of 64 could
    # therefore be exhausted before a worker started. 128 retains a coarse
    # fork-bomb barrier while leaving room for the bounded native pools and the
    # one guarded nested Python worker.
    #
    # RLIMIT_AS is dependable in the Linux production container but behaves
    # inconsistently for native scientific libraries on macOS.
    if sys.platform.startswith("linux"):
        _set_limit(
            resource,
            "RLIMIT_NPROC",
            _MAX_REAL_UID_TASKS,
            _MAX_REAL_UID_TASKS,
        )
        memory_bytes = max(256, memory_limit_mb) * 1024 * 1024
        _set_limit(resource, "RLIMIT_AS", memory_bytes, memory_bytes)


def _write_response(result_path: Path, response: dict[str, Any]) -> None:
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(response, result_file, ensure_ascii=False, allow_nan=True)
    os.chmod(result_path, 0o600)


def _discover_requested_modules(script_content: str) -> set[str]:
    syntax_tree = ast.parse(script_content)
    requested = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            requested.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            requested.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Name):
            requested.add(node.id)
    return requested


def _preload_native_modules(requested_modules: set[str]) -> None:
    """Load explicitly requested native modules before the ctypes guard."""
    model_modules = {"torch", "transformers", "safetensors"}
    if requested_modules.intersection(model_modules):
        from app.core.native_threads import configure_torch_threads

        torch = __import__("torch")
        configure_torch_threads(torch)

        def deny_native_library_loading(*_args, **_kwargs):
            raise PermissionError(
                "Loading native libraries is disabled in custom evaluators"
            )

        # Torch's dynamic extension loaders bypass Python's guarded open().
        # Replace both instance and class entry points before participant code
        # receives the trusted preloaded module.
        for loader in (torch.ops, torch.classes):
            try:
                loader.load_library = deny_native_library_loading
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                type(loader).load_library = deny_native_library_loading
            except (AttributeError, RuntimeError, TypeError):
                pass


def main() -> int:
    if len(sys.argv) != 7:
        sys.stderr.write("Invalid custom evaluator worker invocation\n")
        return 2

    project_root = Path(sys.argv[1]).resolve()
    request_path = Path(sys.argv[2]).resolve()
    result_path = Path(sys.argv[3]).resolve()
    timeout_seconds = float(sys.argv[4])
    memory_limit_mb = int(sys.argv[5])

    # The final argv slot is reserved so adding worker protocol versions later
    # does not require exposing environment variables to participant code.
    protocol_version = sys.argv[6]
    if protocol_version != "1":
        sys.stderr.write("Unsupported custom evaluator worker protocol\n")
        return 2

    sys.path.insert(0, str(project_root))

    # Apply native-library limits before importing CustomEvaluator, which loads
    # NumPy and Pandas at module import time. The parent also supplies these
    # values, but applying them here makes direct worker invocation safe.
    from app.core.native_threads import apply_native_thread_environment

    apply_native_thread_environment()
    _apply_resource_limits(timeout_seconds, memory_limit_mb)
    policy = None

    try:
        with request_path.open("r", encoding="utf-8") as request_file:
            request = json.load(request_file)

        requested_modules = _discover_requested_modules(request["script_content"])
        os.environ["RO_OAI_CUSTOM_EVALUATOR_IMPORTS"] = ",".join(requested_modules)
        try:
            from app.evaluator.engines.custom_evaluator import CustomEvaluator
        finally:
            os.environ.pop("RO_OAI_CUSTOM_EVALUATOR_IMPORTS", None)
        from app.evaluator.engines.sandbox_policy import (
            SandboxPolicy,
            discover_runtime_roots,
        )

        evaluator = CustomEvaluator(
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
        evaluator.load_script(request["script_content"])
        _preload_native_modules(requested_modules)
        _install_linux_syscall_filter()
        _purge_native_escape_modules()

        data_roots = []
        for value in (
            request.get("extraction_path"),
            request.get("ground_truth_path"),
        ):
            if value:
                data_roots.append(Path(value).resolve(strict=False))

        scratch_root = result_path.parent
        policy = SandboxPolicy(
            scratch_root=scratch_root,
            data_roots=data_roots,
            runtime_roots=discover_runtime_roots(project_root),
            forbidden_paths=[
                Path("/proc"),
                Path("/sys"),
                Path("/dev"),
                Path("/app/cache"),
                project_root / ".env",
                project_root / ".env.production",
                project_root / ".env.local",
                project_root / "config.yaml",
                project_root / "cache",
                request_path,
                result_path,
            ],
            project_root=project_root,
            allow_python_subprocess=True,
            nested_preload_modules=requested_modules,
        )
        policy.install()
        CustomEvaluator.ALLOWED_BUILTINS["open"] = policy.safe_open
        CustomEvaluator.ALLOWED_BUILTINS["__import__"] = policy.safe_import

        # Do not expose request/result locations through /proc or sys.argv.
        sys.argv = ["custom-evaluator"]

        if request["mode"] == "records":
            result = evaluator._execute_in_process(
                request["predictions"],
                request["ground_truth"],
                capture_internal_logs=False,
            )
        elif request["mode"] == "paths":
            result = evaluator._execute_with_paths_in_process(
                extraction_path=request.get("extraction_path"),
                predictions=request.get("predictions"),
                ground_truth_path=request.get("ground_truth_path"),
                ground_truth=request.get("ground_truth"),
                capture_internal_logs=False,
            )
        else:
            raise ValueError(f"Unsupported execution mode: {request['mode']}")

        response = {
            "status": "ok",
            "result": {name: metrics.model_dump() for name, metrics in result.items()},
        }
        if policy is not None:
            with policy.trusted_operations():
                _write_response(result_path, response)
        else:
            _write_response(result_path, response)
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            response = {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            if policy is not None:
                with policy.trusted_operations():
                    _write_response(result_path, response)
            else:
                _write_response(result_path, response)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Include an explicit protocol argument in the parent command. Keeping the
    # worker as a plain script lets it apply resource limits before importing
    # pandas, scipy, OpenCV, or participant code.
    raise SystemExit(main())
