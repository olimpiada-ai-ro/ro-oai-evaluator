"""Defense-in-depth policy for custom evaluator child processes.

This is not a replacement for the container boundary. It narrows the Python
process' ambient authority and ensures the one compatibility subprocess used by
ZIP evaluators inherits the same restrictions.
"""

from __future__ import annotations

import base64
import builtins
import json
import os
import site
import struct
import subprocess
import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from app.core.native_threads import (
    MODEL_RUNTIME_ENVIRONMENT,
    NATIVE_THREAD_ENVIRONMENT,
    configure_torch_threads,
)

_DENIED_PROCESS_EVENTS = {
    "os.fork",
    "os.forkpty",
    "os.killpg",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.setpgid",
    "os.setsid",
    "os.spawn",
    "os.system",
    "pty.spawn",
}
_WRITE_EVENTS = {
    "os.chmod",
    "os.chown",
    "os.link",
    "os.mkdir",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "os.symlink",
    "os.truncate",
    "os.unlink",
    "os.utime",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
}
_READ_PATH_EVENTS = {"os.chdir", "os.listdir", "os.scandir"}
_DENIED_IMPORT_ROOTS = {
    "_cffi_backend",
    "_ctypes",
    "_multiprocessing",
    "_posixsubprocess",
    "cffi",
    "ctypes",
    "multiprocessing",
    "resource",
}
_NESTED_PRELOAD_ROOTS = {
    "numpy",
    "pandas",
    "safetensors",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
}
_DENIED_FFI_ROOTS = {"_cffi_backend", "_ctypes", "cffi", "ctypes"}


class _BlockedNativeModule:
    """Inert import result for a platform-specific optional native module."""

    __slots__ = ()

    def __getattr__(self, name: str):
        raise ImportError(f"Native attribute {name!r} is disabled")


_BLOCKED_CTYPES_STUB = _BlockedNativeModule()


class _SafeCLong:
    """Marker used only for SymPy's platform word-size detection."""

    __slots__ = ()


class _SafeCtypesSizeModule:
    """Minimal ctypes substitute without pointers, addresses, or native loading."""

    __slots__ = ()
    c_long = _SafeCLong

    @staticmethod
    def sizeof(value: Any) -> int:
        if value is _SafeCLong or isinstance(value, _SafeCLong):
            return struct.calcsize("@l")
        raise ImportError("Only the safe c_long size probe is available")


_SAFE_CTYPES_SIZE_STUB = _SafeCtypesSizeModule()


class _SafePlatformCtypesModule:
    """POSIX import shim for Torch code whose ctypes paths are Windows-only."""

    __slots__ = ()
    cdll = _BLOCKED_CTYPES_STUB
    wintypes = _BLOCKED_CTYPES_STUB

    @staticmethod
    def find_library(_name: str):
        return None


_SAFE_PLATFORM_CTYPES_STUB = _SafePlatformCtypesModule()


def _resolved(path: os.PathLike[str] | str) -> Path:
    return Path(os.fsdecode(path)).expanduser().resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def discover_runtime_roots(project_root: Path) -> list[Path]:
    """Return stdlib/site-package roots while excluding application source."""
    candidates: list[Path] = []
    for value in sys.path:
        if not value:
            continue
        try:
            candidate = _resolved(value)
        except (OSError, TypeError, ValueError):
            continue
        if candidate.exists() and not _within(candidate, project_root):
            candidates.append(candidate)

    runtime_candidates = [
        Path(sys.base_prefix) / "lib",
        Path(sys.prefix) / "lib",
        *(Path(value) for value in site.getsitepackages()),
    ]
    for value in runtime_candidates:
        try:
            candidate = _resolved(value)
        except (OSError, TypeError, ValueError):
            continue
        if candidate.exists():
            candidates.append(candidate)

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


class SandboxPolicy:
    def __init__(
        self,
        *,
        scratch_root: Path,
        data_roots: Iterable[Path],
        runtime_roots: Iterable[Path],
        forbidden_paths: Iterable[Path],
        project_root: Path,
        allow_python_subprocess: bool,
        nested_preload_modules: Iterable[str] = (),
    ):
        self.scratch_root = _resolved(scratch_root)
        self.data_roots = tuple(_resolved(path) for path in data_roots)
        self.runtime_roots = tuple(_resolved(path) for path in runtime_roots)
        self.project_root = _resolved(project_root)
        self._guard_dir = self.scratch_root / ".python_guard"
        self.forbidden_paths = (
            *tuple(_resolved(path) for path in forbidden_paths),
            self._guard_dir,
        )
        self.allow_python_subprocess = allow_python_subprocess
        self.nested_preload_modules = tuple(
            sorted(
                {
                    name.split(".", 1)[0]
                    for name in nested_preload_modules
                    if name.split(".", 1)[0] in _NESTED_PRELOAD_ROOTS
                }
            )
        )
        self._state = threading.local()
        self._original_open = builtins.open
        self._original_import = builtins.__import__
        self._original_popen = subprocess.Popen
        self._nested_spawn_count = 0
        self._nested_pids: set[int] = set()

    @property
    def read_roots(self) -> tuple[Path, ...]:
        return (self.scratch_root, *self.data_roots, *self.runtime_roots)

    @property
    def execution_roots(self) -> tuple[Path, ...]:
        return (self.scratch_root, *self.data_roots)

    def install(self) -> None:
        self._prepare_nested_guard()
        sys.addaudithook(self.audit)
        builtins.open = self.safe_open
        builtins.__import__ = self.safe_import
        subprocess.Popen = self.guarded_popen  # type: ignore[assignment]

    @contextmanager
    def trusted_operations(self) -> Iterator[None]:
        previous = getattr(self._state, "trusted", False)
        self._state.trusted = True
        try:
            yield
        finally:
            self._state.trusted = previous

    def audit(self, event: str, args: tuple[Any, ...]) -> None:
        if getattr(self._state, "trusted", False):
            return

        if event.startswith("socket."):
            raise PermissionError("Network access is disabled for custom evaluators")
        if event.startswith("ctypes."):
            raise PermissionError("ctypes is disabled for custom evaluators")
        if event == "os.kill":
            pid = args[0] if args else None
            if isinstance(pid, int) and pid in self._nested_pids:
                return
            raise PermissionError("Process control is disabled for custom evaluators")
        if event in _DENIED_PROCESS_EVENTS:
            raise PermissionError("Process control is disabled for custom evaluators")
        if event == "subprocess.Popen":
            if not getattr(self._state, "approved_spawn", False):
                raise PermissionError(
                    "Only the guarded Python evaluator worker may be launched"
                )
            return
        if event == "open" and args:
            path = args[0]
            if isinstance(path, int):
                if getattr(self._state, "approved_spawn", False):
                    return
                raise PermissionError("Opening inherited file descriptors is disabled")
            if (
                getattr(self._state, "approved_spawn", False)
                and os.fspath(path) == os.devnull
            ):
                # subprocess opens the platform null device itself for
                # stdin=DEVNULL. It is safe only while launching the already
                # validated nested worker.
                return
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            write = self._is_write_open(mode, flags)
            self._check_path(path, write=write)
            return
        if event in _READ_PATH_EVENTS and args and args[0] is not None:
            # shutil.rmtree uses os.scandir(directory_fd) after opening the
            # directory. Validate the descriptor's actual target so cleanup
            # works without turning descriptor access into a path escape.
            if isinstance(args[0], int):
                self._check_directory_fd(args[0])
            else:
                self._check_path(args[0], write=False)
            return
        if event in _WRITE_EVENTS and args:
            if event in {"os.link", "os.symlink"}:
                raise PermissionError("Creating filesystem links is disabled")
            path_indexes = (
                (0, 1)
                if event
                in {
                    "os.rename",
                    "os.replace",
                    "shutil.copyfile",
                }
                else (0,)
            )
            for index in path_indexes:
                if index < len(args) and args[index] is not None:
                    # Copy/link sources may be read-only data, while targets
                    # must always stay in scratch.
                    is_source = index == 0 and event in {
                        "shutil.copyfile",
                    }
                    self._check_path(args[index], write=not is_source)

    def safe_open(
        self,
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        opener=None,
    ):
        if isinstance(file, int):
            raise PermissionError("Opening inherited file descriptors is disabled")
        self._check_path(file, write=self._is_write_open(mode, 0))
        return self._original_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    def safe_import(
        self,
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        root = name.split(".", 1)[0]
        if root in _DENIED_IMPORT_ROOTS:
            requester = (globals or {}).get("__name__", "")
            if (
                root == "ctypes"
                and os.name != "nt"
                and requester == "huggingface_hub.utils._terminal"
            ):
                # Hugging Face imports ctypes for Windows console handling even
                # on POSIX, where it is never used. Return an inert module only
                # for that exact trusted compatibility import; participant and
                # all other package imports remain denied.
                return _BLOCKED_CTYPES_STUB
            if (
                root == "ctypes"
                and os.name != "nt"
                and requester == "sympy.external.gmpy"
            ):
                # SymPy only needs the native C long width to choose its safe
                # integer implementation. Provide that value without exposing
                # pointers, memory addresses, or dynamic-library loading.
                return _SAFE_CTYPES_SIZE_STUB
            if (
                root == "ctypes"
                and os.name != "nt"
                and requester == "torch._inductor.cpp_builder"
            ):
                # This cross-platform module imports Windows loader helpers at
                # module import time. Its POSIX inference path does not use
                # them, so provide inert names rather than real native access.
                return _SAFE_PLATFORM_CTYPES_STUB
            raise ImportError(f"Import of {root!r} is disabled in custom evaluators")
        return self._original_import(name, globals, locals, fromlist, level)

    def guarded_popen(self, *popenargs, **kwargs):
        if not self.allow_python_subprocess:
            raise PermissionError("Nested process creation is disabled")
        if self._nested_spawn_count >= 1:
            raise PermissionError("Only one nested Python evaluator worker is allowed")

        args = popenargs[0] if popenargs else kwargs.get("args")
        command = self._validate_python_command(args, kwargs)
        nested_env = self._nested_environment()

        guarded_kwargs = dict(kwargs)
        # Legacy evaluator workers commonly provide their own filtered env and
        # request a new session for timeout cleanup. These are compatibility
        # hints only: always replace both with the stricter sandbox values.
        guarded_kwargs["env"] = nested_env
        guarded_kwargs["close_fds"] = True
        guarded_kwargs["start_new_session"] = False
        guarded_kwargs.pop("preexec_fn", None)
        guarded_kwargs.pop("creationflags", None)

        self._nested_spawn_count += 1
        self._state.approved_spawn = True
        try:
            if popenargs:
                process = self._original_popen(
                    command, *popenargs[1:], **guarded_kwargs
                )
            else:
                guarded_kwargs["args"] = command
                process = self._original_popen(**guarded_kwargs)
            self._nested_pids.add(process.pid)
            return process
        finally:
            self._state.approved_spawn = False

    def _validate_python_command(self, args: Any, kwargs: dict[str, Any]) -> list[str]:
        if kwargs.get("shell"):
            raise PermissionError("shell=True is disabled")
        if kwargs.get("preexec_fn") is not None:
            raise PermissionError("preexec_fn is disabled")
        if kwargs.get("process_group") not in (None, -1):
            raise PermissionError(
                "Changing the nested worker process group is disabled"
            )
        if kwargs.get("creationflags"):
            raise PermissionError("Nested worker creation flags are disabled")
        if any(
            kwargs.get(name) is not None for name in ("user", "group", "extra_groups")
        ):
            raise PermissionError("Changing nested worker identity is disabled")
        if kwargs.get("umask", -1) != -1:
            raise PermissionError("Changing nested worker umask is disabled")
        if kwargs.get("pass_fds"):
            raise PermissionError("Passing file descriptors is disabled")

        if not isinstance(args, (list, tuple)) or len(args) < 2:
            raise PermissionError(
                "Nested workers must use [sys.executable, script.py, ...]"
            )
        command = [os.fspath(value) for value in args]
        executable = kwargs.get("executable") or command[0]
        if _resolved(executable) != _resolved(sys.executable):
            raise PermissionError("Only the current Python interpreter is allowed")
        if command[0] != os.fspath(sys.executable):
            raise PermissionError("Python executable must be exactly sys.executable")

        script_arg = command[1]
        if script_arg.startswith("-"):
            raise PermissionError("Python command/module execution flags are disabled")

        cwd = _resolved(kwargs.get("cwd") or os.getcwd())
        if not any(_within(cwd, root) for root in self.execution_roots):
            raise PermissionError("Nested worker cwd is outside allowed roots")

        script_path = _resolved(
            script_arg if os.path.isabs(script_arg) else cwd / script_arg
        )
        if (
            script_path.suffix != ".py"
            or not script_path.is_file()
            or not any(_within(script_path, root) for root in self.execution_roots)
        ):
            raise PermissionError("Nested worker script is outside allowed roots")

        return command

    def _prepare_nested_guard(self) -> None:
        self._guard_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        sitecustomize = self._guard_dir / "sitecustomize.py"
        sitecustomize.write_text(
            "import os as _os\n"
            "try:\n"
            "    from app.evaluator.engines.sandbox_policy import "
            "install_nested_guard_from_environment as _install\n"
            "    _install()\n"
            "except BaseException:\n"
            "    _os._exit(126)\n",
            encoding="utf-8",
        )
        sitecustomize.chmod(0o600)

    def _nested_environment(self) -> dict[str, str]:
        config = {
            "scratch_root": str(self.scratch_root),
            "data_roots": [str(path) for path in self.data_roots],
            "runtime_roots": [str(path) for path in self.runtime_roots],
            "forbidden_paths": [str(path) for path in self.forbidden_paths],
            "project_root": str(self.project_root),
            "preload_modules": list(self.nested_preload_modules),
        }
        encoded_config = base64.urlsafe_b64encode(
            json.dumps(config).encode("utf-8")
        ).decode("ascii")
        environment = {
            "PATH": os.defpath,
            "PYTHONPATH": os.pathsep.join(
                (str(self._guard_dir), str(self.project_root))
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "random",
            "TMPDIR": str(self.scratch_root),
            "TEMP": str(self.scratch_root),
            "TMP": str(self.scratch_root),
            "HF_HOME": str(self.scratch_root / ".model-cache"),
            "RO_OAI_SANDBOX_POLICY": encoded_config,
        }
        environment.update(NATIVE_THREAD_ENVIRONMENT)
        environment.update(MODEL_RUNTIME_ENVIRONMENT)
        return environment

    def _check_path(self, raw_path: Any, *, write: bool) -> None:
        try:
            path = _resolved(raw_path)
        except (OSError, TypeError, ValueError) as exc:
            raise PermissionError("Invalid filesystem path") from exc

        if any(_within(path, forbidden) for forbidden in self.forbidden_paths):
            raise PermissionError(f"Access denied: {path}")
        if _within(path, self.project_root) and (
            path.name.startswith(".env") or path.name in {"config.yaml", "config.yml"}
        ):
            raise PermissionError(f"Access denied: {path}")

        if write:
            if not any(_within(path, root) for root in self.execution_roots):
                raise PermissionError(
                    "Writes are restricted to evaluator scratch and data roots"
                )
            return

        if not any(_within(path, root) for root in self.read_roots):
            raise PermissionError(f"Read access denied: {path}")

    def _check_directory_fd(self, file_descriptor: int) -> None:
        if file_descriptor < 0:
            raise PermissionError("Invalid directory file descriptor")

        for descriptor_root in (Path("/proc/self/fd"), Path("/dev/fd")):
            descriptor_path = descriptor_root / str(file_descriptor)
            try:
                target = os.readlink(descriptor_path)
            except (OSError, ValueError):
                continue

            target_path = Path(target)
            if not target_path.is_absolute():
                target_path = descriptor_root / target_path
            self._check_path(target_path, write=False)
            return

        raise PermissionError("Unable to validate directory file descriptor")

    @staticmethod
    def _is_write_open(mode: Any, flags: Any) -> bool:
        if isinstance(mode, str) and any(char in mode for char in "wax+"):
            return True
        if isinstance(flags, int):
            write_flags = (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            return bool(flags & write_flags)
        return False


def install_nested_guard_from_environment() -> None:
    encoded = os.environ.pop("RO_OAI_SANDBOX_POLICY", "")
    if not encoded:
        raise RuntimeError("Missing nested evaluator sandbox policy")

    config = json.loads(
        base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
    )

    # Native scientific packages may import ctypes as an implementation detail.
    # Load only the evaluator-requested allowlist before the strict import guard
    # is installed; Newsroom-style solution workers always retain torch
    # compatibility.
    requested_preloads = {
        str(name).split(".", 1)[0] for name in config.get("preload_modules", [])
    }.intersection(_NESTED_PRELOAD_ROOTS)
    requested_preloads.add("torch")
    for module_name in (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "torch",
        "safetensors",
        "transformers",
    ):
        if module_name not in requested_preloads:
            continue
        try:
            module = __import__(module_name)
            if module_name == "torch":
                configure_torch_threads(module)
        except Exception:
            # Preserve normal missing/broken dependency behavior when the
            # participant worker later performs its own import.
            continue

    policy = SandboxPolicy(
        scratch_root=Path(config["scratch_root"]),
        data_roots=[Path(path) for path in config["data_roots"]],
        runtime_roots=[Path(path) for path in config["runtime_roots"]],
        forbidden_paths=[Path(path) for path in config["forbidden_paths"]],
        project_root=Path(config["project_root"]),
        allow_python_subprocess=False,
    )
    policy.install()
    _harden_preloaded_native_modules()


def _harden_preloaded_native_modules() -> None:
    def is_denied_reference(value: Any) -> bool:
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

    try:
        torch = sys.modules.get("torch")
        if torch is not None:

            def deny_native_library_loading(*_args, **_kwargs):
                raise PermissionError(
                    "Loading native libraries is disabled in custom evaluators"
                )

            for loader in (torch.ops, torch.classes):
                try:
                    loader.load_library = deny_native_library_loading
                except (AttributeError, RuntimeError, TypeError):
                    pass
                try:
                    type(loader).load_library = deny_native_library_loading
                except (AttributeError, RuntimeError, TypeError):
                    pass
    except (AttributeError, RuntimeError, TypeError):
        pass

    for module in tuple(sys.modules.values()):
        if type(module) is not types.ModuleType:
            continue
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for attribute, value in tuple(namespace.items()):
            if is_denied_reference(value):
                namespace.pop(attribute, None)

    for module_name in tuple(sys.modules):
        if module_name.split(".", 1)[0] in _DENIED_IMPORT_ROOTS:
            sys.modules.pop(module_name, None)
