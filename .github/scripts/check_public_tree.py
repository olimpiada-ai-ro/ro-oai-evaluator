#!/usr/bin/env python3
"""Enforce the public evaluator repository boundary without printing content."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Iterable
from urllib.parse import unquote


MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_GIT_MODE = "100644"
REQUIRED_PATHS = {
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/scripts/check_public_tree.py",
    ".github/workflows/ci.yml",
    ".github/workflows/pr-policy.yml",
    ".gitignore",
    "CONTRIBUTING.md",
    "README.md",
    "app/main.py",
    "poetry.lock",
    "pyproject.toml",
    "tests/conftest.py",
}

TRUSTED_OVERLAY_PATHS = (
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/scripts/check_public_tree.py",
    ".github/workflows/ci.yml",
    ".github/workflows/pr-policy.yml",
    ".gitignore",
    "CONTRIBUTING.md",
    "README.md",
)

SOURCE_ALLOWED_PATTERNS = (
    "app/*.py",
    "app/**/*.py",
    "pyproject.toml",
    "poetry.lock",
    "test_data/*.py",
    "test_data/**/*.py",
    "test_payloads/*.py",
    "test_payloads/**/*.py",
    "tests/*.py",
    "tests/**/*.py",
)

ALLOWED_PATTERNS = TRUSTED_OVERLAY_PATHS + SOURCE_ALLOWED_PATTERNS

FORBIDDEN_PREFIXES = (
    ".agents/",
    ".codex/",
    ".cursor/",
    ".idea/",
    ".kiro/",
    ".openai/",
    ".public-mirror/",
    ".vscode/",
    ".windsurf/",
    "custom_problems/",
    "docs/",
    "kubernetes/",
    "message_schemas/",
    "problematic-problem2/",
    "scripts/",
)

SOURCE_DENIED_PATHS = {
    "tests/test_kubernetes_health_probes.py",
}

FORBIDDEN_EXACT_PATHS = SOURCE_DENIED_PATHS

FORBIDDEN_BASENAMES = {
    ".env",
    "agents.md",
    "claude.md",
    "copilot-instructions.md",
    "config.yaml",
    "dockerfile",
    "gemini.md",
    "license",
    "license.md",
    "makefile",
    "product.md",
    "skill.md",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".a",
    ".bin",
    ".bz2",
    ".crt",
    ".db",
    ".der",
    ".dll",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".feather",
    ".gz",
    ".h5",
    ".hdf5",
    ".jar",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".key",
    ".kubeconfig",
    ".mdb",
    ".npz",
    ".npy",
    ".onnx",
    ".p12",
    ".parquet",
    ".pem",
    ".pdf",
    ".pfx",
    ".pickle",
    ".pkl",
    ".png",
    ".rar",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".tiff",
    ".webp",
    ".whl",
    ".xls",
    ".xlsx",
    ".xz",
    ".zip",
}

SECRET_PATTERNS = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
)

URL_USERINFO = re.compile(
    r"\b[a-z][a-z0-9+.-]*://([^/@\s\"'<>:]+):([^/@\s\"'<>]+)@[^\s\"'<>/]+",
    re.IGNORECASE,
)

EXPLICIT_TEST_URL_USERINFO = {
    ("example-user", "example-password"),
    (
        "example-user@example.invalid",
        "example-password/with-symbols?#",
    ),
}

PRIVATE_AI_TOOL_REFERENCE = re.compile(
    r"""(?ix)
    (?:
        (?:^|[/\s"'`])\.(?:agents|codex|cursor|kiro|openai|windsurf)(?:/|\\)
        |
        \b(?:agents|claude|copilot-instructions|gemini|skill)\.md\b
    )
    """
)

PRIVATE_OPERATIONAL_PATTERNS = (
    (
        "internal-kubernetes-host",
        re.compile(r"\b[a-z0-9.-]+\.svc\.cluster\.local\b", re.IGNORECASE),
    ),
    (
        "private-evaluator-host",
        re.compile(
            r"\bevaluator\.platform\." + r"olimpiada-ai\.ro\b",
            re.IGNORECASE,
        ),
    ),
    (
        "private-container-registry",
        re.compile(r"\bghcr\.io/" + r"ro-oai/", re.IGNORECASE),
    ),
)

LITERAL_CREDENTIAL = re.compile(
    r"""(?ix)
    \b(password|passwd|api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret)
    \b[^\n]{0,40}?
    (?:default\s*=\s*|[:=]\s*)
    ["']([^"']*)["']
    """
)

ACTION_REFERENCE = re.compile(r"(?m)^\s*(?:-\s*)?uses:\s*([^\s#]+)")
PINNED_ACTION_REFERENCE = re.compile(r"^[^@\s]+@[0-9a-fA-F]{40}$")


def _matches(path: str, patterns: Iterable[str]) -> bool:
    pure_path = PurePosixPath(path)
    if not pure_path.parts:
        return False

    for pattern in patterns:
        pattern_path = PurePosixPath(pattern)
        if not pattern_path.parts or pure_path.parts[0] != pattern_path.parts[0]:
            continue
        if fnmatch.fnmatchcase(path, pattern) or pure_path.match(pattern):
            return True
    return False


def _path_issue(relative: str) -> str | None:
    lowered = relative.lower()
    path = PurePosixPath(lowered)
    if (
        relative != relative.strip()
        or "\\" in relative
        or any(ord(character) < 32 or ord(character) == 127 for character in relative)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return "Unsafe public path is forbidden"
    if not _matches(relative, ALLOWED_PATTERNS):
        return f"Path is outside the public allowlist: {relative}"
    if relative in FORBIDDEN_EXACT_PATHS:
        return f"Private path is forbidden: {relative}"
    if lowered.startswith(FORBIDDEN_PREFIXES):
        return f"Private path is forbidden: {relative}"
    if path.name.startswith(".env"):
        return f"Environment file is forbidden: {relative}"
    if path.name in FORBIDDEN_BASENAMES:
        return f"Private or legal-policy file is forbidden: {relative}"
    if path.suffix in FORBIDDEN_SUFFIXES:
        return f"Binary, archive, database, or credential file is forbidden: {relative}"
    return None


def _content_issues(relative: str, content: bytes) -> list[str]:
    issues: list[str] = []
    if b"\x00" in content:
        return [f"NUL-bearing or UTF-16 content is forbidden: {relative}"]
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [f"Unexpected binary file: {relative}"]
    if any(
        ord(character) < 32 and character not in "\n\r\t"
        for character in text
    ) or "\x7f" in text:
        return [f"Control characters are forbidden: {relative}"]

    for rule, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            issues.append(f"Potential {rule} detected in {relative}")
    for rule, pattern in PRIVATE_OPERATIONAL_PATTERNS:
        if pattern.search(text):
            issues.append(f"Private operational value ({rule}) detected in {relative}")
    for match in URL_USERINFO.finditer(text):
        userinfo = (unquote(match.group(1)), unquote(match.group(2)))
        if not (
            relative.startswith("tests/")
            and userinfo in EXPLICIT_TEST_URL_USERINFO
        ):
            issues.append(f"Credential-bearing URL detected in {relative}")
    projected_source = (
        relative.startswith(("app/", "tests/", "test_data/", "test_payloads/"))
        or relative in {"poetry.lock", "pyproject.toml"}
    )
    if projected_source and PRIVATE_AI_TOOL_REFERENCE.search(text):
        issues.append(f"Private AI/tool configuration reference detected in {relative}")

    if relative.startswith("app/"):
        credential_matches = LITERAL_CREDENTIAL.finditer(text)
        if any(match.group(2).strip() for match in credential_matches):
            issues.append(f"Literal credential default detected in {relative}")
    if relative.startswith(".github/workflows/"):
        for match in ACTION_REFERENCE.finditer(text):
            if not PINNED_ACTION_REFERENCE.fullmatch(match.group(1)):
                issues.append(
                    f"Workflow action is not pinned to a full commit SHA: {relative}"
                )
    return issues


def _git_mode_issues(root: Path) -> list[str]:
    """Validate index modes when ``root`` is a Git checkout.

    Filesystem inspection alone cannot reliably distinguish a submodule from an
    empty directory after checkout. The index check rejects gitlinks, symlinks,
    executable bits, and unresolved merge stages before any candidate is used.
    """

    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return []

    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=False,
        capture_output=True,
    )
    if listing.returncode != 0:
        return ["Could not inspect Git index modes"]

    issues: list[str] = []
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, _object_id, raw_stage = metadata.split(b" ", 2)
            relative = raw_path.decode("utf-8")
            mode = raw_mode.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (UnicodeDecodeError, ValueError):
            issues.append("Malformed Git index entry")
            continue
        if stage != "0":
            issues.append(f"Unresolved Git index entry is forbidden: {relative}")
        if mode != ALLOWED_GIT_MODE:
            issues.append(f"Unexpected Git object mode is forbidden: {relative}")
    return issues


def validate_tree(root: Path) -> list[str]:
    root = root.resolve()
    issues: list[str] = []
    if not root.is_dir():
        return ["Public-tree root is not a directory"]

    for required in sorted(REQUIRED_PATHS):
        if not (root / required).is_file():
            issues.append(f"Required public file is missing: {required}")
    issues.extend(_git_mode_issues(root))
    for candidate in sorted(root.rglob("*")):
        if ".git" in candidate.relative_to(root).parts:
            continue
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            issues.append(f"Symlink is forbidden: {relative}")
            continue
        if candidate.is_dir():
            continue
        mode = candidate.stat().st_mode
        if not stat.S_ISREG(mode):
            issues.append(f"Non-regular file is forbidden: {relative}")
            continue
        if stat.S_IMODE(mode) != 0o644:
            issues.append(f"Unexpected file mode is forbidden: {relative}")
            continue
        path_issue = _path_issue(relative)
        if path_issue:
            issues.append(path_issue)
            continue
        size = candidate.stat().st_size
        if size > MAX_FILE_BYTES:
            issues.append(f"File exceeds the public size limit: {relative}")
            continue
        issues.extend(_content_issues(relative, candidate.read_bytes()))
    return issues


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    issues = validate_tree(_parse_args().root)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    print("Public repository boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
