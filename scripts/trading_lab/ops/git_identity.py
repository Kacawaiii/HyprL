"""Which commit is this, read from .git without running git.

No subprocess: a release builder that shells out inherits whatever ``git``
happens to be on PATH and whatever that binary decides about safe.directory,
and it fails differently on a machine that has no git at all.

The case that matters here is the **worktree**. In a linked worktree ``.git``
is a file containing ``gitdir: <path>``, not a directory, and its refs live
in the shared object store reached through ``commondir``. Code that assumes
``.git/HEAD`` exists silently reports "unknown commit" for every build made
from a worktree -- which is exactly how this project is developed, so the
release manifests would all have carried a null identity.
"""

from __future__ import annotations

import pathlib


def _git_directory(root: pathlib.Path):
    """The real git directory, following a worktree pointer if there is one."""
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except OSError:                                  # pragma: no cover
        return None
    if not text.startswith("gitdir:"):
        return None
    target = pathlib.Path(text.split(":", 1)[1].strip())
    if not target.is_absolute():
        target = (root / target).resolve()
    return target if target.is_dir() else None


def _common_directory(git_dir: pathlib.Path) -> pathlib.Path:
    """Where shared refs live. In a worktree that is not the worktree's own dir."""
    pointer = git_dir / "commondir"
    if not pointer.is_file():
        return git_dir
    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except OSError:                                  # pragma: no cover
        return git_dir
    candidate = pathlib.Path(text)
    if not candidate.is_absolute():
        candidate = (git_dir / candidate).resolve()
    return candidate if candidate.is_dir() else git_dir


def _resolve_ref(reference: str, git_dir: pathlib.Path,
                 common: pathlib.Path):
    for base in (git_dir, common):
        target = base / reference
        if target.is_file():
            try:
                return target.read_text(encoding="utf-8").strip()[:40]
            except OSError:                          # pragma: no cover
                continue
    for base in (git_dir, common):
        packed = base / "packed-refs"
        if not packed.is_file():
            continue
        try:
            lines = packed.read_text(encoding="utf-8").splitlines()
        except OSError:                              # pragma: no cover
            continue
        for line in lines:
            if line.endswith(f" {reference}"):
                return line.split(" ", 1)[0][:40]
    return None


def head_commit(root):
    """The current commit, or None when there is no readable git metadata."""
    root = pathlib.Path(root)
    git_dir = _git_directory(root)
    if git_dir is None:
        return None
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:                                  # pragma: no cover
        return None
    if not text.startswith("ref: "):
        return text[:40] or None
    return _resolve_ref(text[5:].strip(), git_dir, _common_directory(git_dir))
