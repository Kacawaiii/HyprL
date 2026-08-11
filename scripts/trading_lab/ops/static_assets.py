"""Serving the built frontend from one origin, without serving the repository.

A static file server is a function from an untrusted string to a filesystem
path, which is the shape of most path traversal bugs ever written. The
defence here is not a blocklist of suspicious substrings -- those get bypassed
by encodings nobody thought of -- but a containment check on the *resolved*
path: whatever the request decodes to, the real file it names must live under
the fixed dist root, or it is refused.

That check also handles the cases a substring filter misses:

* ``%2e%2e`` decodes to ``..`` and is caught after decoding, not before;
* ``%252e%252e`` decodes once to the literal text ``%2e%2e``, which is simply
  a filename that does not exist -- decoding twice is what would create the
  vulnerability, so it is done exactly once;
* an absolute path in the request is joined *relative* to the root, never
  used as a root of its own;
* a symlink pointing out of dist resolves outside the root and is refused,
  which is why the check runs on the resolved path rather than the joined one.

The API prefix is handled before any of this and never falls through to the
SPA. A ``/api/v1/typo`` that returned index.html would hand a JSON caller a
200 and a page of HTML, and the resulting bug report would blame the client.
"""

from __future__ import annotations

import mimetypes
import pathlib
from urllib.parse import unquote

API_PREFIX = "/api/"

# Vite writes content-hashed filenames into assets/. A hashed name is a new
# name whenever the bytes change, so it can be cached effectively forever.
IMMUTABLE_DIRECTORIES = ("assets",)
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
# The entry point is not hashed. It must be revalidated or an updated build is
# invisible to a browser that already has it.
INDEX_CACHE_CONTROL = "no-cache"
# Everything else served from dist: short, revalidated.
DEFAULT_CACHE_CONTROL = "public, max-age=300"

INDEX_FILE = "index.html"

# Never guessed from the filesystem; a fixed table avoids inheriting whatever
# a machine happens to have registered in /etc/mime.types.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".map": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}


class StaticAssetError(RuntimeError):
    """Raised when a request cannot be served from the dist root."""


class ForbiddenPathError(StaticAssetError):
    """Raised when a request resolves outside the dist root."""


def content_type(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix in CONTENT_TYPES:
        return CONTENT_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def cache_control(relative: str) -> str:
    parts = [segment for segment in relative.split("/") if segment]
    if parts and parts[0] in IMMUTABLE_DIRECTORIES:
        return IMMUTABLE_CACHE_CONTROL
    if not parts or parts[-1] == INDEX_FILE:
        return INDEX_CACHE_CONTROL
    return DEFAULT_CACHE_CONTROL


def is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith(API_PREFIX)


class StaticSite:
    """A frontend build, served from one resolved root and nothing else."""

    def __init__(self, dist_root):
        self.root = pathlib.Path(dist_root).resolve()

    @property
    def available(self) -> bool:
        return (self.root / INDEX_FILE).is_file()

    def index(self) -> pathlib.Path:
        target = self.root / INDEX_FILE
        if not target.is_file():
            raise StaticAssetError(
                "no frontend build found; run ./scripts/hyprl.sh build")
        return target

    def _contains(self, candidate: pathlib.Path) -> bool:
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return False
        return True

    def resolve(self, request_path: str) -> pathlib.Path:
        """Map a URL path to a real file under the root, or refuse.

        Raises ForbiddenPathError for anything that escapes, and
        FileNotFoundError for anything that simply is not there. The caller
        needs to tell those apart: one is a 403, the other feeds SPA fallback.
        """
        if is_api_path(request_path):
            raise ForbiddenPathError("API paths are never served from disk")
        decoded = unquote(request_path.split("?", 1)[0].split("#", 1)[0])
        if "\x00" in decoded:
            raise ForbiddenPathError("null byte in path")
        # Strip every leading slash so an absolute path cannot replace the
        # root during the join.
        relative = decoded.lstrip("/")
        if not relative:
            relative = INDEX_FILE
        # Windows-style separators would survive the POSIX checks below.
        if "\\" in relative:
            raise ForbiddenPathError("backslash in path")
        candidate = (self.root / relative)
        # strict=False: a missing file must still be containment-checked, so
        # that a 404 cannot be used to probe outside the root.
        resolved = pathlib.Path(candidate).resolve(strict=False)
        if not self._contains(resolved):
            raise ForbiddenPathError(f"{request_path!r} resolves outside the site root")
        if resolved.is_dir():
            resolved = resolved / INDEX_FILE
            if not self._contains(resolved.resolve(strict=False)):
                raise ForbiddenPathError("directory index escapes the site root")
        if not resolved.is_file():
            raise FileNotFoundError(request_path)
        # A symlink inside dist pointing elsewhere resolves outside the root
        # and is already refused above; this re-check covers the directory
        # index branch too.
        if not self._contains(resolved.resolve(strict=False)):
            raise ForbiddenPathError("symlink escapes the site root")
        return resolved

    def serve(self, request_path: str) -> dict:
        """Resolve a request to bytes plus headers, with SPA fallback.

        An unknown path that is not a file is the router's business, so it
        gets index.html and a 200 -- that is what makes a deep link work on
        reload. An unknown path that looks like an asset gets a real 404,
        because handing HTML to something expecting a script produces a
        console error that describes the wrong problem.
        """
        target = self.resolve(request_path)
        relative = target.relative_to(self.root).as_posix()
        return {
            "path": target,
            "relative": relative,
            "content_type": content_type(target),
            "cache_control": cache_control(relative),
            "status": 200,
        }

    def spa_fallback(self, request_path: str) -> dict:
        target = self.index()
        return {
            "path": target,
            "relative": INDEX_FILE,
            "content_type": CONTENT_TYPES[".html"],
            "cache_control": INDEX_CACHE_CONTROL,
            "status": 200,
            "fallback": True,
            "requested": request_path,
        }

    def looks_like_asset(self, request_path: str) -> bool:
        tail = request_path.split("?", 1)[0].rsplit("/", 1)[-1]
        return "." in tail

    def describe(self) -> dict:
        """Build metadata with no absolute path in it."""
        if not self.available:
            return {"available": False, "files": 0, "bytes": 0}
        files = [item for item in self.root.rglob("*") if item.is_file()]
        return {
            "available": True,
            "files": len(files),
            "bytes": sum(item.stat().st_size for item in files),
            "entry": INDEX_FILE,
            "cache_policy": {
                "hashed_assets": IMMUTABLE_CACHE_CONTROL,
                "entry_document": INDEX_CACHE_CONTROL,
                "other": DEFAULT_CACHE_CONTROL,
                "api": "no-store",
            },
        }
