import ast
import tomllib
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (SERVICE_ROOT / "src", SERVICE_ROOT / "scripts")
FORBIDDEN_MODULE_FRAGMENTS = (
    "alpaca",
    "broker",
    "hyprl.crypto",
    "hyprl.broker",
)
FORBIDDEN_IDENTIFIERS = {
    "CryptoTrader",
    "submit_order",
    "close_position",
    "place_order",
}
FORBIDDEN_NETWORK_MODULES = {
    "aiohttp",
    "httpx",
    "requests",
    "socket",
    "urllib.request",
    "websockets",
}
NETWORK_IMPORT_ALLOWLIST = {
    "urllib.request": SERVICE_ROOT / "src" / "crypto_news" / "transport.py",
}


def production_python_files() -> list[Path]:
    return sorted(
        path
        for root in SOURCE_ROOTS
        if root.exists()
        for path in root.rglob("*.py")
    )


def test_ast_has_no_execution_or_broker_symbols() -> None:
    violations: list[str] = []
    for path in production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                lowered = name.lower()
                if any(fragment in lowered for fragment in FORBIDDEN_MODULE_FRAGMENTS):
                    violations.append(f"{path}:{node.lineno}: import {name}")
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{path}:{node.lineno}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{path}:{node.lineno}: attribute {node.attr}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in FORBIDDEN_IDENTIFIERS:
                    violations.append(f"{path}:{node.lineno}: dynamic symbol")
    assert violations == []


def test_network_client_is_confined_and_environment_secret_access_is_absent() -> None:
    violations: list[str] = []
    allowed_imports_seen: set[tuple[str, Path]] = set()
    for path in production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                if name in FORBIDDEN_NETWORK_MODULES:
                    allowed_path = NETWORK_IMPORT_ALLOWLIST.get(name)
                    if path != allowed_path:
                        violations.append(f"{path}:{node.lineno}: network import {name}")
                    else:
                        allowed_imports_seen.add((name, path))
            if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
                violations.append(f"{path}:{node.lineno}: environment access {node.attr}")
            if isinstance(node, ast.Name) and node.id == "getenv":
                violations.append(f"{path}:{node.lineno}: environment access getenv")
    assert violations == []
    assert allowed_imports_seen == set(NETWORK_IMPORT_ALLOWLIST.items())


def test_separate_pyproject_has_no_execution_dependency() -> None:
    payload = tomllib.loads((SERVICE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_groups = [payload["project"].get("dependencies", [])]
    dependency_groups.extend(payload["project"].get("optional-dependencies", {}).values())
    dependencies = [item.lower() for group in dependency_groups for item in group]

    assert dependencies
    assert not any(
        fragment in dependency
        for dependency in dependencies
        for fragment in ("alpaca", "broker", "hyprl")
    )
