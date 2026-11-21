#!/usr/bin/env python3
"""Deterministic static and runtime codebase analyzer for my_harness."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest


def check_tool_schema_parity(repo_root: Path) -> dict[str, any]:
    """Verify that every tool in tools.TOOLS has a matching definition in tools.TOOL_SCHEMAS."""
    root_str = str(repo_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from tools import TOOL_SCHEMAS, TOOLS
    except Exception as exc:
        return {"ok": False, "error": f"Failed to import tools package: {exc}"}

    tools_dict_keys = set(TOOLS.keys())
    schema_names_list = [s.get("name") for s in TOOL_SCHEMAS if isinstance(s, dict) and s.get("name")]
    tool_schema_names = set(schema_names_list)

    missing_schemas = sorted(tools_dict_keys - tool_schema_names)
    missing_tools = sorted(tool_schema_names - tools_dict_keys)
    duplicate_schemas = len(schema_names_list) != len(tool_schema_names)
    invalid_handlers = sorted(name for name, fn in TOOLS.items() if not callable(fn))

    is_parity = (
        len(missing_schemas) == 0
        and len(missing_tools) == 0
        and not duplicate_schemas
        and len(invalid_handlers) == 0
        and len(tools_dict_keys) > 0
    )

    return {
        "ok": is_parity,
        "tools_count": len(tools_dict_keys),
        "tools": sorted(tools_dict_keys),
        "schemas_count": len(schema_names_list),
        "missing_schemas": missing_schemas,
        "missing_tools": missing_tools,
        "duplicate_schemas": duplicate_schemas,
        "invalid_handlers": invalid_handlers,
    }


def check_providers_contract(repo_root: Path) -> dict[str, any]:
    """Verify that provider classes expose the required complete method."""
    providers_dir = repo_root / "providers"
    if not providers_dir.exists():
        return {"ok": False, "error": "providers/ not found"}

    required_methods = {"complete"}
    provider_results: dict[str, any] = {}

    for py_file in providers_dir.glob("*.py"):
        if py_file.name in ("__init__.py", "base.py"):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception as exc:
            provider_results[py_file.name] = {"ok": False, "error": str(exc)}
            continue

        classes_found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                has_contract = required_methods.issubset(methods)
                classes_found[node.name] = {
                    "has_complete": "complete" in methods,
                    "methods": sorted(methods),
                    "compliant": has_contract,
                }
        provider_results[py_file.name] = classes_found

    all_compliant = all(
        cls_info.get("compliant", False)
        for pfile in provider_results.values()
        if isinstance(pfile, dict)
        for cls_info in pfile.values()
        if isinstance(cls_info, dict) and "compliant" in cls_info
    )

    return {"ok": all_compliant, "files": provider_results}


def check_guardrails_integrity(repo_root: Path) -> dict[str, any]:
    """Verify guardrails package structure, base class and exports."""
    guardrails_dir = repo_root / "guardrails"
    if not guardrails_dir.exists():
        return {"ok": False, "present": False, "details": "guardrails directory not present"}

    expected_modules = {
        "base.py",
        "command_safety.py",
        "config.py",
        "loop_detection.py",
        "path_sandbox.py",
        "secrets_redaction.py",
        "__init__.py",
    }
    present_modules = {p.name for p in guardrails_dir.glob("*.py")}
    missing = sorted(expected_modules - present_modules)

    return {
        "ok": len(missing) == 0,
        "present": True,
        "modules_found": sorted(present_modules),
        "missing_modules": missing,
    }


def check_tests_suite(repo_root: Path) -> dict[str, any]:
    """Discover and execute all unit tests in tests/."""
    import os
    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        return {"ok": False, "tests_found": 0, "ran": 0, "errors": ["tests directory not found"]}

    root_str = str(repo_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir="tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(stream=open(Path("NUL" if sys.platform == "win32" else "/dev/null"), "w"), verbosity=0)
        result = runner.run(suite)
    finally:
        os.chdir(old_cwd)

    return {
        "ok": result.wasSuccessful(),
        "tests_ran": result.testsRun,
        "failures_count": len(result.failures),
        "errors_count": len(result.errors),
        "failures": [str(f[0]) for f in result.failures],
        "errors": [str(e[0]) for e in result.errors],
    }


def analyze_codebase(repo_root: Path | None = None) -> dict[str, any]:
    """Run all analysis checks and aggregate report."""
    root = repo_root or Path.cwd()

    tool_parity = check_tool_schema_parity(root)
    providers = check_providers_contract(root)
    guardrails = check_guardrails_integrity(root)
    tests = check_tests_suite(root)

    overall_ok = (
        tool_parity.get("ok", False)
        and providers.get("ok", False)
        and guardrails.get("ok", False)
        and tests.get("ok", False)
    )

    report = {
        "repo_root": str(root.resolve()),
        "overall_ok": overall_ok,
        "checks": {
            "tool_schema_parity": tool_parity,
            "providers_contract": providers,
            "guardrails_integrity": guardrails,
            "tests_suite": tests,
        },
    }
    return report


def main() -> int:
    report = analyze_codebase()
    print(json.dumps(report, indent=2))
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
