#!/usr/bin/env python3
"""Lightweight test runner (no pytest required)."""

import importlib
import inspect
import sys
import traceback


def run_module_tests(module_name: str) -> int:
    """Run all test_* functions in a module."""
    module = importlib.import_module(module_name)
    failures = 0
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            try:
                func()
                print(f"  PASS  {module_name}.{name}")
            except Exception as e:
                failures += 1
                print(f"  FAIL  {module_name}.{name}")
                traceback.print_exc()
    return failures


def main() -> int:
    total_failures = 0
    modules = ["tests.test_regression_pipeline", "tests.test_classification_pipeline"]

    for mod in modules:
        print(f"\n{mod}")
        total_failures += run_module_tests(mod)

    print(f"\n{'=' * 50}")
    if total_failures == 0:
        print("All tests passed!")
        return 0
    else:
        print(f"{total_failures} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
