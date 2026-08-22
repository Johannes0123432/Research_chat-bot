"""
Simple local Python code executor for scientific simulations.
WARNING: This executes arbitrary code. Only use in trusted environments.
For production, prefer Docker or a proper sandbox (e.g. E2B, RestrictedPython + limits).
"""

from __future__ import annotations
import sys
import io
import traceback
import contextlib
from typing import Dict, Any, Optional, Tuple
import ast


# Allowed built-ins (keep relatively safe)
# Note: __import__ is required for normal "import" statements to work.
SAFE_BUILTINS = {
    "__import__": __import__,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "map": map,
    "filter": filter,
    "isinstance": isinstance,
    "type": type,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "RuntimeError": RuntimeError,
    "True": True,
    "False": False,
    "None": None,
}


def execute_code(
    code: str,
    timeout_seconds: int = 30,
    extra_globals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute Python code and capture stdout, stderr, and a result variable if present.

    Returns a dict with:
      - success: bool
      - stdout: str
      - stderr: str
      - error: Optional[str]
      - result: Any (if the code assigned to a variable named `result`)
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Restricted globals
    restricted_globals = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__simulation__",
    }

    # Pre-import common + selected trusted scientific libraries
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats, integrate, optimize, signal, ndimage
        import sympy as sp
        import networkx as nx
        from sklearn import datasets, metrics, model_selection, preprocessing
        import statsmodels.api as sm

        restricted_globals.update({
            "np": np,
            "numpy": np,
            "pd": pd,
            "pandas": pd,
            "plt": plt,
            "matplotlib": matplotlib,
            "sns": sns,
            "seaborn": sns,
            "stats": stats,
            "integrate": integrate,
            "optimize": optimize,
            "signal": signal,
            "ndimage": ndimage,
            "sp": sp,
            "sympy": sp,
            "nx": nx,
            "networkx": nx,
            "sm": sm,
            "statsmodels": sm,
        })
    except ImportError:
        pass

    # Optional trusted packages (fail gracefully if not installed)
    optional_imports = [
        ("Bio", "Bio"),
        ("numba", "numba"),
        ("lmfit", "lmfit"),
        ("pint", "pint"),
        ("emcee", "emcee"),
        ("astropy", "astropy"),
    ]
    for mod_name, alias in optional_imports:
        try:
            mod = __import__(mod_name)
            restricted_globals[alias] = mod
        except ImportError:
            pass

    if extra_globals:
        restricted_globals.update(extra_globals)

    result_data: Dict[str, Any] = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "error": None,
        "result": None,
        "figures": [],  # reserved for future plot capture
    }

    try:
        # Basic syntax check first
        ast.parse(code)

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Execute in the restricted namespace
            exec(code, restricted_globals)

            # Try to pull a variable named `result` if the user/LLM created one
            if "result" in restricted_globals:
                result_data["result"] = restricted_globals["result"]

        result_data["success"] = True
        result_data["stdout"] = stdout_capture.getvalue()
        result_data["stderr"] = stderr_capture.getvalue()

    except Exception as e:
        result_data["success"] = False
        result_data["stdout"] = stdout_capture.getvalue()
        result_data["stderr"] = stderr_capture.getvalue()
        result_data["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    return result_data


def format_execution_result(exec_result: Dict[str, Any]) -> str:
    """Turn the execution result into a clean string for the LLM / UI."""
    parts = []

    if exec_result.get("stdout"):
        parts.append("=== STDOUT ===\n" + exec_result["stdout"].strip())

    if exec_result.get("stderr"):
        parts.append("=== STDERR ===\n" + exec_result["stderr"].strip())

    if exec_result.get("error"):
        parts.append("=== ERROR ===\n" + exec_result["error"].strip())

    if exec_result.get("result") is not None:
        parts.append("=== RESULT VARIABLE ===\n" + str(exec_result["result"]))

    if not parts:
        parts.append("(No output produced)")

    status = "SUCCESS" if exec_result.get("success") else "FAILED"
    return f"Execution status: {status}\n\n" + "\n\n".join(parts)
