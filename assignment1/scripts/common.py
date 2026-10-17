"""Shared helpers for the Assignment 1 scripts (standard library only)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
EXE = ".exe" if os.name == "nt" else ""


def load_config():
    return json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8-sig"))


def build_info():
    return json.loads((BUILD / "build-info.json").read_text(encoding="utf-8-sig"))


def kernel_command(language, variant=None, python=None, java=None):
    """Command prefix that runs one implementation; append the CLI arguments.

    language: 'python' | 'java' | 'c'. `variant` is only meaningful for C (stem of a build in build/c, default 'matmul').
    """
    cfg = load_config()
    if language == "python":
        return [python or sys.executable, str(ROOT / "src" / "python" / "matmul.py")]
    if language == "java":
        exe = java or build_info()["java"]
        return [exe, *cfg["java_flags"], "-cp", str(BUILD / "java"), "MatMul"]
    if language == "c":
        return [str(BUILD / "c" / ((variant or "matmul") + EXE))]
    raise ValueError(language)


def input_paths(n):
    return str(INPUTS / f"A_{n}.bin"), str(INPUTS / f"B_{n}.bin")
