"""Shared paths and helpers for the Assignment 2 scripts (standard library only)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
EXE = ".exe" if os.name == "nt" else ""
DENSE = str(BUILD / f"dense{EXE}")
SPARSE = str(BUILD / f"sparse{EXE}")


def load_config():
    return json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8-sig"))


def dense_inputs(n):
    """Part A inputs: uniform [0,1) dense matrices, same rule and seed as Assignment 1."""
    return str(INPUTS / f"A_{n}.bin"), str(INPUTS / f"B_{n}.bin")


def density_tag(d):
    return f"d{round(d * 1_000_000):07d}"  # parts per million, e.g. 1 % -> d0010000


def sparse_inputs(n, d, kind):
    """Part B inputs for one (n, density): kind 'bin' (dense layout with zeros) or 'csr'."""
    t = density_tag(d)
    return str(INPUTS / f"SA_{n}_{t}.{kind}"), str(INPUTS / f"SB_{n}_{t}.{kind}")
