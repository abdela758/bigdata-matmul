"""Create dist/Assignment01_Choukri_Materials.zip: source, scripts, configuration, raw results, tables, figures, report.

Large generated inputs (inputs/*.bin) and build output are not included; inputs/checksums.json lets anyone regenerate and
verify them. Run after analyze.py and build_report.py.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
NAME = "Assignment01_Choukri_Materials"
INCLUDE = ["README.md", "src", "scripts", "config", "jmh", "results", "report", "inputs/checksums.json", "../requirements.txt"]
SKIP_PARTS = {"__pycache__", "target", "build"}
SKIP_SUFFIX = {".pyc", ".bin"}


def main():
    DIST.mkdir(exist_ok=True)
    out = DIST / f"{NAME}.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = (ROOT / item).resolve()
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                if SKIP_PARTS & set(f.parts) or f.suffix in SKIP_SUFFIX or f.stem.endswith("_pilot") or "_pilot" in f.name:
                    continue
                rel = f.relative_to(ROOT.parent) if ROOT.parent in f.parents else f.relative_to(ROOT)
                z.write(f, f"{NAME}/{rel.as_posix()}")
                count += 1
    print(f"{count} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
