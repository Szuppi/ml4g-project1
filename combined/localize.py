"""Localize the Kaggle-wrapped reproduction runners so they run here, bodies verbatim.

Strategy: the reproduction's stage_kaggle_inputs.py builds the exact input layout the runners expect
(prediction/ training/ upgrade/ rich/ + SHA manifests). Then each runner needs only:
  1. /kaggle/input  -> the staged dir
  2. /kaggle/working -> a local work dir
  3. the wheel `pip install` subprocess -> a no-op (we use the conda env)
  4. trim its split/variant loop to just the held-out (joint == D1) validation fit,
     skipping the wasted x1_to_x2 / x2_to_x1 / production fits.
Everything else -- the model, targets, loss, SWA, calibration -- is left byte-identical.

A unified diff of every change is printed so nothing slips through unseen.

Usage:  python combined/localize.py <member> [<member> ...]   (or 'all')
"""
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reproduction" / "training"
OUT = ROOT / "combined" / "ensemble_runners"
STAGED = ROOT / "combined" / "work" / "staged"
WORK = ROOT / "combined" / "work" / "ensemble"

PIP = re.compile(r"subprocess\.run\(\[sys\.executable,'-m','pip','install'.*?\],check=True\)")

# Per-member loop trims: keep only the single held-out validation fit on D1.
TRIM = {
    "cnn": [("fit(output/'production',train,test,True,selected)",
             "None  # [localized] skip production; ensemble X3 from committed artifacts")],
    "tree": [("cnn_fit(dest,train,test,True,production=True,selection=selected)",
              "None  # [localized] skip production"),
             ("tree_fit(dest,train,test,True,production=True,selection=selected)",
              "None  # [localized] skip production")],
    "extra": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "tree_log": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "basic": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "combo": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "seed43": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "seed44": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "wide": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "wide_combo": [("['x1_to_x2','x2_to_x1','joint','production']", "['joint']")],
    "weighted": [("['source1','source2','weight1','weight2']", "['weight1']"),
                 ("for split in ['production']", "for split in ['development']")],
    "variants": [("['control','gain','shape','dilated']", "['gain','dilated']"),
                 ("['development','production']", "['development']")],
}


def localize(member):
    original = (SRC / f"{member}.py").read_text()
    work = WORK / member
    text = original
    text = text.replace("Path('/kaggle/input')", f"Path(r'{STAGED}')")
    text = text.replace("Path('/kaggle/working')", f"Path(r'{work}')")
    n_pip = len(PIP.findall(text))
    text = PIP.sub("None  # [localized] pip install skipped (conda env)", text)
    trims = TRIM.get(member, [])
    for old, new in trims:
        if old not in text:
            raise SystemExit(f"[{member}] trim target missing: {old!r}")
        text = text.replace(old, new)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{member}.py").write_text(text)
    diff = [d for d in difflib.unified_diff(
        original.splitlines(), text.splitlines(), lineterm="")
        if d and d[0] in "+-" and not d.startswith(("+++", "---"))]
    print(f"[{member}] pip-skipped={n_pip} trims={len(trims)} changed-lines={len(diff)}")
    for d in diff:
        print("    " + d)
    print()


if __name__ == "__main__":
    members = list(TRIM) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for mem in members:
        localize(mem)
