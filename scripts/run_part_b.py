"""Part b on the real Freddie Mac sample. Exits BLOCKED-ON-TRISTAN until files are in data/raw/freddie/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frc.part_b import analyze  # noqa: E402

RAW = ROOT / "data/raw/freddie"
orig = sorted(RAW.rglob("sample_orig_*.txt"))
svcg = sorted(RAW.rglob("sample_perf_*.txt"))
if not orig or len(orig) != len(svcg):
    sys.exit(f"BLOCKED-ON-TRISTAN: need matched sample_orig_*/sample_perf_* pairs in {RAW} "
             f"(found {len(orig)} orig, {len(svcg)} perf). See 'Tristan's steps' in docs/PLAN.md. "
             "First verify the column positions in src/frc/freddie.py against the release's User Guide.")
met, contrasts = analyze(orig, svcg, ROOT / "outputs")
print(met.to_string(), contrasts.to_string(), sep="\n\n")
