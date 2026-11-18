import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Real data root (partition folders train/ and student_test/). Override with env var.
DATA_ROOT = Path(os.environ.get(
    "PATHSNAP_DATA_DIR",
    ROOT.parent.parent / "PathSnap_student_materials" / "Student folder"))
