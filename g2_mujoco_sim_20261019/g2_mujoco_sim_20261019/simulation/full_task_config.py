"""Select an offline transfer profile without replacing historical profiles."""
import os
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def load_task():
    filename=Path(os.environ.get('G2_SIM_TASK','simulation/map9_spread_forward5_full.yaml'))
    return yaml.safe_load((ROOT/filename).read_text())
