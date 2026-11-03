"""Check bundled models, kinematics and saved trajectories without robot IO."""
from pathlib import Path
import os,sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from simulation.refine_grasp_elbow_down import load
_,checker,_,joints=load(ROOT/'simulation/grasp_wrist_parallel_front.yaml')
assert len(joints)==46
model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/map9_current.xml'))
for goal in (1,5):subprocess.run([sys.executable,'tools/replay.py','--goal',str(goal),'--check-only'],check=True)
manifest=ROOT/'MANIFEST.sha256'
if manifest.exists():
    for line in manifest.read_text().splitlines():
        digest,path=line.split('  ',1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
print('PASS: URDF/meshes/SRDF, 46 controlled joints, editable MJCF and G1/G5 replay; no hardware IO.')
print('Package hashes verified.' if manifest.exists() else 'Package manifest not generated yet.')
