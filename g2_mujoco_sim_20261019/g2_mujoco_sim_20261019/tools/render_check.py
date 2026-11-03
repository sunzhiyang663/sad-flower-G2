"""Render one frame from bundled MJCF/meshes, without a desktop or robot IO."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from simulation.replay_extraction import configure_graphics
configure_graphics(False)
import mujoco,numpy as np,yaml
from PIL import Image
from simulation.map9_route_preview import Clearance
from simulation.full_task_config import load_task
cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task()
model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/map9_current.xml'))
saved=dict(np.load(ROOT/task['output']/'state.npz'))
c=Clearance(model,saved,cfg['base_height_m'],.1)
camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
camera.lookat[:]=[*saved['initial_base'][:2,3],.9];camera.distance=3.5;camera.azimuth=140;camera.elevation=-25
option=mujoco.MjvOption();option.geomgroup[:]=[0,1,1,1,1,0]
with mujoco.Renderer(model,height=540,width=960) as renderer:
    renderer.update_scene(c.data,camera=camera,scene_option=option)
    rgb=renderer.render()
    assert rgb.std()>3, 'Unexpected blank frame'
    target=ROOT/'verification/render_check.png';target.parent.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(target)
print('PASS: bundled MJCF rendered to',target)
