"""Grid + whole-body scene validity shared by expert collection and tests."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from simulation.map9_route_preview import Clearance, occupied_footprint

ROOT = Path(__file__).resolve().parents[2]


class CollectionScene:
    def __init__(self, mode, clearance=.003):
        if mode not in ('empty','loaded') or not 0 < clearance < .1:
            raise ValueError('Invalid mode or clearance (must be between 0 and .1 m)')
        self.mode, self.clearance = mode, clearance
        inputs = ROOT/'simulation/flow/data/collection_inputs'
        map_root = ROOT/'simulation/reference_inputs/map9_scene'
        self.scene_path = ROOT/'scenes/map9_current.xml'
        self.input_paths = [self.scene_path,inputs/'robot_states.npz',inputs/'grid.json',
                            map_root/'scene_map.npz',map_root/'native_filtered.yaml',ROOT/'simulation/map9.yaml']
        for mesh in ET.parse(self.scene_path).getroot().iter('mesh'):
            if 'file' in mesh.attrib:
                self.input_paths.append((self.scene_path.parent/mesh.get('file')).resolve())
        cfg = yaml.safe_load((ROOT/'simulation/map9.yaml').read_text(encoding='utf-8'))
        self.bounds = cfg['work_bounds_xy_m']
        self.meta = json.loads((inputs/'grid.json').read_text(encoding='utf-8'))
        if not np.allclose(self.meta['origin']['orientation_xyzw'],[0,0,0,1]):
            raise ValueError('Rotated occupancy grids are not supported')
        self.resolution = self.meta['resolution_m']
        self.origin = np.array(self.meta['origin']['position'][:2])
        with np.load(map_root/'scene_map.npz') as record:
            self.grid = record['data'].copy()
        if self.grid.shape != (self.meta['height'],self.meta['width']):
            raise ValueError('Grid metadata dimensions mismatch')
        self.footprint = np.array(yaml.safe_load((map_root/'native_filtered.yaml').read_text())['robot']['footprint'])
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        with np.load(inputs/'robot_states.npz') as record:
            saved = dict(qpos=record[f'{mode}_qpos'].copy(),initial_base=record['initial_base'].copy(),
                         T_base_payload=record['loaded_payload'].copy())
            self.shell_world = record['shell_world'].copy()
        self.checker = Clearance(self.model,saved,cfg['base_height_m'],.1)
        # For empty motion the object remains at its original shelf pose.
        self.shell_quat = Rotation.from_matrix(self.shell_world[:3,:3]).as_quat(scalar_first=True)
        yy,xx = np.nonzero(self.grid==0)
        positions = np.column_stack([xx,yy])*self.resolution+self.origin+self.resolution/2
        x0,x1,y0,y1 = self.bounds
        self.free_centers = positions[(positions[:,0]>=x0)&(positions[:,0]<=x1)&(positions[:,1]>=y0)&(positions[:,1]<=y1)]
        if not len(self.free_centers):
            raise ValueError('No free cells in work bounds')

    def inspect(self,p):
        if not np.isfinite(p).all():
            return False,'nonfinite',None
        x0,x1,y0,y1 = self.bounds
        if not (x0<=p[0]<=x1 and y0<=p[1]<=y1):
            return False,'bounds',None
        # Check every footprint, rather than relying on the old .52 m shortcut.
        if occupied_footprint(p,self.footprint,.01,self.resolution,self.origin,self.grid):
            return False,'occupancy_footprint',None
        values,_ = self.checker.check(p,include_payload=self.mode=='loaded')
        minimum = float(values.min())
        if minimum < self.clearance:
            return False,'environment_clearance',minimum
        if self.mode=='empty':
            self.checker.data.mocap_pos[0] = self.shell_world[:3,3]
            self.checker.data.mocap_quat[0] = self.shell_quat
            mujoco.mj_forward(self.model,self.checker.data)
            minimum = min(minimum,min(float(mujoco.mj_geomDistance(self.model,self.checker.data,
                g,self.checker.payload,.1,None)) for g in self.checker.robot))
            if minimum < self.clearance:
                return False,'stationary_payload_clearance',minimum
        return True,'valid',minimum

    def valid(self,p):
        return self.inspect(p)[0]

    def sample_pose(self,rng,budget=200):
        for _ in range(budget):
            position = self.free_centers[rng.integers(len(self.free_centers))]
            p = np.r_[position,rng.uniform(-np.pi,np.pi)]
            if self.valid(p):
                return p
        raise ValueError('Endpoint sampling budget exhausted')

    def audit(self,states,translation_step=.01,angle_step=.02):
        checked,minimum = 0,.1
        for index,(a,b) in enumerate(zip(states[:-1],states[1:])):
            count = max(1,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/translation_step)),
                        int(np.ceil(abs(b[2]-a[2])/angle_step)))
            for t in np.linspace(0.,1.,count+1):
                ok,reason,distance = self.inspect(a+(b-a)*t)
                checked += 1
                if distance is not None:
                    minimum = min(minimum,distance)
                if not ok:
                    return dict(passed=False,reason=reason,segment=index,fraction=float(t),samples=checked,minimum_m=minimum)
        return dict(passed=True,samples=checked,minimum_m=minimum)
