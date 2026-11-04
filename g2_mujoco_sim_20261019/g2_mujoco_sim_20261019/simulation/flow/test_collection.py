"""Collection control-flow tests plus real grid/scene preflight (no OMPL)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulation.flow.generate_experts import collect
from simulation.flow.collection_scene import CollectionScene
from simulation.flow.dataset import G2TrajectoryDataset


class FakeScene:
    def sample_pose(self,rng):
        return np.r_[rng.uniform(0,4,2),0.]

    def valid(self,p):
        return True

    def audit(self,*args):
        return dict(passed=True,samples=20,minimum_m=.1)


class CollectionTests(unittest.TestCase):
    def test_resume_and_dataset(self):
        settings = dict(count=3,seed=77,dt=.1,speed=.45,reverse_speed=.25,yaw_speed=.45,
            min_distance=.001,max_distance=100.,min_states=2,max_duration=120.,
            translation_step=.01,angle_step=.02,mode='empty')
        calls = [0]
        def plan(a,b,valid):
            calls[0] += 1
            if calls[0]==1:
                raise RuntimeError('test_planner_failure')
            return np.linspace(a,b,20)
        with tempfile.TemporaryDirectory() as temp:
            first = collect(FakeScene(),plan,settings,temp,max_attempts=2)
            self.assertFalse(first['complete'])
            self.assertEqual(first['accepted'],1)
            self.assertEqual(first['statistics']['rejection_reasons'],{'test_planner_failure':1})
            final = collect(FakeScene(),plan,settings,temp,resume=True,max_attempts=8)
            self.assertTrue(final['complete'])
            self.assertEqual(final['statistics']['split_counts'],{'train':1,'validation':1,'test':1})
            for split in ('train','validation','test'):
                ds = G2TrajectoryDataset(temp,split,horizon=2)
                self.assertGreater(len(ds),0)
            with self.assertRaises(ValueError):
                collect(FakeScene(),plan,dict(settings,dt=.2),temp,resume=True)
            episode = Path(temp)/final['episodes'][0]['file']
            episode.write_bytes(b'corruption test')
            with self.assertRaises(ValueError):
                collect(FakeScene(),plan,settings,temp,resume=True)

    def test_real_scenes(self):
        for mode in ('empty','loaded'):
            scene = CollectionScene(mode)
            self.assertFalse(scene.valid(np.array([-1000.,-1000.,0.])))
            yy,xx = np.nonzero(scene.grid!=0)
            centers = np.column_stack([xx,yy])*scene.resolution+scene.origin+scene.resolution/2
            x0,x1,y0,y1 = scene.bounds
            inside = centers[(centers[:,0]>x0)&(centers[:,0]<x1)&(centers[:,1]>y0)&(centers[:,1]<y1)]
            blocked = np.r_[inside[0],0.]
            self.assertEqual(scene.inspect(blocked)[1],'occupancy_footprint')
            pose = scene.sample_pose(np.random.default_rng(77))
            self.assertTrue(scene.valid(pose))
            self.assertTrue(scene.audit(np.array([pose,pose]))['passed'])


if __name__ == '__main__':
    unittest.main()
