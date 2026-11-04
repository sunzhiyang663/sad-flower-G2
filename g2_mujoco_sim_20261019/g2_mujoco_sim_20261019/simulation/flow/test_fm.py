"""Small CPU tests for FM masks, gradients and standalone sampling."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulation.flow.model import G2Transformer
from simulation.flow.flow_matcher import G2FlowMatcher
from simulation.flow.fm_inference import load_model, sample


class FMTests(unittest.TestCase):
    def test_mask_gradient(self):
        prediction = torch.nn.Parameter(torch.zeros(2,8,6))
        x = torch.randn_like(prediction)
        mask = torch.ones_like(x)
        mask[:,-1,:3] = 0
        mask[:,[0,-1],3:] = 0
        loss,_ = G2FlowMatcher(lambda xt,t:prediction).loss(x,{0:x[:,0,3:],7:x[:,-1,3:]},mask)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[mask==0]==0).all())
        self.assertTrue((prediction.grad[mask==1]!=0).any())

    def test_transformer_backward(self):
        model = G2Transformer(horizon=8,width=16,depth=1,heads=4)
        output = model(torch.randn(2,8,6),torch.rand(2))
        self.assertEqual(tuple(output.shape),(2,8,6))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_sampling(self):
        model = G2Transformer(horizon=8,width=16,depth=1,heads=4)
        config = dict(horizon=8,width=16,depth=1,heads=4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            torch.save(dict(model_config=config,model=model.state_dict(),
                normalizer=dict(mean=[0.]*6,scale=[1.]*6),dt=.1),path)
            loaded,saved,_ = load_model(path)
            q,u = sample(loaded,saved,[1.,2.,.3],[2.,3.,.4],steps=3,candidates=2)
            q2,u2 = sample(loaded,saved,[1.,2.,.3],[2.,3.,.4],steps=3,candidates=2)
            self.assertEqual(q.shape,(2,8,3))
            self.assertEqual(u.shape,(2,7,3))
            np.testing.assert_allclose(q[:,0],[[1.,2.,.3]]*2,atol=1e-6)
            np.testing.assert_allclose(q[:,-1],[[2.,3.,.4]]*2,atol=1e-6)
            np.testing.assert_array_equal(q,q2)
            np.testing.assert_array_equal(u,u2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
