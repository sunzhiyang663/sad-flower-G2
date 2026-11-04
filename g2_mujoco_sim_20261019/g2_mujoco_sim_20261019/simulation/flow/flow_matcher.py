"""G2 masked conditional flow matching, linear path with zero noise scale.

Adapted from the cfm training path in the local SAD-Flower reference.
Generation time tau is independent of physical trajectory time.
"""
import torch


class G2FlowMatcher:
    def __init__(self, model, action_dim=3):
        self.model = model
        self.action_dim = action_dim

    def loss(self, x, conditions, loss_mask):
        if x.ndim != 3 or loss_mask.shape != x.shape:
            raise ValueError('Expected matching [batch, horizon, channels] tensors')
        x0 = torch.randn_like(x)
        tau = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        t = tau[:, None, None]
        xt = (1-t)*x0+t*x
        target = x-x0
        for index, state in conditions.items():
            xt[:, index, self.action_dim:] = state
        xt[:, -1, :self.action_dim] = 0.
        prediction = self.model(xt, tau)
        loss = ((prediction-target).square()*loss_mask).sum()/loss_mask.sum().clamp_min(1.)
        return loss, {'loss': loss.detach().item()}
