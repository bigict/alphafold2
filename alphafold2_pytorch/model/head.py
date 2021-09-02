import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from alphafold2_pytorch.utils import *

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
  return loss

class DistogramHead(nn.Module):
    """Head to predict a distogram.
    """
    def __init__(self, dim,
            first_break, last_break, num_buckets):
        super().__init__()

        self.num_buckets = num_buckets
        self.buckets = torch.linspace(first_break, last_break, steps=num_buckets-1)
        self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_buckets))

    def forward(self, representations, batch, is_training):
        """Builds DistogramHead module.

       Arguments:
         representations: Dictionary of representations, must contain:
           * 'pair': pair representation, shape [N_res, N_res, c_z].
         batch: Batch, unused.
         is_training: Whether the module is in training mode.

       Returns:
         Dictionary containing:
           * logits: logits for distogram, shape [N_res, N_res, N_bins].
           * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
        """
        x = representations['pair']
        trunk_embeds = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # symmetrize
        return dict(logits=self.net(trunk_embeds))

    def loss(self, value, batch):
        """Log loss of a distogram."""
        logits = value['logits']
        assert len(logits.shape) == 4
        positions = batch['pseudo_beta']
        mask = batch['pseudo_beta_mask']

        assert positions.shape[-1] == 3

        sq_breaks = torch.square(self.buckets)

        dist2 = torch.sum(
            torch.square(
                rearrange(positions, 'b l c -> b l () c') -
                rearrange(positions, 'b l c -> b () l c')),
            dim=-1,
            keepdims=True)

        true_bins = torch.sum(dist2 > sq_breaks, axis=-1)

        errors = softmax_cross_entropy(
            labels=F.one_hot(true_bins, self.num_buckets), logits=logits)


        square_mask = rearrange(mask, 'b l -> b () l') * rearrange(mask, 'b l -> b l ()')

        avg_error = (
            torch.sum(errors * square_mask) /
            (1e-6 + torch.sum(square_mask)))
        return dict(loss=avg_error, true_dist=torch.sqrt(dist2+1e-6))

class HeaderBuilder:
    _headers = {
            'distogram': DistogramHead
        }
    @staticmethod
    def build(config):
        if exists(config):
            return dict((k, (HeaderBuilder._headers[k](**args), data)) for k, (args, data) in config.items())
        return None
