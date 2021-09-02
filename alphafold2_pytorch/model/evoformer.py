from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from alphafold2_pytorch.model.commons import *

# main evoformer class

class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, global_column_attn = global_column_attn),
            FeedForward(dim = dim, dropout = ff_dropout),
            MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim = dim, dropout = ff_dropout),
        ])

    def forward(self, inputs):
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # msa attention and transition

        m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        m = msa_ff(m) + m

        # pairwise attention and transition

        x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = ff(x) + x

        return x, m, mask, msa_mask

class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m

