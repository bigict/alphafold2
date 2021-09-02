from dataclasses import dataclass

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

from alphafold2_pytorch import constants
from alphafold2_pytorch.model.evoformer import *
from alphafold2_pytorch.model.folding import IPABlock
from alphafold2_pytorch.model.head import HeaderBuilder
from alphafold2_pytorch.model.mlm import MLM
from alphafold2_pytorch.utils import *

# constants

@dataclass
class Recyclables:
    coords: torch.Tensor
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor

@dataclass
class ReturnValues:
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None
    headers: dict = None
    loss: torch.Tensor = None

class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len = 2048,
        depth = 6,
        heads = 8,
        dim_head = 64,
        max_rel_dist = 32,
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        max_num_msas = constants.MAX_NUM_MSA,
        max_num_templates = constants.MAX_NUM_TEMPLATES,
        extra_msa_evoformer_layers = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        templates_dim = 32,
        templates_embed_layers = 4,
        templates_angles_feats_dim = 55,
        predict_angles = False,
        symmetrize_omega = False,
        predict_coords = False,                # structure module related keyword arguments below
        structure_module_depth = 4,
        structure_module_heads = 1,
        structure_module_dim_head = 4,
        disable_token_embed = False,
        mlm_mask_prob = 0.15,
        mlm_random_replace_token_prob = 0.1,
        mlm_keep_token_same_prob = 0.1,
        mlm_exclude_token_ids = (0,),
        recycling_distance_buckets = 32,
        headers = None
    ):
        super().__init__()
        self.dim = dim

        # token embedding

        self.token_emb = nn.Embedding(num_tokens + 1, dim) if not disable_token_embed else Always(0)
        self.to_pairwise_repr = nn.Linear(dim, dim * 2)
        self.disable_token_embed = disable_token_embed

        # positional embedding

        self.max_rel_dist = max_rel_dist
        self.pos_emb = nn.Embedding(max_rel_dist * 2 + 1, dim)

        # extra msa embedding

        self.extra_msa_evoformer = Evoformer(
            dim = dim,
            depth = extra_msa_evoformer_layers,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            global_column_attn = True
        )

        # template embedding
        self.template_embedding = TemplateEmbedding(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                max_seq_len = max_seq_len,
                attn_dropout = attn_dropout,
                templates_dim = templates_dim,
                templates_embed_layers = templates_embed_layers,
                templates_angles_feats_dim = templates_angles_feats_dim)

        # projection for angles, if needed

        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)

        # custom embedding projection

        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules

        self.evoformer = Evoformer(
            dim = dim,
            depth = depth,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # MSA SSL MLM

        self.mlm = MLM(
            dim = dim,
            num_tokens = num_tokens,
            mask_id = num_tokens, # last token of embedding is used for masking
            mask_prob = mlm_mask_prob,
            keep_token_same_prob = mlm_keep_token_same_prob,
            random_replace_token_prob = mlm_random_replace_token_prob,
            exclude_token_ids = mlm_exclude_token_ids
        )

        # calculate distogram logits

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

        # to coordinate output

        self.predict_coords = predict_coords
        self.structure_module_depth = structure_module_depth

        self.msa_to_single_repr_dim = nn.Linear(dim, dim)
        self.trunk_to_pairwise_repr_dim = nn.Linear(dim, dim)

        with torch_default_dtype(torch.float32):
            self.ipa_block = IPABlock(
                dim = dim,
                heads = structure_module_heads,
            )

            self.to_quaternion_update = nn.Linear(dim, 6)

        init_zero_(self.ipa_block.attn.to_out)

        self.to_points = nn.Linear(dim, 3)

        # aux confidence measure

        self.lddt_linear = nn.Linear(dim, 1)

        # recycling params

        self.recycling_msa_norm = nn.LayerNorm(dim)
        self.recycling_pairwise_norm = nn.LayerNorm(dim)
        self.recycling_distance_embed = nn.Embedding(recycling_distance_buckets, dim)
        self.recycling_distance_buckets = recycling_distance_buckets

        self.headers = default(HeaderBuilder.build(headers), {})

    def forward(
        self,
        seq,
        msa = None,
        mask = None,
        msa_mask = None,
        extra_msa = None,
        extra_msa_mask = None,
        seq_index = None,
        seq_embed = None,
        msa_embed = None,
        templates_feats = None,
        templates_mask = None,
        templates_angles = None,
        embedds = None,
        recyclables = None,
        return_trunk = False,
        return_confidence = False,
        return_recyclables = False,
        return_aux_logits = False,
        batch = None
    ):
        assert not (self.disable_token_embed and not exists(seq_embed)), 'sequence embedding must be supplied if one has disabled token embedding'
        assert not (self.disable_token_embed and not exists(msa_embed)), 'msa embedding must be supplied if one has disabled token embedding'

        # if MSA is not passed in, just use the sequence itself

        if not exists(embedds) and not exists(msa):
            msa = rearrange(seq, 'b n -> b () n')
            msa_mask = rearrange(mask, 'b n -> b () n')

        # assert on sequence length

        assert not exists(msa) or msa.shape[-1] == seq.shape[-1], 'sequence length of MSA and primary sequence must be the same'

        # variables

        b, n, device = *seq.shape[:2], seq.device
        n_range = torch.arange(n, device = device)

        # unpack (AA_code, atom_pos)

        if isinstance(seq, (list, tuple)):
            seq, seq_pos = seq

        # embed main sequence

        x = self.token_emb(seq)

        if exists(seq_embed):
            x += seq_embed

        # mlm for MSAs

        if self.training and exists(msa):
            original_msa = msa
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

            noised_msa, replaced_msa_mask = self.mlm.noise(msa, msa_mask)
            msa = noised_msa

        # embed multiple sequence alignment (msa)

        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m = m + msa_embed

            # add single representation to msa representation

            m = m + rearrange(x, 'b n d -> b () n d')

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

        elif exists(embedds):
            m = self.embedd_project(embedds)
            
            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(embedds[..., -1]).bool())
        else:
            raise Error('either MSA or embeds must be given')

        # derive pairwise representation

        x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim = -1)
        x = rearrange(x_left, 'b i d -> b i () d') + rearrange(x_right, 'b j d-> b () j d') # create pair-wise residue embeds
        x_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j') if exists(mask) else None

        # add relative positional embedding

        seq_index = default(seq_index, lambda: torch.arange(n, device = device))
        seq_rel_dist = rearrange(seq_index, 'i -> () i ()') - rearrange(seq_index, 'j -> () () j')
        seq_rel_dist = seq_rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = self.pos_emb(seq_rel_dist)

        x = x + rel_pos_emb

        # add recyclables, if present

        if exists(recyclables):
            m[:, 0] = m[:, 0] + self.recycling_msa_norm(recyclables.single_msa_repr_row)
            x = x + self.recycling_pairwise_norm(recyclables.pairwise_repr)

            distances = torch.cdist(recyclables.coords, recyclables.coords, p=2)
            boundaries = torch.linspace(2, 20, steps = self.recycling_distance_buckets, device = device)
            discretized_distances = torch.bucketize(distances, boundaries[:-1])
            distance_embed = self.recycling_distance_embed(discretized_distances)

            x = x + distance_embed

        # embed templates, if present

        x, x_mask, m, msa_mask = self.template_embedding(
                x, x_mask, m, msa_mask,
                templates_feats=templates_feats,
                templates_angles=templates_angles,
                templates_mask=templates_mask)

        # embed extra msa, if present

        if exists(extra_msa):
            extra_m = self.token_emb(msa)
            extra_msa_mask = default(extra_msa_mask, torch.ones_like(extra_m).bool())

            x, extra_m = self.extra_msa_evoformer(
                x,
                extra_m,
                mask = x_mask,
                msa_mask = extra_msa_mask
            )

        # trunk

        x, m = self.evoformer(
            x,
            m,
            mask = x_mask,
            msa_mask = msa_mask
        )

        # ready output container

        ret = ReturnValues()

        # calculate theta and phi before symmetrization

        if self.predict_angles:
            ret.theta_logits = self.to_prob_theta(x)
            ret.phi_logits = self.to_prob_phi(x)

        representations = {'pair': x}
        ret.headers = {}
        for name, (module, options) in self.headers.items():
            ret.headers[name] = module(representations, batch, self.training)
            if self.training:
                loss = module.loss(ret.headers[name], batch)
                ret.headers[name].update(loss)
                if ret.loss is None:
                    ret.loss = loss['loss'] * options.get('weight', 1.0)
                else:
                    ret.loss += loss['loss'] * options.get('weight', 1.0)

        # calculate mlm loss, if training

        ret.msa_mlm_loss = None
        if self.training and exists(msa):
            num_msa = original_msa.shape[1]
            ret.msa_mlm_loss = self.mlm(m[:, :num_msa], original_msa, replaced_msa_mask)

        # determine angles, if specified

        if self.predict_angles:
            omega_input = trunk_embeds if self.symmetrize_omega else x
            ret.omega_logits = self.to_prob_omega(omega_input)

        if not self.predict_coords or return_trunk:
            return ret

        # derive single and pairwise embeddings for structural refinement

        single_msa_repr_row = m[:, 0]

        single_repr = self.msa_to_single_repr_dim(single_msa_repr_row)
        pairwise_repr = self.trunk_to_pairwise_repr_dim(x)

        # prepare float32 precision for equivariance

        original_dtype = single_repr.dtype
        single_repr, pairwise_repr = map(lambda t: t.float(), (single_repr, pairwise_repr))

        # iterative refinement with equivariant transformer in high precision

        with torch_default_dtype(torch.float32):

            quaternions = torch.tensor([1., 0., 0., 0.], device = device) # initial rotations
            quaternions = repeat(quaternions, 'd -> b n d', b = b, n = n)
            translations = torch.zeros((b, n, 3), device = device)

            # go through the layers and apply invariant point attention and feedforward

            for i in range(self.structure_module_depth):
                is_last = i == (self.structure_module_depth - 1)

                # the detach comes from
                # https://github.com/deepmind/alphafold/blob/0bab1bf84d9d887aba5cfb6d09af1e8c3ecbc408/alphafold/model/folding.py#L383
                rotations = quaternion_to_matrix(quaternions)

                if not is_last:
                    rotations = rotations.detach()

                single_repr = self.ipa_block(
                    single_repr,
                    mask = mask.bool(),
                    pairwise_repr = pairwise_repr,
                    rotations = rotations,
                    translations = translations
                )

                # update quaternion and translation

                quaternion_update, translation_update = self.to_quaternion_update(single_repr).chunk(2, dim = -1)
                quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)

                quaternions = quaternion_multiply(quaternions, quaternion_update)
                translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

            points_local = self.to_points(single_repr)
            rotations = quaternion_to_matrix(quaternions)
            coords = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations

        coords.type(original_dtype)

        if return_recyclables:
            coords, single_msa_repr_row, pairwise_repr = map(torch.detach, (coords, single_msa_repr_row, pairwise_repr))
            ret.recyclables = Recyclables(coords, single_msa_repr_row, pairwise_repr)

        if return_aux_logits:
            return coords, ret

        if return_confidence:
            return coords, self.lddt_linear(single_repr.float())

        return coords
