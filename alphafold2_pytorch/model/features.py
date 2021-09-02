import torch

from alphafold2_pytorch import constants
from alphafold2_pytorch.utils import default,exists

# adapted from https://github.com/facebookresearch/esm

class ESMEmbeddingExtractor:
    def __init__(self, repo_or_dir, model):
        self.model, alphabet = torch.hub.load(repo_or_dir, model)
        self.batch_converter = alphabet.get_batch_converter()

    def extract(self, seqs, repr_layer=None, return_contacts=False, device=None):
        """ Returns the ESM embeddings for a protein.
            Inputs:
            * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
            Outputs: tensor of (batch, n_seqs, L, embedd_dim)
                * n_seqs: number of sequences in the MSA. 1 for ESM-1b
                * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
        """
        # use ESM transformer
        device = default(device, getattr(seqs, 'device', None))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(seqs)

        max_seq_len = max(map(lambda x: len(x[1]), seqs))
        repr_layer = default(repr_layer, constants.ESM_EMBED_LAYER)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            if exists(device):
                batch_tokens = batch_tokens.to(device)
            results = self.model(batch_tokens, repr_layers=[repr_layer], return_contacts=return_contacts)

        # index 0 is for start token. so take from 1 one
        if return_contacts:
            return results['representations'][repr_layer][...,1:max_seq_len+1,:], results['contacts']
        return results['representations'][repr_layer][...,1:max_seq_len+1,:]


