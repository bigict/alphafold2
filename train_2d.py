import os
import sys
import argparse
import logging
import resource

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from alphafold2_pytorch import constants
from alphafold2_pytorch.common import residue_constants
from alphafold2_pytorch.data import esm,scn
from alphafold2_pytorch.model import Alphafold2
from alphafold2_pytorch.utils import *

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""

  #all_atom_positions = all_atom_positions.reshape(:, scn.NUM_COORDS_PER_RES, ...)
  #all_atom_positions = rearrange(all_atom_positions, 'b (l c) d -> b l c d', c=scn.NUM_COORDS_PER_RES)
  #all_atom_positions = torch.split(all_atom_positions, scn.NUM_COORDS_PER_RES, dim=1)
  #all_atom_positions = torch.stack(all_atom_positions, dim=1)
  is_gly = torch.eq(aatype, residue_constants.restype_order['G'])
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = torch.where(
      torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = torch.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.float()
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta

def main(args):
    # constants
    
    DEVICE = constants.DEVICE # defaults to cuda if available, else cpu
    
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    
    # set emebdder model from esm if appropiate - Load ESM-1b model
    
    if args.features == "esm":
        esm_extractor = esm.ESMEmbeddingExtractor(*esm.ESM_MODEL_PATH)
    
    # helpers
    
    def cycle(loader, cond = lambda x: True):
        epoch = 0
        while True:
            logging.info('epoch: {}'.format(epoch))

            data_iter = iter(loader)
            for data in data_iter:
                if cond(data):
                    yield data

            epoch += 1
    
    # get data
    
    data = scn.load(
        casp_version = args.casp_version,
        thinning = 30,
        with_pytorch = 'dataloaders',
        batch_size = args.batch_size,
        num_workers = 0,
        dynamic_batching = False
    )
    
    train_loader = data['train']
    data_cond = lambda t: True #args.min_protein_len <= t[1].shape[1] and t[1].shape[1] <= args.max_protein_len
    dl = cycle(train_loader, data_cond)

    # model
    
    if args.alphafold2_continue:
        model = torch.load(os.path.join(args.prefix, 'model.pkl'))
        mode.eval()
        model.to(DEVICE)
    else:
        model = Alphafold2(
            dim = args.alphafold2_dim,
            depth = args.alphafold2_depth,
            heads = 8,
            dim_head = 64,
            predict_coords = False,
            predict_angles = False,
            structure_module_depth = 2,
            structure_module_heads = 4,
            structure_module_dim_head = 16,
            headers = {'distogram': ({'dim': args.alphafold2_dim,
                    'first_break': 2.3125, 
                    'last_break': 21.6875,
                    'num_buckets': constants.DISTOGRAM_BUCKETS}, {'weigth': 1.0})}
        ).to(DEVICE)
    #    structure_module_dim = 8,
    #    structure_module_refinement_iters = 2

    # optimizer 
    
    optim = Adam(model.parameters(), lr = args.learning_rate)
    
    # tensorboard
    writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval'))
    
    # training loop
    
    for it in range(args.num_batches):
        running_loss = 0
        for jt in range(args.gradient_accumulate_every):
            batch = next(dl)
            pids, seq, mask, str_seqs, coords, coord_masks = batch['pid'], batch['seq'], batch['mask'], batch['str_seq'], batch['coord'], batch['coord_mask']
            logging.debug('seq.shape: {}'.format(seq.shape))
    
            # prepare data and mask labels
            seq, mask, coords, coord_masks = seq.to(DEVICE), mask.to(DEVICE), coords.to(DEVICE), coord_masks.to(DEVICE)
            # coords = rearrange(coords, 'b (l c) d -> b l c d', l = l) # no need to rearrange for now
            # mask the atoms and backbone positions for each residue
    
            # sequence embedding (msa / esm / attn / or nothing)
            msa, embedds = None, None
    
            # get embedds
            if args.features == "esm":
                data = list(zip(pids, str_seqs))
                embedds = rearrange(
                        esm_extractor.extract(data, repr_layer=esm.ESM_EMBED_LAYER, device=DEVICE),
                        'b l c -> b () l c')
            # get msa here
            elif args.features == "msa":
                pass 
            # no embeddings 
            else:
                pass
    
            pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(seq, coords, coord_masks)
            batch = {'pseudo_beta': pseudo_beta,
                    'pseudo_beta_mask': pseudo_beta_mask}
            # predict - out is (batch, L * 3, 3)
    
            r = model(
                seq,
                mask = mask,
                embedds = embedds,
                msa = msa,
                batch = batch
            )
    
            if it == 0 and jt == 0 and args.tensorboard_add_graph:
                with SummaryWriter(os.path.join(args.prefix, 'runs', 'network')) as w:
                    w.add_graph(model, (seq, mask, embedds), verbose=True)
   
            # atom mask
            #_, atom_mask, _ = scn_backbone_mask(seq, boolean=True)
            atom_mask = torch.zeros(scn.NUM_COORDS_PER_RES).to(seq.device)
            atom_mask[..., 1] = 1

            # loss - distogram_dispersion
            running_loss += r.loss.item()
            r.loss.backward()
    
        running_loss /= (args.gradient_accumulate_every*args.batch_size)
        logging.info('{} loss: {}'.format(it, running_loss))
        writer.add_scalar('Loss/train', running_loss, it)
        running_loss = 0
    
        optim.step()
        optim.zero_grad()

    writer.close()

    # save model
    torch.save(model, os.path.join(args.prefix, 'model.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--prefix', type=str, default='.', help='prefix of out directory, default=\'.\'')
    parser.add_argument('-C', '--casp-version', type=int, default=12, help='CASP version, default=12')
    parser.add_argument('-F', '--features', type=str, default='esm', help='AA residue features one of [esm,msa], default=esm')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads used for intraop parallelism on CPU., default=0')
    parser.add_argument('-m', '--min_protein_len', type=int, default=50, help='filter out proteins whose length<LEN, default=50')
    parser.add_argument('-M', '--max_protein_len', type=int, default=1024, help='filter out proteins whose length>LEN, default=1024')

    parser.add_argument('-n', '--num_batches', type=int, default=100000, help='number of batches, default=10^5')
    parser.add_argument('-k', '--gradient_accumulate_every', type=int, default=16, help='accumulate grads every k times, default=16')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size, default=1')
    parser.add_argument('-l', '--learning_rate', type=float, default='3e-4', help='learning rate, default=3e-4')

    parser.add_argument('--alphafold2_dim', type=int, default=256, help='dimension of alphafold2, default=256')
    parser.add_argument('--alphafold2_depth', type=int, default=1, help='depth of alphafold2, default=1')
    parser.add_argument('--alphafold2_fape_min', type=float, default=1e-4, help='minimum of dij in alphafold2, default=1e-4')
    parser.add_argument('--alphafold2_fape_max', type=float, default=10.0, help='maximum of dij in alphafold2, default=10.0')
    parser.add_argument('--alphafold2_fape_z', type=float, default=10.0, help='Z of dij in alphafold2, default=10.0')
    parser.add_argument('--alphafold2_continue', action='store_true', help='load a model and continue to train')

    parser.add_argument('--tensorboard_add_graph', action='store_true', help='call tensorboard.add_graph')
    parser.add_argument('--save_pdb', action='store_true', help='save pdb')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    # logging

    if not os.path.exists(args.prefix):
        os.makedirs(os.path.abspath(args.prefix))
    logging.basicConfig(
            format = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s',
            level = logging.DEBUG if args.verbose else logging.INFO,
            handlers = [
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.prefix, '{}.log'.format(
                    os.path.splitext(os.path.basename(__file__))[0])))
            ]
        )

    logging.info('-----------------')
    logging.info('Arguments: {}'.format(args))
    logging.info('-----------------')

    main(args)

    logging.info('-----------------')
    logging.info('Resources(myself): {}'.format(resource.getrusage(resource.RUSAGE_SELF)))
    logging.info('Resources(children): {}'.format(resource.getrusage(resource.RUSAGE_CHILDREN)))
    logging.info('-----------------')
