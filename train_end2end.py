import os
import sys
import argparse
import logging
import resource

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

# models & data

from alphafold2_pytorch import constants
from alphafold2_pytorch.data import scn,esm
from alphafold2_pytorch.model import Alphafold2,sidechain
from alphafold2_pytorch.utils import *

def main(args):
    # constants
    
    DEVICE = constants.DEVICE # defaults to cuda if available, else cpu
    
    DISTOGRAM_BUCKETS = constants.DISTOGRAM_BUCKETS

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
                yield data

            epoch += 1
    
    # get data
    
    data = scn.load(
        max_seq_len = args.max_protein_len,
        casp_version = args.casp_version,
        thinning = 30,
        with_pytorch = 'dataloaders',
        batch_size = args.batch_size,
        num_workers = 0,
        dynamic_batching = False
    )
    
    train_loader = data['train']
    data_cond = lambda t: args.min_protein_len <= t[1].shape[1] and t[1].shape[1] <= args.max_protein_len
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
            predict_coords = True,
            predict_angles = True,
            structure_module_depth = 2,
            structure_module_heads = 4,
            structure_module_dim_head = 16
        ).to(DEVICE)
    #    structure_module_dim = 8,
    #    structure_module_refinement_iters = 2

    # optimizer 
    
    dispersion_weight = 0.1
    criterion = nn.MSELoss()
    optim = Adam(model.parameters(), lr = args.learning_rate)
    
    # tensorboard
    writer = SummaryWriter(os.path.join(args.prefix, 'runs', 'eval'))
    
    # training loop
    
    for it in range(args.num_batches):
        running_loss = 0
        for jt in range(args.gradient_accumulate_every):
            batch = next(dl)
            seq, mask, str_seqs, coords, coord_mask = batch['seq'], batch['mask'], batch['str_seq'], batch['coord'], batch['coord_mask']
            logging.debug('seq.shape: {}'.format(seq.shape))
    
            # prepare data and mask labels
            seq, coords, mask, coord_mask = seq.to(DEVICE), coords.to(DEVICE), mask.to(DEVICE), coord_mask.to(DEVICE)
            # coords = rearrange(coords, 'b (l c) d -> b l c d', l = l) # no need to rearrange for now
            # mask the atoms and backbone positions for each residue
    
            # sequence embedding (msa / esm / attn / or nothing)
            msa, embedds = None, None
    
            # get embedds
            if args.features == "esm":
                data = list(zip(batch['pid'], str_seqs))
                embedds = rearrange(
                        esm_extractor.extract(data, repr_layer=esm.ESM_EMBED_LAYER, device=DEVICE),
                        'b l c -> b () l c')
            # get msa here
            elif args.features == "msa":
                pass 
            # no embeddings 
            else:
                pass
    
            # predict - out is (batch, L * 3, 3)
    
            backbones = model(
                seq,
                mask = mask,
                embedds = embedds,
                msa = msa
            )
    
            if it == 0 and jt == 0 and args.tensorboard_add_graph:
                with SummaryWriter(os.path.join(args.prefix, 'runs', 'network')) as w:
                    w.add_graph(model, (seq, mask, embedds), verbose=True)
   
            # atom mask
            #_, atom_mask, _ = scn_backbone_mask(seq, boolean=True)
            atom_mask = torch.zeros(scn.NUM_COORDS_PER_RES).to(seq.device)
            atom_mask[..., 1] = 1
    
            ## build SC container. set SC points to CA and optionally place carbonyl O
            proto_sidechain = sidechain.fold(str_seqs, backbones=backbones, atom_mask=atom_mask,
                                                  cloud_mask=coord_mask, num_coords_per_res=scn.NUM_COORDS_PER_RES)
    
            #proto_sidechain = rearrange(proto_sidechain, 'b l c d -> b (l c) d')
    
            flat_cloud_mask = rearrange(coord_mask, 'b l c -> b (l c)')
            # rotate / align
            coords_aligned, labels_aligned = Kabsch(
                    rearrange(proto_sidechain, 'b l c d -> b (l c) d')[flat_cloud_mask], 
                    rearrange(coords, 'b l c d -> b (l c) d')[flat_cloud_mask])
    
            # save pdb files for visualization
    
            #if args.save_pdb: 
            #    # idx from batch to save prot and label
            #    idx = 0
            #    coords2pdb(seq[idx, :, 0], coords_aligned[idx], cloud_mask, prefix=args.prefix, name="pred.pdb")
            #    coords2pdb(seq[idx, :, 0], labels_aligned[idx], cloud_mask, prefix=args.prefix, name="label.pdb")
    
            weights = 1
            # loss - RMSE + distogram_dispersion
            #loss = torch.sqrt(criterion(coords_aligned[flat_chain_mask], labels_aligned[flat_chain_mask])) + \
            #                  dispersion_weight * torch.norm( (1/weights)-1 )
            #loss = torch.sqrt(criterion(coords_aligned, labels_aligned)) + \
            #                  dispersion_weight * torch.norm( (1/weights)-1 )
            loss = torch.clamp(torch.sqrt(criterion(coords_aligned, labels_aligned)), args.alphafold2_fape_min, args.alphafold2_fape_max) / args.alphafold2_fape_z
            running_loss += loss.item()

            loss.backward()
    
        running_loss /= (args.batch_size*args.gradient_accumulate_every)
        logging.info('{} loss: {}'.format(it, running_loss))
        writer.add_scalar('Loss/train', running_loss, it)
    
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
