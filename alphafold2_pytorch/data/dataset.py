import os
import random
import math
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from alphafold2_pytorch.common import residue_constants
from alphafold2_pytorch.data.parsers import parse_a3m,parse_fasta


class ProteinStructureDataset(Dataset):

    atom_map = {'N': 0, 'CA': 1, 'C': 2}

    def __init__(self, work_dir, msa_max_size=128) -> None:
        super().__init__()
        self.dir = work_dir
        self.max_msa = msa_max_size
        with open(os.path.join(self.dir, "pdb_names")) as fin:
            self.protein_ids = list(map(str.rstrip, fin.readlines()))
        

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        return {'seq':self.get_seq_features(protein_id),
                'msa':self.get_msa_features(protein_id),
                'coords':self.get_structure_label(protein_id)}

    def __len__(self):
        return len(self.protein_ids)

    def get_msa_features(self, protein_id):
        """Constructs a feature dict of MSA features."""
        msa_path = os.path.join(self.dir, f'a3ms/{protein_id}.a3m')
        with open(msa_path) as fin:
            string = fin.read()
        msa, del_matirx = parse_a3m(string)
        msas = (msa,)
        if not msas:
            raise ValueError('At least one MSA must be provided.')
        deletion_matrices = (del_matirx,)

        int_msa = []
        deletion_matrix = []
        seen_sequences = set()
        for msa_index, msa in enumerate(msas):
            if not msa:
                raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
            for sequence_index, sequence in enumerate(msa):
                if sequence in seen_sequences:
                    continue
                seen_sequences.add(sequence)
                int_msa.append(
                    [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
                deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

        features = {}
        if self.max_msa == 0:
            features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
            features['msa'] = np.array(int_msa, dtype=np.int32)
        else:
            features['deletion_matrix_int'] = np.array(random.choices(deletion_matrix, k=self.max_msa), dtype=np.int32)
            features['msa'] = np.array(random.choices(int_msa, k=self.max_msa), dtype=np.int32)

        msa = torch.tensor(features['msa'], dtype=torch.long)  # 22 possible values.
        msa_one_hot = nn.functional.one_hot(msa, 23).float()  # extra dim for mask flag.
        deletion_matrix_int = torch.tensor(features['deletion_matrix_int'], dtype=torch.float)
        cluster_deletion_value = 2 / math.pi * torch.arctan(deletion_matrix_int / 3)
        cluster_deletion_mean = deletion_matrix_int.mean(axis=0, keepdim=True)
        cluster_deletion_mean = cluster_deletion_mean.expand(deletion_matrix_int.shape)
        cluster_deletion_mean = 2 / math.pi * torch.arctan(cluster_deletion_mean / 3)
        cluster_has_deletion = (deletion_matrix_int != 0).float()
        deletion_features = torch.stack((cluster_deletion_value, cluster_deletion_mean, cluster_has_deletion), dim=2)  # (N, L, 3) <- ...
        result = torch.cat((msa_one_hot, deletion_features), dim=2)
        return result

    def get_structure_label(self, protein_id):
        input_structure_path = os.path.join(self.dir, f"pdbs/{protein_id}.pkl")
        with open(input_structure_path, 'rb') as fin:
            structure = pickle.load(fin)
        
        seq_len = len(structure)
        result = torch.empty(seq_len, 3, 3)
        for (i, residule) in enumerate(structure):
            is_empty = True
            for atom_name, coords in residule.items():
                is_empty = False
                if atom_name in self.atom_map:
                    result[i][self.atom_map[atom_name]] = torch.tensor(coords)
            if is_empty:
                result[i].fill_(float('nan'))

        v1 = result[:, 2:3, :] - result[:, 1:2, :]  # (L, 1, 3)
        v2 = result[:, 0:1, :] - result[:, 1:2, :]  # (L, 1, 3)
        e1 = v1 / torch.linalg.norm(v1, ord=2, dim=-1, keepdim=True)
        u2 = v2 - e1 * torch.bmm(e1, v2.transpose(1, 2))
        e2 = u2 / torch.linalg.norm(u2, ord=2, dim=-1, keepdim=True)
        e3 = torch.cross(e1, e2, dim=2)  # (L, 1, 3)
        result2 = torch.cat((e1, e2, e3), dim=1)  # (L, 3, 3) <- ...
        transition = result[:, 1, :]  # (L, 3)
        return {'coord': result, 'rotation': result2, 'transition': transition}

    def get_seq_features(self, protein_id):
        """Runs alignment tools on the input sequence and creates features."""
        input_fasta_path = os.path.join(self.dir, f"fastas/{protein_id}.fasta")
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)
        features = {}
        features['aatype'] = residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True)
        features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
        #features['domain_name'] = np.array([input_description.encode('utf-8')], dtype=np.object_)
        features['residue_index'] = np.array(range(num_res), dtype=np.int32)
        features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
        #features['sequence'] = np.array([input_sequence.encode('utf-8')], dtype=np.object_)
        return features
