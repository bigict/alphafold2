import os
import argparse
import logging
import resource

from alphafold2_pytorch.data.pipeline import DataPipeline
from alphafold2_pytorch.data.templates import TemplateHitFeaturizer

MAX_TEMPLATE_HITS=20

def main(args):
  template_featurizer = TemplateHitFeaturizer(
      mmcif_dir=args.template_mmcif_dir,
      max_template_date=args.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=args.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=args.obsolete_pdbs_path)
  data_pipeline = DataPipeline(
      jackhmmer_binary_path=args.jackhmmer_binary_path,
      hhblits_binary_path=args.hhblits_binary_path,
      hhsearch_binary_path=args.hhsearch_binary_path,
      uniref90_database_path=args.uniref90_database_path,
      mgnify_database_path=args.mgnify_database_path,
      bfd_database_path=args.bfd_database_path,
      uniclust30_database_path=args.uniclust30_database_path,
      small_bfd_database_path=args.small_bfd_database_path,
      pdb70_database_path=args.pdb70_database_path,
      template_featurizer=template_featurizer,
      use_small_bfd=args.use_small_bfd)

  feature_dict = data_pipeline.process(
      input_fasta_path=args.fasta_path,
      msa_output_dir=os.path.join(args.prefix, 'msa_output_dir'))

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(args.prefix, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--prefix', type=str, default='.', help='prefix of out directory, default=\'.\'')
  parser.add_argument('--template_mmcif_dir', type=str, default='template_mmcif_dir', help='template_mmcif_dir, default=\'template_mmcif\'')
  parser.add_argument('--max_template_date', type=str, default='max_template_date', help='max_template_date, default=\'2020-09-11\'')
  parser.add_argument('--kalign_binary_path', type=str, default='kalign_binary_path', help='kalign_binary_path, default=\'kalign\'')
  parser.add_argument('--obsolete_pdbs_path', type=str, default=None, help='obsolete_pdbs_path, default=None')
  parser.add_argument('--jackhmmer_binary_path', type=str, default='jackhmmer', help='jackhmmer_binary_path, default=\'jackhmmer\'')
  parser.add_argument('--hhblits_binary_path', type=str, default='hhblits', help='hhblits_binary_path, default=\'hhblits\'')
  parser.add_argument('--hhsearch_binary_path', type=str, default='hhsearch', help='hhsearch_binary_path, default=\'hhsearch\'')
  parser.add_argument('--uniref90_database_path', type=str, default='uniref90_database_path', help='uniref90_database, default=\'uniref90_database_path\'')
  parser.add_argument('--bfd_database_path', type=str, default='bfd_database', help='bfd_database_path, default=\'bfd_database\'')
  parser.add_argument('--small_bfd_database_path', type=str, default='small_bfd_database', help='small_bfd_database_path, default=\'small_bfd_database\'')
  parser.add_argument('--uniclust30_database_path', type=str, default='uniclust30_database', help='uniclust30_database_path, default=\'uniclust30_database\'')
  parser.add_argument('--mgnify_database_path', type=str, default='mgnify_database', help='mgnify_database_path, default=\'mgnify_database\'')
  parser.add_argument('--pdb70_database_path', type=str, default='pdb70_database', help='pdb70_database_path, default=\'pdb70_database\'')
  parser.add_argument('--fasta_path', type=str, default='fasta_path', help='fasta_path, default=\'fasta_path\'')
  parser.add_argument('--use_small_bfd', action='store_true', help='use_small_bfd')
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
