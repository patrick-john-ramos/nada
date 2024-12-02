from data import classify_dataset_helper_registry, detect_dataset_helper_registry
from argparse import ArgumentParser
from functools import partial

import glob
import os
from pprint import pprint


def verify_files(registry, dataset, dataset_split, save_dir, prioritize_existing_files=False, verbose=False, **registry_kwargs):
  # checks which items in the dataset have existing predictions in the save directory
  # and whether all existing files in the save directory correspond to an item in the dataset
  
  *_, check_files = registry[dataset](split=dataset_split, **registry_kwargs)

  inputs, completed_files = check_files(save_dir)
  print(completed_files)
  existing_files = glob.glob(os.path.join(save_dir, '*', '*.json'))
  if len(existing_files) == 0: # anns are save_dir/label/json, labels are save_dir/json
    existing_files = glob.glob(os.path.join(save_dir, '*.json'))

  all_completed_exist = True
  for file in completed_files:
    if file not in existing_files:
      if verbose:
        print(f'{file} not in existing files')
      all_completed_exist = False

  all_existing_completed = True
  for file in existing_files:
    if file not in completed_files:
      if verbose:
        print(f'{file} not in completed files')
      all_existing_completed = False


  verified = len(completed_files) == len(existing_files) and all_completed_exist and all_existing_completed
  if verified or prioritize_existing_files:
    pprint(sorted(existing_files, key=os.path.getmtime)[-5:])
    if verbose:
      if prioritize_existing_files:
        print('Assuming all existing files are in completed files...', end='')
        if not all_existing_completed:
          print('Not all existing files were completed but assuming they are anyway.')
        else:
          print('Assumption not needed, all existing files were completed')
      print(f'{len(existing_files)}/{len(completed_files)+len(inputs)}')

  return verified, inputs, completed_files, existing_files

verify_labels = partial(verify_files, classify_dataset_helper_registry)
verify_annotations = partial(verify_files, detect_dataset_helper_registry)

if __name__ == '__main__':
  parser = ArgumentParser(
    prog='count_data',
    description='check how much data has been generated'
  )
  parser.add_argument('--dataset', type=str, help='Which dataset to verify.')
  parser.add_argument('--dataset-split', type=str, help='Which split of the dataset to verify.')
  parser.add_argument('--save-dir', type=str, help='Where to save annotations')
  parser.add_argument('--mode', type=str, choices=['classify', 'detect'], help='Types of files to verify. Use "classify" for classification annotations and "detect" for detection annotations.')
  parser.add_argument('--prioritize-existing-files', action='store_true', help='Assumes all existing files are in completed files when counting files at the end. Can use when generating on so many devices that new files are generated during the identification of completed_files.')
  parser.add_argument('--label-dir', type=str, help='Where to load labels from')
  parser.add_argument('--caption-dir', type=str, help='Where to load captoins from')
  parser.add_argument('--prompt-type', type=str, default='standard', help='Which prompt to use. Check `data/detect.py` to see valid prompt types per dataset.')
  args = parser.parse_args()

  registry_kwargs = {'prompt_type': args.prompt_type}
  registry_kwargs['label_dir'] = args.label_dir
  registry_kwargs['caption_dir'] = args.caption_dir
    
  _, inputs, completed_files, existing_files = verify_files(
    classify_dataset_helper_registry if args.mode == 'classify' else detect_dataset_helper_registry,
    args.dataset,
    args.dataset_split,
    args.save_dir,
    args.prioritize_existing_files,
    verbose=True,
    **registry_kwargs
  )
