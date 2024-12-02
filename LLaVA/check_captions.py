from argparse import ArgumentParser
import json
import os
from pprint import pprint
import sys

from accelerate import init_empty_weights
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from tqdm.auto import tqdm
import lovely_tensors as lt

sys.path.append(os.path.abspath('../prompt-to-prompt'))
from data import classify_with_labels_dataset_helper_registry
from heatmap import compute_token_merge_indices

if __name__ == '__main__':

  lt.monkey_patch()

  # command-line arguments
  parser = ArgumentParser(
    prog='classify_data',
    description='classify data'
  )
  parser.add_argument('-s', '--split', default=1, type=int, help='How many sub-arrays to split prompts into before splitting between devices. Use this to split the data across different running instances of this script on the same gpu/s.')
  parser.add_argument('-i', '--index', default=0, type=int, help='Which sub-array of prompts to split between devices. Use this to split the data across different running instances of this script on the same gpu/s.')
  parser.add_argument('--dataset', type=str, help='Which dataset to generate for')
  parser.add_argument('--dataset-split', type=str, help='Which split of dataset to classify')
  parser.add_argument('--prompt-type', type=str, help='Which type of promting to use')
  parser.add_argument('--save-dir', type=str, help='Where captions are saved')

  args = parser.parse_args()
 
  # prepare data
  class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
  class_labels = np.array(class_labels)

  # get split
  _dataset = dataset[args.dataset_split]
  print('Dataset')
  print(_dataset)

  revised_class_labels = class_labels
  if args.dataset == 'artdl':
    revised_class_labels = [
      'Anthony of Padua',
      'John the Baptist',
      'Paul the Apostle',
      'Francis of Assisi',
      'Mary Magdalene',
      'Saint Jerome',
      'Saint Dominic',
      'Mary, mother of Jesus',
      'Saint Peter',
      'Saint Sebastian'
    ]
  elif args.dataset == 'iconart':
    if args.prompt_type == 'custom_1':
      revised_class_labels = [
       'person',
       'crucifixion of jesus',
       'angel',
       'mary',
       'baby',
       'naked person',
       'ruins'
     ]
    else:
      revised_class_labels = [
        'Saint Sebastien',
        'crucifixion of Jesus',
        'angel',
        'Mary, mother of Jesus',
        'Child Jesus',
        'nudity',
        'ruins'
      ]

  label_to_revised_label = dict(zip(class_labels, revised_class_labels))

  # get tokenizer
  with init_empty_weights():
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-base', scheduler=scheduler)
  tokenizer = ldm_stable.tokenizer
  del ldm_stable

  for image_id in tqdm(_dataset['image_id'], dynamic_ncols=True):

    filepath = os.path.join(args.save_dir, f'{image_id}.json')
    with open(filepath) as f:
      captions = json.load(f)

    prompts_for_generation = []
    for label_caption in captions['captions']:
      # if label is not in caption or is beyond at a position beyondthe input length of the diffusion model, prepend the caption with a short caption containing the label
      label, caption = list(label_caption.items())[0]
      revised_label = label_to_revised_label[label]

      prompt_for_generation = caption
      prepend_prompt = False
      if ((revised_label in caption.split()) if len(revised_label.split()) == 1 else (revised_label in caption)):
        try:
          merge_idxs, _ = compute_token_merge_indices(tokenizer, caption, revised_label)
          if any([idx >= tokenizer.model_max_length for idx in merge_idxs]):
            print(f'"{revised_label}" is at least partially out of range for the model in caption "{caption}". Ids are {merge_idxs} while the model\'s max length in {tokenizer.model_max_length}.')
            prepend_prompt = True
        except:
          print(f'Edge case encountered with label "{revised_label}" and caption "{caption}". Probably something like "angel" still being in "angels".')
          prepend_prompt = True
      else:
        print(f'"{revised_label}" not in "{caption}"')
        prepend_prompt = True

      if prepend_prompt:
        prompt_for_generation = tokenizer.decode(tokenizer(f'A painting of {revised_label}. {caption}', truncation=True)['input_ids'], skip_special_tokens=True)

      prompts_for_generation.append({label: prompt_for_generation})

    captions['prompts_for_generation'] = prompts_for_generation

    with open(filepath, 'w') as f:
      json.dump(captions, f)
