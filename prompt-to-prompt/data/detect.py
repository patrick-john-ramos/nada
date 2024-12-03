import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import glob
import json
import os
from functools import partial

from .classify_with_labels import classify_with_labels_dataset_helper_registry


def artdl(split, label_dir, caption_dir, prompt_type):
  '''Prepares variables and functions for working with ArtDL'''

  class_labels, dataset = classify_with_labels_dataset_helper_registry['artdl']()
  dataset = dataset[split]
  
  class_labels_for_prompts = class_labels

  if prompt_type == 'standard':
    def create_prompt(label):
      return label, f'a painting of {label}'
  if prompt_type == 'wikipedia':
    class_labels_for_prompts = [
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
    def create_prompt(label):
      label = class_labels_for_prompts[class_labels.index(label)]
      return label, f'a painting of {label}'

  if not caption_dir and label_dir:
    def extract_annotations(item):
      with open(os.path.join(label_dir, f"{item['image_id']}.json")) as f:
        labels = json.load(f)
      return [[label, *create_prompt(label)] for label in labels['labels']]
  elif caption_dir and not label_dir:
    def extract_annotations(item):
      with open(os.path.join(caption_dir, f"{item['image_id']}.json")) as f:
        captions = json.load(f)['prompts_for_generation']
      labels_prompts = []
      for label_prompt in captions:
        label, prompt = list(label_prompt.items())[0]
        new_label, _ = create_prompt(label)
        labels_prompts.append([label, new_label, prompt])
      return labels_prompts
  else:
    raise Exception('Exactly one of `label_dir` or `caption_dir` must be set')
    
  def check_files(save_dir):
    # check if filepath and filename not mixed up
    inputs = []
    completed_files = []
    
    for item in tqdm(dataset):
      for label, new_label, prompt in extract_annotations(item):
        filepath = os.path.join(save_dir, label, f'{item["image_id"]}.json')
        if not os.path.exists(filepath):
          inputs.append((item['file_name'], prompt, label, new_label))
        else:
          completed_files.append(filepath)
    return sorted(set(inputs)), sorted(set(completed_files))


  return class_labels, class_labels_for_prompts, check_files


def iconart(split, label_dir, caption_dir, prompt_type):
  '''Prepares variables and functions for working with IconArt'''

  class_labels, dataset = classify_with_labels_dataset_helper_registry['iconart']()
  dataset = dataset[split]
  
  class_labels_for_prompts = class_labels

  if prompt_type == 'standard':
    def create_prompt(label):
      return label, f'a painting of {label}'
  elif prompt_type == 'custom_0':
    class_labels_for_prompts = [
      'person',
      # 'turban',
      'crucifixion of jesus',
      'child',
      # 'capital',
      'woman',
      # 'beard',
      'baby',
      'naked person',
      'ruins'
    ]
    def create_prompt(label):
      if label == 'angel':
        return 'child', 'a painting of a child'
      elif label == 'child jesus':
        return 'baby', 'a painting of a baby'
      elif label == 'mary':
        return 'woman', 'a painting of a woman'
      elif label == 'nudity':
        return 'naked person', 'a painting of a naked person'
      elif label == 'saint sebastien':
        return 'person', 'a painting of a person'
      else:
        return label, f'a painting of {label}'
  elif prompt_type == 'custom_1':
    class_labels_for_prompts = [
      'person',
      # 'turban',
      'crucifixion of jesus',
      'angel',
      # 'capital',
      'mary',
      # 'beard',
      'baby',
      'naked person',
      'ruins'
    ]
    def create_prompt(label):
      # # this will not work if multiple classes are mapped to the same "new label"
      # # but it should work for this setup
      # return class_labels_for_prompts[class_labels.index(label)], f'a painting of {class_labels_for_prompts[class_labels.index(label)]}'
      if label == 'child jesus':
        return 'baby', 'a painting of a baby'
      elif label == 'nudity':
        return 'naked person', 'a painting of a naked person'
      elif label == 'saint sebastien':
        return 'person', 'a painting of a person'
      else:
        return label, f'a painting of {label}'
  else:
    raise ValueError(f'Prompt type {prompt_type} is invalid')


  if not caption_dir and label_dir:
    def extract_annotations(item):
      with open(os.path.join(label_dir, f"{item['image_id']}.json")) as f:
        labels = json.load(f)
      return [[label, *create_prompt(label)] for label in labels['labels']]
  elif caption_dir and not label_dir:
    def extract_annotations(item):
      with open(os.path.join(caption_dir, f"{item['image_id']}.json")) as f:
        captions = json.load(f)['prompts_for_generation']
      labels_prompts = []
      for label_prompt in captions:
        label, prompt = list(label_prompt.items())[0]
        new_label, _ = create_prompt(label)
        labels_prompts.append([label, new_label, prompt])
      return labels_prompts
  else:
    raise Exception('Exactly one of `label_dir` or `caption_dir` must be set')
    
  def check_files(save_dir):
    inputs = []
    completed_files = []
    
    for item in tqdm(dataset):
      for label, new_label, prompt in extract_annotations(item):
        filepath = os.path.join(save_dir, label, f'{item["image_id"]}.json')
        if not os.path.exists(filepath):
          inputs.append((item['file_name'], prompt, label, new_label))
        else:
          completed_files.append(filepath)
    return sorted(set(inputs)), sorted(set(completed_files))

  return class_labels, class_labels_for_prompts, check_files

  
detect_dataset_helper_registry = {
  'artdl': artdl,
  'iconart': iconart
}
