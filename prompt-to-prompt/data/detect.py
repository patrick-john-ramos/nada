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
    
#   if split in 'train':
#     df = pd.read_csv('../../old_WSOD_art/datasets_orig/IconArt/ImageSets/Main/IconArt_v2.csv')
#     def check_files(save_dir):
#       inputs = []
#       completed_files = []
# 
#       selector = np.logical_and(np.logical_and(df.set == split, df[label_cols].any(axis=1)), df['Anno'].astype(bool))
#       for _, item in tqdm(df[selector].iterrows(), total=selector.sum()):
#         filename = f'{item["item"]}.jpg'
#         current_classes = item[label_cols]
#         present_classes = current_classes.keys()[current_classes.astype(bool)].map(lambda label: label.replace("_", " ").lower()).tolist()
#         for label in present_classes:
#           filepath = os.path.join(save_dir, label, f'{os.path.splitext(filename)[0]}.json')
#           if not os.path.exists(filepath):
#             new_label, prompt = create_prompt(label)
#             inputs.append((os.path.join('../../old_WSOD_art/datasets_orig/IconArt/JPEGImages', filename), prompt, label, new_label))
#           else:
#             completed_files.append(filepath)
#       return inputs, completed_files
# 
#   elif split == 'test':
# 
#     def extract_annotations(item):
#       with open(os.path.join(label_dir, f"{item['image_id']}.json")) as f:
#         labels = json.load(f)
#       return labels['labels']
#     extract_class = lambda label: class_labels.index(label)
#   
#     def check_files(save_dir):
# 
#       with open('../faster-rcnn/annotations/iconart_test_annotations.json') as f:
#         data = json.load(f)
#       # check if filepath and filename not mixed up
#       inputs = []
#       completed_files = []
#       
#       for item in tqdm(data, total=len(data)):
#         filename = os.path.basename(item['file_name'])
#         for annotation in extract_annotations(item):
#           label = class_labels[extract_class(annotation)]
#           # print(filename)
#           filepath = os.path.join(save_dir, label, f'{os.path.splitext(filename)[0]}.json')
#           # print(filepath)
#           if not os.path.exists(filepath):
#             new_label, prompt = create_prompt(label)
#             inputs.append((os.path.join('../../old_WSOD_art/datasets_orig/IconArt/JPEGImages', filename), prompt, label, new_label))
#           else:
#             completed_files.append(filepath)
#       inputs = list(set(inputs))
#       return list(set(inputs)), list(set(completed_files))
#   else:
#       raise Exception('`split` must be in ("train", "test")')

  return class_labels, class_labels_for_prompts, check_files




def voc_2007(split):
  '''Prepares variables and functions for working with Pascal VOC 2007'''
  class_labels = [
      "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
      "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
      "pottedplant", "sheep", "sofa", "train", "tvmonitor"
  ]

  def check_files(save_dir):
    with open(f'../faster-rcnn/pascal_voc_2007_{split}_annotations.json') as f:
      anns = json.load(f)

    inputs = []
    completed_files = []

    for item in tqdm(anns):
      filename = os.path.basename(item['file_name'])
      for ann in item['annotations']:
        label = class_labels[ann['category_id']]
        filepath = os.path.join(save_dir, label, f'{os.path.splitext(filename)[0]}.json')
        if not os.path.exists(filepath):
          inputs.append((os.path.join('../faster-rcnn/datasets/VOC2007/JPEGImages', filename), f'a photo of a {label}', label))
        else:
          completed_files.append(filepath)

     # one image in the PASCAL VOC annotation file might have multiple instances of the samae class
    return list(set(inputs)), list(set(completed_files))

  return class_labels, check_files


def voc_2012(split):
  '''Prepares variables and functions for working with Pascal VOC 2012'''
  class_labels = [
      "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
      "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
      "pottedplant", "sheep", "sofa", "train", "tvmonitor"
  ]
  
  def check_files(save_dir):
    with open(f'../faster-rcnn/pascal_voc_2012_{split}_annotations.json') as f:
      anns = json.load(f)

    inputs = []
    completed_files = []

    for item in tqdm(anns):
      filename = os.path.basename(item['file_name'])
      for ann in item['annotations']:
        label = class_labels[ann['category_id']]
        filepath = os.path.join(save_dir, label, f'{os.path.splitext(filename)[0]}.json')
        if not os.path.exists(filepath):
          inputs.append((os.path.join('../faster-rcnn/datasets/VOC2007/JPEGImages', filename), f'a photo of a {label}', label))
        else:
          completed_files.append(filepath)

     # one image in the PASCAL VOC annotation file might have multiple instances of the samae class
    return list(set(inputs)), list(set(completed_files))

  return class_labels, check_files


# def deart():
#   '''Prepares variables and functions for working with DEArt
# 
#   This currently is split agnostic. The authors say they use a train test split but don't provide the IDs,
#   so I'm just gonna generate everything. I'll need them anyway.
#   '''
#   
#   class_labels = [
#     'angel', 'lily', 'person', 'dove', 'halo', 'prayer', 'skull', 'crucifixion',
#     'monk', 'scroll', 'helmet', 'shield', 'nude', 'tree', 'dog', 'cow', 'arrow',
#     'devil', 'crozier', 'tiara', 'mitre', 'sword', 'lance', 'crown', 'trumpet', 
#     'palm', 'banner', 'god the father', 'centaur', 'donkey', 'camauro', 
#     'key of heaven', 'stole', 'orange', 'horse', 'eagle', 'deer', 'bird', 
#     'swan', 'book', 'jug', 'dragon', 'knight', 'lion', 'chalice', 'hands', 
#     'apple', 'monkey', 'crown of thorns', 'boat', 'serpent', 'judith', 'head', 
#     'shepherd', 'sheep', 'butterfly', 'zucchetto', 'saturno', 'cat', 'unicorn', 
#     'pegasus', 'elephant', 'mouse', 'horn', 'zebra', 'rooster', 'holy shroud', 
#     'bear', 'fish', 'banana'
#   ]
# 
#   def check_files(save_dir):
#     with open(f'../faster-rcnn/deart_all_annotations.json') as f:
#       anns = json.load(f)
# 
#     inputs = []
#     completed_files = []
# 
#     for item in tqdm(anns):
#       filename = os.path.basename(item['file_name'])
#       for ann in item['annotations']:
#         label = class_labels[ann['category_id']]
#         filepath = os.path.join(save_dir, label, f'{os.path.splitext(filename)[0]}.json')
#         if not os.path.exists(filepath):
#           inputs.append((os.path.join('../faster-rcnn', item['file_name']), f'a painting of {label}', label))
#         else:
#           completed_files.append(filepath)
# 
#     return list(set(inputs)), list(set(completed_files))
# 
#   return class_labels, check_files


def deart(split, label_dir, prompt_type=None):
  '''Prepares variables and functions for working with DEArt
  '''
  
  class_labels = [
    'angel', 'lily', 'person', 'dove', 'halo', 'prayer', 'skull', 'crucifixion',
    'monk', 'scroll', 'helmet', 'shield', 'nude', 'tree', 'dog', 'cow', 'arrow',
    'devil', 'crozier', 'tiara', 'mitre', 'sword', 'lance', 'crown', 'trumpet', 
    'palm', 'banner', 'god the father', 'centaur', 'donkey', 'camauro', 
    'key of heaven', 'stole', 'orange', 'horse', 'eagle', 'deer', 'bird', 
    'swan', 'book', 'jug', 'dragon', 'knight', 'lion', 'chalice', 'hands', 
    'apple', 'monkey', 'crown of thorns', 'boat', 'serpent', 'judith', 'head', 
    'shepherd', 'sheep', 'butterfly', 'zucchetto', 'saturno', 'cat', 'unicorn', 
    'pegasus', 'elephant', 'mouse', 'horn', 'zebra', 'rooster', 'holy shroud', 
    'bear', 'fish', 'banana'
  ]

  def extract_annotations(item):
    with open(os.path.join(label_dir, f"{item['image_id']}.json")) as f:
      labels = json.load(f)
    return labels['labels']
    
  extract_class = lambda label: class_labels.index(label)
  
  def check_files(save_dir):
    with open(f'../faster-rcnn/annotations/deart_{split}_annotations.json') as f:
      anns = json.load(f)

    inputs = []
    completed_files = []

    for item in tqdm(anns):
      filename = os.path.basename(item['file_name'])
      for label in extract_annotations(item):
        filepath = os.path.join(save_dir, label, f'{os.path.splitext(filename)[0]}.json')
        if not os.path.exists(filepath):
          inputs.append((os.path.join('../faster-rcnn', item['file_name']), f'a painting of {label}', label, label))
        else:
          completed_files.append(filepath)

    return list(set(inputs)), list(set(completed_files))

  return class_labels, class_labels, check_files


def wikiart(split, label_dir, caption_dir, prompt_type):
  '''Prepares variables and functions for working with WikiArt'''

  class_labels, dataset = classify_with_labels_dataset_helper_registry['wikiart']()
  dataset = dataset[split]
  class_labels_for_prompts = class_labels

  def create_prompt(label):
    label = class_labels_for_prompts[class_labels.index(label)]
    return label, f'a painting of {label}'
    
#   with open('cleaned_wikiart.json') as f:
#     dataset = json.load(f)
#   class_labels = class_labels_for_prompts = list(dataset.keys())
# 
#   def check_files(save_dir):
#     inputs = []
#     completed_files = []
#     
#     for label, items in tqdm(dataset.items()):
#       for item in items:
#         filepath = os.path.join(save_dir, label, f'{os.path.splitext(os.path.basename(item["image"]))[0]}.json')
#         # print('CHECKING', filepath)
#         if not os.path.exists(filepath):
#           inputs.append((os.path.join('wikiart', label, os.path.basename(item['image'])), f'a painting of a {label}', label, label))
#         else:
#           completed_files.append(filepath)
#     return sorted(set(inputs)), sorted(set(completed_files))

  if not caption_dir and label_dir:
    def extract_annotations(item):
      with open(os.path.join(label_dir, f"{item['image_id']}.json")) as f:
        labels = json.load(f)
      return [[label, *create_prompt(label)] for label in labels['labels']]
  elif caption_dir and not label_dir:
    def extract_annotations(item):
      # print(item)
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

  
detect_dataset_helper_registry = {
  'artdl': artdl,
  'iconart': iconart,
  'deart': deart,
  'wikiart': wikiart
}
