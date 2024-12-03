from datasets import Dataset, DatasetDict
from datasets import Image as hf_Image
import numpy as np
import pandas as pd


from itertools import chain
import json
import os


def load_dataset(filepath, class_labels, img_dir=''):
  with open(filepath, 'r') as f:
    dataset = json.load(f)
  for item in dataset:
    item['file_name'] = os.path.join(img_dir, item['file_name'])
    labels = np.zeros(len(class_labels), dtype=int)
    labels[list({ann['category_id'] for ann in item['annotations']})] = 1
    item['labels'] = labels
  return dataset

def artdl():

  def preprocess(item):
    output = {}
    # output['file_name'] = '../faster-rcnn/' + item['file_name']
    output['file_name'] = '../../old_WSOD_art/datasets_orig/ArtDL/JPEGImages/' + item['file_name']
    output['image'] = output['file_name']
    # output['file_name'] = '../../old_WSOD_art/datasets_orig/IconArt/JPEGImages/' + item['file_name']
    labels = np.zeros(len(class_labels))
    labels[list({ann['category_id'] for ann in item['annotations']})] = 1
    output['labels'] = labels.astype(int)
    return output
    
  class_labels = [
  	'antony of padua',
  	'john the baptist',
  	'paul',
  	'francis',
  	'mary magdalene',
  	'jerome',
  	'dominic',
  	'mary',
  	'peter',
  	'sebastian'
  ]

  df = (
      pd.read_csv('../../old_WSOD_art/datasets_orig/ArtDL/ArtDL.csv')
      .drop(
          columns=[
              '11H(ANTONY ABBOT)',
              '11H(AUGUSTINE)',
              '11H(JOHN)',
              '11H(JOSEPH)',
              '11H(STEPHEN)',
              '11HH(BARBARA)',
              '11HH(CATHERINE)',
              'John Baptist - Child',
              'John Baptist - Dead'
          ]
      )
      .set_index('item')
  )

  classify_datasets = []

  for split in ('train', 'val', 'test'):
    _df = df[np.logical_and(df.set == split, df.drop(columns=['set']).any(axis=1))].drop(columns='set').astype(bool)
    _df = _df.rename(columns=lambda label : class_labels.index(label[label.index('(')+1:label.index(')')].lower()))

  
    _dataset = []
    for image_id, data in _df.iterrows():
      # duplicate file_name key as image to convert to actual image later
      ann = {'image_id': image_id, 'file_name': f'{image_id}.jpg', 'width': None, 'height': None, 'annotations': []}
      for label, present in data.items():
        if present:
          ann['annotations'].append({'category_id': label})
      _dataset.append(ann)
    classify_datasets.append(_dataset)
    
  train_dataset, val_classify_dataset, test_classify_dataset = classify_datasets
    
  val_detect_dataset, test_detect_dataset = [
    load_dataset(f'../faster-rcnn/annotations/artdl_{split}_annotations.json', class_labels)
    for split in ('val', 'test')
  ]

  dataset = (
    DatasetDict({
      'train': Dataset.from_list(train_dataset),
      'val_classify': Dataset.from_list(val_classify_dataset),
      'test_classify': Dataset.from_list(test_classify_dataset),
      'val_detect': Dataset.from_list(val_detect_dataset),
      'test_detect': Dataset.from_list(test_detect_dataset)
    })
    .map(preprocess)
    # .rename_column('file_name', 'image')
    .cast_column('image', hf_Image())
  )
  
  return class_labels, dataset


def iconart():
  '''Prepares variables and functions for working with IconArt'''
  
  def preprocess(item):
    output = {}
    # output['file_name'] = '../faster-rcnn/' + item['file_name']
    output['file_name'] = '../../redownloaded_datasets/IconArt_v1/JPEGImages/' + item['file_name']
    output['image'] = output['file_name']
    labels = np.zeros(len(class_labels))
    # print(item)
    labels[list({ann['category_id'] for ann in item['annotations']})] = 1
    output['labels'] = labels.astype(int)
    return output

  label_cols = [
    'Saint_Sebastien',
    'crucifixion_of_Jesus',
    'angel',
    'Mary',
    'Child_Jesus',
    'nudity',
    'ruins'
  ]

  class_labels = [label.replace('_', ' ').lower() for label in label_cols]

  df = pd.read_csv('../../redownloaded_datasets/IconArt_v1/ImageSets/Main/IconArt_v1.csv')
  # df = df[df[label_cols].any(axis=1)]
  # df['labels'] = df.apply(lambda row: row[label_cols].astype(int).tolist(), axis=1)
  # df['file_name'] = df.item.apply(lambda image_id: f'{image_id}.jpg')
  # df['image'] = df.file_name.apply(lambda file_name: os.path.join('../../old_WSOD_art/datasets_orig/IconArt/JPEGImages', file_name))

  # Refer to IconArt README.txt for Anno (whether image has associated bounding boxes).
  # Since this is for classification, we don't care if an image has bounding boxes or not.
  # df = (
  #    df
  #    .drop(columns=label_cols+['Anno'])
  #    .rename(columns={'item': 'image_id'})
  # )[['image_id', 'file_name', 'image', 'labels', 'set']]
  # train_df = df[df.set == 'train']

  classify_datasets = []
  for split in ('train', 'test'):
    _df = df[np.logical_and(df.set == split, df[label_cols].any(axis=1))].drop(columns=['Anno', 'set']).set_index('item').astype(bool)
    _df = _df.rename(columns=lambda label : class_labels.index(label.replace('_', ' ').lower()))

    _dataset = []
    for image_id, data in _df.iterrows():
      # duplicate file_name key as image to convert to actual image later
      ann = {'image_id': image_id, 'file_name': f'{image_id}.jpg', 'width': None, 'height': None, 'annotations': []}
      for label, present in data.items():
        if present:
          ann['annotations'].append({'category_id': label})
      _dataset.append(ann)

    classify_datasets.append(_dataset)

  train_dataset, test_classify_dataset = classify_datasets
  test_detect_dataset = load_dataset(f'../faster-rcnn/annotations/iconart_test_annotations.json', class_labels)

  dataset = (
    DatasetDict({
      'train': Dataset.from_list(train_dataset),
      'test_classify': Dataset.from_list(test_classify_dataset),
      'test_detect': Dataset.from_list(test_detect_dataset)
    })
    .map(preprocess)
    # .rename_column('file_name', 'image')
    .cast_column('image', hf_Image())
  )
  
  
  # dataset = (
  #   DatasetDict({
  #       split: Dataset.from_pandas(df[df.set == split]).remove_columns('__index_level_0__')
  #       for split in df.set.unique()
  #   })
  #   .cast_column('image', hf_Image())
  # )

  train_val_dataset = dataset['train'].train_test_split(test_size=0.3, seed=42)
  dataset['train'] = train_val_dataset['train']
  dataset['val_classify'] = train_val_dataset['test']

  return class_labels, dataset


def deart():
  '''Prepares variables and functions for working with DEArt

  This currently is split agnostic. The authors say they use a train test split but don't provide the IDs,
  so I'll split it myself
  '''

  def preprocess(item):
    output = {}
    output['image'] = item['file_name']
    labels = np.zeros(len(class_labels))
    labels[list({ann['category_id'] for ann in item['annotations']})] = 1
    output['labels'] = labels.astype(int)
    return output
  
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

  dataset = (
    DatasetDict({
      split: Dataset.from_list(load_dataset(f'../faster-rcnn/annotations/deart_{split}_annotations.json', class_labels, '../faster-rcnn'))
      for split in ('train', 'val', 'test')
    })
    .map(preprocess)
    .cast_column('image', hf_Image())
  )
  
  return class_labels, dataset

def wikiart():
  '''Prepares variables and functions for custom WikiArt data'''

  with open('../prompt-to-prompt/cleaned_wikiart.json') as f:
    _dataset = json.load(f)

  class_labels = class_labels_for_prompts = list(_dataset.keys())
  label2id = {label: i for i, label in enumerate(class_labels)}

  def preprocess(item):
    output = {}
    output['file_name'] = os.path.join('../prompt-to-prompt/wikiart', item['file_name'])
    output['image'] = output['file_name']
    labels = np.zeros(len(class_labels))
    labels[list({ann['category_id'] for ann in item['annotations']})] = 1
    output['labels'] = labels.astype(int)
    return output
  
  dataset = {}
  for label, items in _dataset.items():
    label = label2id[label]
    for item in items:
      item['image_id'] = item.pop('id')
      item['file_name'] = os.path.basename(item['image'])
      if dataset.get(item['image_id']) is None:
        dataset[item['image_id']] = item
      cur_item = dataset[item['image_id']]
      if cur_item.get('annotations') is None:
        cur_item['annotations'] = [{'category_id': label}]
      else:
        cur_item['annotations'].append({'category_id': label})
  dataset = list(dataset.values())

  dataset = (
    DatasetDict({'all': Dataset.from_list(dataset)})
    .map(preprocess)
    .cast_column('image', hf_Image())
  )
  
  return class_labels, dataset

classify_with_labels_dataset_helper_registry = {
	'artdl': artdl,
	'iconart': iconart,
	'deart': deart,
	'wikiart': wikiart
}
