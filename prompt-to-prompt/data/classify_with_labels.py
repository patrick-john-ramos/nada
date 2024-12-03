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
    output['file_name'] = '../data/ArtDL/JPEGImages/' + item['file_name']
    output['image'] = output['file_name']
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
      pd.read_csv('../data/ArtDL/ArtDL.csv')
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
    load_dataset(f'../detectron2/annotations/artdl_{split}_annotations.json', class_labels)
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
    .cast_column('image', hf_Image())
  )
  
  return class_labels, dataset


def iconart():
  '''Prepares variables and functions for working with IconArt'''
  
  def preprocess(item):
    output = {}
    output['file_name'] = '../data/IconArt_v1/JPEGImages/' + item['file_name']
    output['image'] = output['file_name']
    labels = np.zeros(len(class_labels))
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

  df = pd.read_csv('../data/IconArt_v1/ImageSets/Main/IconArt_v1.csv')

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
  test_detect_dataset = load_dataset(f'../detectron2/annotations/iconart_test_annotations.json', class_labels)

  dataset = (
    DatasetDict({
      'train': Dataset.from_list(train_dataset),
      'test_classify': Dataset.from_list(test_classify_dataset),
      'test_detect': Dataset.from_list(test_detect_dataset)
    })
    .map(preprocess)
    .cast_column('image', hf_Image())
  )

  train_val_dataset = dataset['train'].train_test_split(test_size=0.3, seed=42)
  dataset['train'] = train_val_dataset['train']
  dataset['val_classify'] = train_val_dataset['test']

  return class_labels, dataset


classify_with_labels_dataset_helper_registry = {
	'artdl': artdl,
	'iconart': iconart,
}
