from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_pascal_voc
from sklearn.model_selection import train_test_split

import os
import json
from functools import partial
from itertools import chain


ANNOTATIONS_DIR = 'annotations'


def load_dataset(filepath, img_dir):
  with open(filepath, 'r') as f:
    dataset = json.load(f)
  for item in dataset:
    item['file_name'] = os.path.join(img_dir, item['file_name'])
  return dataset


def load_artdl_dataset(split, img_dir):
  return load_dataset(os.path.join(ANNOTATIONS_DIR, f'artdl_{split}_annotations.json'), img_dir)


def setup(base_dataset_name, split, dataset_fn, img_dir, thing_classes):
  dataset_name = f'{base_dataset_name}_{split}'
  try:
    DatasetCatalog.remove(dataset_name)
    MetadataCatalog.remove(dataset_name)
    print(f'{dataset_name} existed, removed first before registering')
  except:
    pass
  DatasetCatalog.register(dataset_name, partial(dataset_fn, split=split, img_dir=img_dir))
  MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
  print(MetadataCatalog.get(dataset_name))


def setup_artdl():
  labels = [
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
  for split in ('train', 'val', 'test',):
    setup('artdl', split, load_artdl_dataset, '../../old_WSOD_art/datasets_orig/ArtDL/JPEGImages/', labels)


def setup_iconart():
  labels = [
      'Saint_Sebastien',
      # 'turban',
      'crucifixion_of_Jesus',
      'angel',
      # 'capital',
      'Mary',
      # 'beard',
      'Child_Jesus',
      'nudity',
      'ruins'
  ]
  labels = [label.lower().replace('_', ' ') for label in labels]
  for split in ('train', 'val', 'test'):
    setup('iconart', split, load_iconart_dataset, '../../old_WSOD_art/datasets_orig/IconArt/JPEGImages', labels)
