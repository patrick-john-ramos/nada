import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
from data import classify_with_labels_dataset_helper_registry

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm.auto import tqdm

from argparse import ArgumentParser
import json
from pprint import pprint
import pickle

if __name__ == '__main__':

  # command-line arguments
  parser = ArgumentParser(
    prog='classify_data',
    description='classify data'
  )
  parser.add_argument('--dataset', type=str, help='Which dataset to generate for')
  parser.add_argument('--dataset-split', type=str, help='Which split of dataset to classify')
  parser.add_argument('--generations-dir', type=str, help='Where are LLaVA generations saved')
  # parser.add_argument('--prompt', type=str, choices=['basic', 'score', 'who', 'christian'], help='What type of prompt was used to generate the texts.')
  parser.add_argument('--save-dir', type=str, help='Where to save annotations')
  parser.add_argument('--modes', choices=['eval', 'label'], nargs='+', type=str, help='Whether to evaluate, predict, or both.')
  
  args = parser.parse_args()

  # prepare data
  class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
  class_labels = np.array(class_labels)

  # get split
  dataset = dataset[args.dataset_split]
  print('Dataset')
  print(dataset)
  
  all_outputs = []
  for id in tqdm(dataset['image_id']):
    with open(os.path.join(args.generations_dir, f'{id}.json'), 'rb') as f:
      outputs = json.load(f)['generated_texts']
    assert len(outputs) == len(class_labels), f'Output {outputs} has length {len(outputs)}, expected {len(class_labels)}'
    all_outputs.append(outputs)

  prompt_template = 'Is {} in this painting?'
  prompts = [prompt_template.format(class_label) for class_label in class_labels]

  print(prompts)
  pprint(all_outputs[:5])

  # convert texts to predictions
  preds = np.array([[1 if text.lower().startswith('yes') else 0 for text in outputs] for outputs in all_outputs])
    
  # evaluate
  if 'eval' in args.modes:

    targets = np.array(dataset['labels'])
    
    def evaluate(preds, targets):
      score_kwargs = dict(y_true=targets, y_pred=preds)
      average_kwargs= dict(average='macro')
      return{
          'acc': (preds == targets).mean(),
          'f1': f1_score(**score_kwargs, **average_kwargs),
          'precision': precision_score(**score_kwargs, **average_kwargs),
          'recall': recall_score(**score_kwargs, **average_kwargs),
          'ap': average_precision_score(y_true=targets, y_score=preds, **average_kwargs)
      }
      
    print(evaluate(preds, targets))

  # save labels
  if 'label' in args.modes:

    dataset = dataset.add_column('generated_text', all_outputs)
    dataset = dataset.add_column('preds', preds.tolist())
  
    os.makedirs(args.save_dir, exist_ok=True)
    
    for item in dataset:
      labels = class_labels[np.nonzero(item['preds'])].tolist()
      with open(os.path.join(args.save_dir, f'{item["image_id"]}.json'), 'w') as f:
        json.dump({'filename': item['file_name'], 'labels': labels, 'generated_texts': [item['generated_text']], 'prompts': prompts}, f)
