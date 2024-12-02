import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
from data import classify_with_labels_dataset_helper_registry

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm.auto import tqdm

from data import classify_with_labels_dataset_helper_registry

from argparse import ArgumentParser
from collections import OrderedDict
import os
import json


pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=0)


if __name__ == '__main__':

  # command-line arguments
  parser = ArgumentParser(
    prog='classify_data',
    description='classify data'
  )
  parser.add_argument('-d', '--dataset', type=str, help='Which dataset to generate for')
  parser.add_argument('-t', '--threshold', type=float, help='Threshold for CLIP-classification. Labels with a score under the threshold will be filtered out.')
  parser.add_argument('-a', '--save-dir', type=str, help='Where to save annotations')
  parser.add_argument('--eval-splits', nargs='+')
  parser.add_argument('--split-to-label', type=str)
  parser.add_argument('--modes', choices=['eval', 'label'], nargs='+', type=str, help='Whether to evaluate, predict, or both.')
  args = parser.parse_args()

  # prepare data
  class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
  class_labels_np = np.array(class_labels)

  # classify
  outputs = {split: [] for split in (args.eval_splits if args.eval_splits else args.split_to_label)}
  for split in args.eval_splits:
    _outputs = []
    for out in tqdm(pipe(KeyDataset(dataset[split], key='image'), candidate_labels=class_labels, hypothesis_template='a painting of {}')):
      _outputs.append([item['score'] for item in sorted(out, key=lambda item: class_labels.index(item['label']))])
    _outputs = np.array(_outputs)
    outputs[split] = _outputs

  # evaluate
  if 'eval' in args.modes:
    print(args.eval_splits)
    for split, _outputs in outputs.items():
      targets = np.array(dataset[split]['labels'])
      preds = (_outputs>args.threshold).astype(int)
      score_kwargs = dict(y_true=targets, y_pred=preds)
      average_kwargs= dict(average='macro')
      print(dict(
          split=split,
          accuracy=(preds == targets).mean(),
          f1=f1_score(**score_kwargs, **average_kwargs),
          precision=precision_score(**score_kwargs, **average_kwargs),
          recall=recall_score(**score_kwargs, **average_kwargs),
          ap=average_precision_score(y_true=targets, y_score=preds, **average_kwargs)
      ))

  # label
  if 'label' in args.modes:

    # create folders
    os.makedirs(args.save_dir, exist_ok=True)

    _outputs = outputs[args.split_to_label]
    dataset[args.split_to_label] = (
      dataset[args.split_to_label]
      .add_column('scores', _outputs.tolist())
      .add_column('preds', (_outputs>args.threshold).astype(int).tolist())
    )

    for item in tqdm(dataset[args.split_to_label]):
      labels = class_labels_np[np.nonzero(item['preds'])].tolist()
      filename = os.path.basename(item['file_name'])
      with open(os.path.join(args.save_dir, f'{item["image_id"]}.json'), 'w') as f:
        json.dump({'filename': filename, 'scores': item['scores'], 'threshold': args.threshold, 'labels': labels}, f)
      
