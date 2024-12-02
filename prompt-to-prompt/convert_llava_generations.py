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

if __name__ == '__main__':

  # command-line arguments
  parser = ArgumentParser(
    prog='classify_data',
    description='classify data'
  )
  parser.add_argument('--dataset', type=str, help='Which dataset to generate for')
  parser.add_argument('--dataset-split', type=str, help='Which split of dataset to classify')
  parser.add_argument('--generations-dir', type=str, help='Where are LLaVA generations saved')
  parser.add_argument('--prompt', type=str, choices=['basic', 'score', 'who', 'christian'], help='What type of prompt was used to generate the texts.')
  parser.add_argument('--threshold', type=float, help='Threshold to use when using score-based prompts. Only used when `--promtp` is "score"')
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
  
  # prepare labels for parsing generations
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

    if args.prompt in ('basic', 'who'):

      def parse_generation(text):
        def _inner_parse(_text):
          output = []
          _text = _text.lower()
          for label in revised_class_labels:
            if label.lower() in _text:
              output.append(label)
          if ('Saint Peter' not in output) and 'peter' in text.lower():
              output.append('Saint Peter')
          return output

        split_word = None
        if 'not' in text:
          split_word = 'not'
        elif 'no' in text:
          split_word = 'no'

        labels = _inner_parse(text)
        if split_word:
          first, rest = text.split(split_word, 1)
          first_labels = _inner_parse(first)
          rest_labels = _inner_parse(rest)

          if len(rest_labels) < 1:
            if len(text.splitlines()) > 1:
              lines = text.splitlines()
              labels = []
              for line in lines:
                if 'not' not in line and 'no' not in line:
                  labels += _inner_parse(line)
            elif text.count('.') > 1:
              sentences = text.split('.')
              labels = []
              for sentence in sentences:
                if 'not' not in sentence and 'no' not in sentence:
                  labels += _inner_parse(sentence)
          else:
            labels = first_labels
        return list(set(labels))

    elif args.prompt == 'score':
    
      def parse_generation(text):
        preds = eval(text.split('```')[1].removeprefix('json').removeprefix('python').strip())
        output = []
        for label in preds.keys():
          if (label not in revised_class_labels) and (label.removeprefix('Saint ') in revised_class_labels):
            fixed_label = label.removeprefix('Saint ')
            preds[fixed_label] = preds.pop(label)
            label = fixed_label
          assert label in revised_class_labels or label == 'None', f'{preds}, {label}, {label not in revised_class_labels}, {label.removeprefix("Saint ")}, {label.removeprefix("Saint ") in revised_class_labels}'
        for label in revised_class_labels:
          output.append(preds.get(label, 0))
        return output
        
  elif args.dataset == 'iconart':
    revised_class_labels = [
      'Saint Sebastien',
      'crucifixion of Jesus',
      'angel',
      'Mary, mother of Jesus',
      'Child Jesus',
      'nudity',
      'ruins'
    ]

    if args.prompt in ('basic', 'christian'):
    
      def parse_generation(text):
        def _inner_parse(_text):
          output = []
          _text = _text.lower()
          for label in revised_class_labels:
            if label.lower() in _text:
              output.append(label)
          if 'saint sebastian' in _text:
            output.append('Saint Sebastien')
          if 'nude' in _text:
            output.append('nudity')
          if 'crucifixion of jesus' not in _text and 'crucifixion' in _text:
            output.append('crucifixion of Jesus')
          return output
      
        split_word = None
        if ' not ' in text and ' no ' in text:
          if text.index(' not ') < text.index (' no '):
            split_word = ' not '
          else:
            split_word = ' no '
        elif ' not ' in text:
          split_word = ' not '
        elif ' no ' in text:
          split_word = ' no '
      
        if split_word:
          first, rest = text.split(split_word, 1)
          first_labels = _inner_parse(first)
          rest_labels = _inner_parse(rest)
      
          if len(rest_labels) == 0:
            if len(text.splitlines()) > 1:
              lines = text.splitlines()
              labels = []
              for line in lines:
                if 'not' not in line and 'no' not in line:
                  labels += _inner_parse(line)
            elif text.count('.') > 1:
              sentences = text.split('.')
              labels = []
              for sentence in sentences:
                if 'not' not in sentence and 'no' not in sentence:
                  #   print(sentence)
                  labels += _inner_parse(sentence)
            else:
              raise Exception(text, rest_labels)
          else:
            labels = first_labels
        else:
          labels = _inner_parse(text)
        return list(set(labels))

  elif args.dataset == 'deart':

    revised_class_labels = class_labels

    if args.prompt == 'basic':

      def parse_generation(text):
        def _inner_parse(_text):
          output = []
          _text = _text.lower()
          for label in revised_class_labels:
            if label.lower() in _text:
              output.append(label)
          return output

        split_word = None
        if ' not ' in text:
          split_word = ' not '
        elif ' no ' in text:
          split_word = ' no '

        if split_word:
          first, rest = text.split(split_word, 1)
          first_labels = _inner_parse(first)
          rest_labels = _inner_parse(rest)

          if len(rest_labels) == 0:
            # print(text, split_word, rest_labels)
            if len(text.splitlines()) > 1:
              lines = text.splitlines()
              labels = []
              for line in lines:
                if 'not' not in line and 'no' not in line:
                  labels += _inner_parse(line)
            elif text.count('.') > 1:
              sentences = text.split('.')
              labels = []
              for sentence in sentences:
                if 'not' not in sentence and 'no' not in sentence:
                  labels += _inner_parse(sentence)
            else:
              raise Exception(text, rest_labels)
          else:
            labels = first_labels
        else:
          labels = _inner_parse(text)
        return list(set(labels))

  all_outputs = []
  for id in tqdm(dataset['image_id']):
    with open(os.path.join(args.generations_dir, f'{id}.json')) as f:
      text = json.load(f)['generated_texts']
    all_outputs.extend(text)

  with open(os.path.join(args.generations_dir, f"{dataset['image_id'][0]}.json")) as f:
      prompt = json.load(f)['prompts'][0]

  print(prompt)
  pprint(all_outputs[:5])

  # convert texts to predictions
  parsed_texts = [parse_generation(text) for text in all_outputs]
  if args.prompt in ('basic', 'who', 'christian'):
    preds = np.array([[1 if label in text else 0 for label in revised_class_labels] for text in parsed_texts])
  elif args.prompt == 'score':
    preds = np.array([[1 if score > args.threshold else 0 for score in scores] for scores in parsed_texts])
  print(preds)
    
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
        json.dump({'filename': item['file_name'], 'labels': labels, 'generated_texts': [item['generated_text']], 'prompts': [prompt]}, f)
