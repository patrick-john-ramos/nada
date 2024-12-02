import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, DefaultDataCollator
from datasets import Dataset, DatasetDict
from datasets import Image as hfImage
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
import lovely_tensors as lt

from data import classify_with_labels_dataset_helper_registry

from argparse import ArgumentParser
import pickle
from pprint import pprint
import json

if __name__ == '__main__':

  # command-line arguments
  parser = ArgumentParser(
    prog='classify_data',
    description='classify data'
  )
  parser.add_argument('--dataset', type=str, help='Which dataset to generate for')
  parser.add_argument('--dataset-split', type=str, help='Which split of dataset to classify')
  parser.add_argument('--save-dir', type=str, help='Where to save annotations')
  parser.add_argument('--device', type=int, default=0, help='Which device to use if not using Accelerate')
  parser.add_argument('--accelerate', action='store_true', help='Whether or not to use Accelerate')
  
  args = parser.parse_args()

  lt.monkey_patch()

  use_accelerate = False

  # prepare model
  if use_accelerate:
    model_kwargs = dict(torch_dtype=torch.float16)
  else:
    model_kwargs = dict(load_in_8bit=True, device_map={'': args.device})
  # model_kwargs ={}
  model_id = "Salesforce/instructblip-vicuna-7b"
  device = torch.device("cuda")
  model = InstructBlipForConditionalGeneration.from_pretrained(
      model_id,
      torch_dtype=torch.float16,
      **model_kwargs
  )
  processor = InstructBlipProcessor.from_pretrained(model_id)

  # prepare data
  class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
  class_labels = np.array(class_labels)
  dataset = dataset[args.dataset_split]
  print('Dataset')
  print(dataset)

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
    revised_class_labels = [
      'Saint Sebastien',
      'crucifixion of Jesus',
      'angel',
      'Mary, mother of Jesus',
      'Child Jesus',
      'nudity',
      'ruins'
    ]

  print(revised_class_labels)
  # ignore completed files
  completed = []
  
  for image_id in tqdm(dataset['image_id'], dynamic_ncols=True):
    filepath = os.path.join(args.save_dir, f'{image_id}.json')
    if os.path.exists(filepath):
      completed.append(image_id)
  print(f'COMPLETED: {len(completed)}/{len(dataset)}')

  dataset = dataset.filter(lambda item: item['image_id'] not in completed)
  print(f'LEFT: {len(dataset)}')

  prompt_template = 'Is {} in this painting?'
  prompts = [prompt_template.format(class_label) for class_label in revised_class_labels]
  print(prompts)

  def repeat_dataset(dataset):
    df = dataset.to_pandas()
    return (
        Dataset.from_dict({
            col: df[col].repeat(len(class_labels)).tolist()
            for col in df.columns
        })
        .cast_column('image', hfImage())
        .add_column('prompt', prompts * len(df))
    )

  repeated_dataset = repeat_dataset(dataset.remove_columns(['file_name', 'width', 'height', 'annotations']))

  def process(item):
    return {'image_id': item['image_id']} | processor(
        images=item['image'],
        text=item['prompt'],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

  new_dataset = repeated_dataset.with_transform(process)

  dataloader = DataLoader(new_dataset, batch_size=len(class_labels), collate_fn=DefaultDataCollator())
  pprint(next(iter(dataloader)))
  print(len(dataloader), len(dataset))
  assert len(dataloader) == len(dataset), f'{len(dataloader)} {len(dataset)}'

  if use_accelerate:
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    device = accelerator.device
  else:
    device = torch.device(f'cuda:{args.device}')

  os.makedirs(args.save_dir, exist_ok=True)

  # perform inference
  with torch.no_grad():
    for image_id, inputs in tqdm(zip(dataset['image_id'], dataloader), total=len(dataset)):
      output = model.generate(
          **{k: v.to(device) for k, v in inputs.items()},
          do_sample=False,
          num_beams=5,
          max_length=256,
          min_length=1,
          top_p=0.9,
          repetition_penalty=1.5,
          length_penalty=1.0,
          temperature=1
      ).cpu()

      generated_texts = [text.strip() for text in processor.batch_decode(output, skip_special_tokens=True)]

      with open(os.path.join(args.save_dir, f'{image_id}.json'), 'w') as f:
        json.dump({'image_id': image_id, 'generated_texts': generated_texts, 'prompts': prompts}, f)
