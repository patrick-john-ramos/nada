from argparse import ArgumentParser
import json
import os
import sys

import numpy as np
from tqdm.auto import tqdm
import lovely_tensors as lt

sys.path.append(os.path.abspath('../prompt-to-prompt'))
from data import classify_with_labels_dataset_helper_registry
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import *


def infer(model, tokenizer, image_processor, args):
  model_name = args.model_name
  qs = args.query
  image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
  if IMAGE_PLACEHOLDER in qs:
      if model.config.mm_use_im_start_end:
          qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
      else:
          qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
  else:
      if model.config.mm_use_im_start_end:
          qs = image_token_se + "\n" + qs
      else:
          qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

  if "llama-2" in model_name.lower():
      conv_mode = "llava_llama_2"
  elif "mistral" in model_name.lower():
      conv_mode = "mistral_instruct"
  elif "v1.6-34b" in model_name.lower():
      conv_mode = "chatml_direct"
  elif "v1" in model_name.lower():
      conv_mode = "llava_v1"
  elif "mpt" in model_name.lower():
      conv_mode = "mpt"
  else:
      conv_mode = "llava_v0"

  if args.conv_mode is not None and conv_mode != args.conv_mode:
      print(
          "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
              conv_mode, args.conv_mode, args.conv_mode
          )
      )
  else:
      args.conv_mode = conv_mode

  conv = conv_templates[args.conv_mode].copy()
  conv.append_message(conv.roles[0], qs)
  conv.append_message(conv.roles[1], None)
  prompt = conv.get_prompt()

  image_files = image_parser(args)
  images = load_images(image_files)
  image_sizes = [x.size for x in images]
  images_tensor = process_images(
      images,
      image_processor,
      model.config
  ).to(model.device, dtype=torch.float16)

  input_ids = (
      tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
      .unsqueeze(0)
      .cuda()
  )

  with torch.inference_mode():
      output_ids = model.generate(
          input_ids,
          images=images_tensor,
          image_sizes=image_sizes,
          do_sample=True if args.temperature > 0 else False,
          temperature=args.temperature,
          top_p=args.top_p,
          num_beams=args.num_beams,
          max_new_tokens=args.max_new_tokens,
          use_cache=True,
      )

  outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
  return outputs

  
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
  parser.add_argument('--save-dir', type=str, help='Where to save captions')
  parser.add_argument('--label-dir', type=str, help='Where to load labels from')

  args = parser.parse_args()
  
  # prepare data
  class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
  class_labels = np.array(class_labels)

  # get split
  _dataset = dataset[args.dataset_split]
  print('Dataset')
  print(_dataset)

  # ignore completed files
  completed = []
  
  for image_id in tqdm(_dataset['image_id'], dynamic_ncols=True):
    filepath = os.path.join(args.save_dir, f'{image_id}.json')
    if os.path.exists(filepath):
      completed.append(image_id)
  print(f'COMPLETED: {len(completed)}/{len(_dataset)}')

  _dataset = _dataset.filter(lambda item: item['image_id'] not in completed)
  print(f'LEFT: {len(_dataset)}')
  
  # split data
  idxs = np.arange(len(_dataset))
  _dataset =  _dataset.select(np.array_split(idxs, args.split)[args.index])
  print('Dataset after splitting')
  print(f'LEFT FOR SCRIPT: {len(_dataset)}')

  # prepare model
  model_path = "liuhaotian/llava-v1.6-34b"
  model_base = None
  model_name = get_model_name_from_path(model_path)

  disable_torch_init()
  
  tokenizer, model, image_processor, context_len = load_pretrained_model(
      model_path, model_base, model_name
  )

  # create save directory
  os.makedirs(args.save_dir, exist_ok=True)

  # change labels if necessary
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

  class_label_to_revised_class_label = dict(zip(class_labels, revised_class_labels))
    
  prompt = 'Describe the visual elements in the image in one sentence. Include the term "{}".'
  
  print(prompt)

  # caption
  for item in tqdm(_dataset):
    with open(os.path.join(args.label_dir, f"{item['image_id']}.json")) as f:
      labels = json.load(f)['labels']

    prompts = []
    captions = []
    for label in labels:
      revised_class_label = class_label_to_revised_class_label[label]
      _prompt = prompt.format(revised_class_label)
      gen_args = type('Args', (), {
          "model_path": model_path,
          "model_base": None,
          "model_name": get_model_name_from_path(model_path),
          "query": _prompt,
          "conv_mode": None,
          "image_file": item['file_name'],
          "sep": ",",
          "temperature": 0,
          "top_p": None,
          "num_beams": 1,
          "max_new_tokens": 512
      })()
      outputs = infer(model, tokenizer, image_processor, gen_args)
      prompts.append({label: _prompt})
      captions.append({label: outputs})
      
    
    with open(os.path.join(args.save_dir, f'{item["image_id"]}.json'), 'w') as f:
      json.dump({'filename': item['file_name'], 'captions': captions, 'prompts': prompts}, f)
