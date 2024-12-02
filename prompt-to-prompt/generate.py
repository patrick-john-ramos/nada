import torch
import torch.nn.functional as F
import diffusers
from diffusers import StableDiffusionPipeline, DDIMScheduler
from accelerate import PartialState
import numpy as np
from safetensors.torch import save_file
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from PIL import Image
import cv2
from tqdm import tqdm
import lovely_tensors as lt

from collections import OrderedDict
import json
import os
import glob
from argparse import ArgumentParser
from types import SimpleNamespace
import pickle

from ptp import NullInversion, NUM_DDIM_STEPS, GUIDANCE_SCALE, AttentionStore, text2image_ldm_stable
from heatmap import extract_heat_maps, compute_token_merge_indices
from data import detect_dataset_helper_registry
from verify import verify_annotations

try:
  from google.colab import userdata
  from huggingface_hub import login
  login(userdata.get('huggingface'))
  print('colab runtime')
except:
  print('local runtime')

assert diffusers.__version__ == '0.10.0'


lt.monkey_patch()

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
MAX_NUM_WORDS = 77
distributed_state = PartialState()
ldm_stable = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-base', scheduler=scheduler).to(distributed_state.device)


try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


null_inversion = NullInversion(ldm_stable)


def invert_and_reconstruct(image, prompt, pipe, seed):

  # prepare image
  if type(image) is str:
    og_image = Image.open(image).convert('RGB')
  else:
    og_image = image
  image = np.array(og_image.resize((512, 512)))[:, :, :3]

  # invert image
  (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image, prompt, offsets=(0,0,200,0))

  # reconstruct image
  controller = AttentionStore()
  image_inv, x_t = text2image_ldm_stable(
      pipe,
      [prompt],
      controller,
      latent=x_t,
      num_inference_steps=NUM_DDIM_STEPS,
      guidance_scale=GUIDANCE_SCALE,
      generator=torch.Generator().manual_seed(seed),
      uncond_embeddings=uncond_embeddings
  )
  return SimpleNamespace(
    image=og_image,
    rec_image=Image.fromarray(image_inv[0]).resize(og_image.size),
    controller=controller
  )


def detect(image, prompt, term, label2id, pipe, seed, heatmap_dim, return_controller=False):

  # invert image and reconstruct
  rec_output = invert_and_reconstruct(image, prompt, pipe, seed)
    
  # extract heat map
  merge_idxs, _ = compute_token_merge_indices(pipe.tokenizer, prompt, term)
  heat_maps = extract_heat_maps(rec_output.controller, heatmap_dim)
  heat_map = heat_maps[merge_idxs].mean(axis=0)

  # preprocessing: normalization, shifting
  norm_heat_map = (heat_map.numpy() * 255).astype(np.uint8)
  reshaped_heat_map = np.tile(np.expand_dims(norm_heat_map, axis=-1), (1, 1, 3))
  shifted_heat_map = cv2.pyrMeanShiftFiltering(reshaped_heat_map, 11, 11)

  # grayscaling and thresholding
  gray_heat_map = cv2.cvtColor(shifted_heat_map, cv2.COLOR_BGR2GRAY)
  _, thresh_heat_map = cv2.threshold(gray_heat_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  # thresh_heat_map = (gray_heat_map > threshold_triangle(gray_heat_map)).astype(np.uint8)

  # compute markers as local maxima
  dists = ndimage.distance_transform_edt(thresh_heat_map)
  loc_max_idxs = peak_local_max(dists, min_distance=20, labels=thresh_heat_map)
  loc_max = np.zeros_like(thresh_heat_map)
  loc_max[loc_max_idxs[:, 0], loc_max_idxs[:, 1]] = 1
  markers, _ = ndimage.label(loc_max, structure=np.ones((3, 3)))

  # watershed method to get mask
  mask = watershed(-dists, markers, mask=thresh_heat_map)

  # create bounding boxes from masks and overlay on image
  bboxes = []
  og_w, og_h = rec_output.image.size[:2]
  height_scale_factor, width_scale_factor = round(og_h/mask.shape[0]), round(og_w/mask.shape[1])
  for val in np.unique(mask)[1:]:
    _mask = (mask == val).astype(int)
    contours, _ = cv2.findContours(_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      x, y, w, h = x*width_scale_factor, y*height_scale_factor, w*width_scale_factor, h*height_scale_factor
      bboxes.append({'id': label2id[term], 'bbox': [x, y, w, h]})
      
  # return og_image, bboxes, heat_map
  return SimpleNamespace(
    image=rec_output.image,
    bboxes=bboxes,
    rec_image=rec_output.rec_image,
    heat_map=heat_map,
    mask=mask,
    controller=rec_output.controller if return_controller else None
  )


h = (ldm_stable.unet.config.sample_size * ldm_stable.vae_scale_factor)
latent_hw = 4096 if h == 512 else 9216
heatmap_dim = int(np.sqrt(latent_hw))
batch_size = 16
seed = 42


if __name__ == '__main__':

  # command-line arguments
  parser = ArgumentParser(
    prog='generate_data',
    description='generate synthetic art wsod data'
  )
  parser.add_argument('-s', '--split', default=1, type=int, help='How many sub-arrays to split prompts into before splitting between devices. Use this to split the data across different running instances of this script on the same gpu/s.')
  parser.add_argument('-i', '--index', default=0, type=int, help='Which sub-array of prompts to split between devices. Use this to split the data across different running instances of this script on the same gpu/s.')
  parser.add_argument('-d', '--dataset', type=str, help='Which dataset to generate for. Must be from ["artdl", "iconart_{split}", "deart"], where `split` is a valid split e.g. "train", "test", "val"')
  parser.add_argument('--dataset-split', type=str, help='Which split of dataset to generate for.')
  parser.add_argument('--prompt-type', type=str, help='Which prompt to use. Check `data/detect.py` to see valid prompt types per dataset.')
  parser.add_argument('-a', '--save-dir', type=str, help='Where to save annotations. An empty string indicates no saving.')
  parser.add_argument('-l', '--label-dir', type=str, help='Where to load labels from')
  parser.add_argument('--caption-dir', type=str, help='Where to load captions from, if using captions.')
  parser.add_argument('-p', '--prioritize-existing-files', action='store_true', help='Assumes all existing files are in completed files when counting files at the end of verification. Can use when generating on so many devices that new files are generated during the identification of completed_files.')

  args = parser.parse_args()

  save_dir = args.save_dir

  # prepare labels
  class_labels, class_labels_for_prompts, _ = detect_dataset_helper_registry[args.dataset](split=args.dataset_split, label_dir=args.label_dir, prompt_type=args.prompt_type, caption_dir=args.caption_dir)
  label2id = OrderedDict({label: id for id, label in enumerate(class_labels_for_prompts)})

  # determine which files to generate
  verified, inputs, completed_files, existing_files = verify_annotations(args.dataset, args.dataset_split, args.save_dir, args.prioritize_existing_files, verbose=True, label_dir=args.label_dir, prompt_type=args.prompt_type, caption_dir=args.caption_dir)
  verified = verified or args.prioritize_existing_files
  assert verified, 'File verification failed'
  
  print(f'COMPLETED: {len(existing_files)}/{len(completed_files) + len(inputs)}')
  print(f'LEFT: {len(inputs)}')

  # create folders
  for label in class_labels:
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)

  # inputs per script
  inputs = sorted(inputs)
  inputs = np.array_split(np.array(inputs), args.split)[args.index].tolist()
  print(f'LEFT FOR SCRIPT: {len(inputs)}')

  # generate
  with distributed_state.split_between_processes(inputs) as inputs_:
    print(f'LEFT FOR DEVICE: {len(inputs_)}')
    print(inputs_[:4])
    print(distributed_state.device)
    for filepath, prompt, label_for_saving, label in inputs_:
      output = detect(
          image=filepath,
          prompt=prompt,
          term=label,
          label2id=label2id,
          pipe=ldm_stable,
          seed=seed,
          heatmap_dim=heatmap_dim
      )
      filename = os.path.basename(filepath)
      if save_dir:
        with open(os.path.join(save_dir, label_for_saving, f'{os.path.splitext(filename)[0]}.json'), 'w') as file:
          json.dump({'filename': filename, 'annotations': output.bboxes, 'heat_map': output.heat_map.detach().numpy().tolist(), 'prompt': prompt, 'seed': seed}, file)
