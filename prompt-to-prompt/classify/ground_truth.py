import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
from data import classify_with_labels_dataset_helper_registry

import numpy as np
from tqdm.auto import tqdm
import lovely_tensors as lt

from argparse import ArgumentParser
import json

if __name__ == '__main__':

	lt.monkey_patch()

	os.environ['TOKENIZERS_PARALLELISM'] = "false"

	# command-line arguments
	parser = ArgumentParser(
		prog='classify_data',
		description='classify data'
	)
	parser.add_argument('--dataset', type=str, help='Which dataset to generate for')
	parser.add_argument('--dataset-split', type=str, help='Which split of dataset to classify')
	parser.add_argument('--save-dir', type=str, help='Where to save annotations')
	
	args = parser.parse_args()

	# prepare data
	class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
	class_labels = np.array(class_labels)

	# get split
	dataset = dataset[args.dataset_split]
	print('Dataset')
	print(dataset)

	# save labels
	os.makedirs(args.save_dir, exist_ok=True)
	
	for item in tqdm(dataset):
		with open(os.path.join(args.save_dir, f'{item["image_id"]}.json'), 'w') as f:
			json.dump(
				{
					'filename': item['file_name'],
					'labels': list({class_labels[ann['category_id']] for ann in item['annotations']})
				},
				f
			)
