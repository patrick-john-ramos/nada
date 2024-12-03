# NADA
Official code for No Annotations for Object Detection in Art through Stable Diffusion (WACV 2025)

# Setup

This repository is composed of three folders corresponding to different parts of training or evaluting NADA. The code is organized this way to prevent conflicting dependicies.

* `prompt-to-prompt`

	This folder contains code for the class proposers not based on LLaVA and the class-conditioned detector. This uses code from [Google's prompt-to-prompt repository](https://github.com/google/prompt-to-prompt) and [DAAM](https://github.com/castorini/daam).

	Create a Python virtual environment and pip install the corresponding requirements file to set up the folder.

	* For the class-conditioned detector and weakly-supervised class proposer
		
		```bash
		cd prompt-to-prompt
		python -m venv env
		source env/bin/activate
		pip install -r requirements.txt
		```

	* For the non-LLaVA zero-shot class proposers

		```bash
		cd prompt-to-prompt
		python -m venv cp_env
		source cp_env/bin/activate
		pip install -r cp_requirements.txt
		```

* `detectron2`

	Code for evaluating predictions made by NADA. Bounding boxes are saved in the COCO format, so we use [Meta's Detectron2 library](https://github.com/facebookresearch/detectron2) to evaluate them.

	Create a virtual environment and pip install from `requirements.txt` to set it up.

	```bash
	cd detectron2
	python -m venv env
	source env/bin/activate
	pip install -r requirements.txt
	```

* `LLaVA`

	Code for generating outputs with LLaVA. We use LLaVA for our zero-shot class proposer and for caption prompt construction. This uses code from the [official LLaVA repository](https://github.com/haotian-liu/LLaVA/tree/main).

	Create a Python environment and install from folder to set it up.

	```bash
	cd LLaVA
	python -m venv env
	source env/bin/activate
	pip install -e .
	```

# Training

## Class proposer

### Weakly-supervised class proposer

`fc.py` contains code for training and evaluating the weakly-supervised class propser. See the example below for how to use it. Items in `{}` are options.

```bash
python classify/fc.py \
--dataset {artdl, iconart} \
--classification-type {single, multi} \
--data-type images \
--modes {train, eval, label} \
--num-layers {2, 3} \
--checkpoint checkpoints/{artdl, iconart}/checkpoint.ckpt \
--eval-label-split test_detect \
--save-dir labels/{artdl, iconart}
```

### Zero-shot class proposer

## Class-conditioned

# Evaluating
