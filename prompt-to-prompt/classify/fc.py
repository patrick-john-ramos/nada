import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
from data import classify_with_labels_dataset_helper_registry

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics import Accuracy, Precision, Recall, F1Score, AveragePrecision
from transformers import (
  CLIPVisionModel, CLIPVisionModelWithProjection, CLIPModel, AutoProcessor,
  get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import DatasetDict
from safetensors.torch import load_file, save_file
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import wandb
from tqdm.auto import tqdm
import lovely_tensors as lt

from argparse import ArgumentParser
from itertools import product
from functools import partial
import json


class LitModel(pl.LightningModule):
  '''MLP classifier'''
  
  def __init__(
    self,
    num_layers,
    input_dim,
    num_classes,
    task,
    dropout=0,
    lr=1e-3,
    weight_decay=1e-3,
    lr_decay=None,
    warmup_ratio=0,
    weight=None
  ):
    super().__init__()
    self.model = create_mlp(num_layers, input_dim, num_classes, dropout)

    self.task = task
    self.num_classes = num_classes
    for metric in (Accuracy, Precision, Recall, F1Score, AveragePrecision):
      for split in ('val', 'test'):
        kwargs = {}
        if metric != Accuracy:
          kwargs['average'] = 'macro'
        if task == 'multiclass':
          kwargs['num_classes'] = num_classes
        elif task == 'multilabel':
          kwargs['num_labels'] = num_classes
        else:
          raise ValueError('`task` must be in ("multiclass", "multilabel")')
        setattr(
            self,
            f'{split}_{metric.__name__.lower()}',
            metric(task=task, **kwargs)
        )
    self.loss = nn.CrossEntropyLoss(weight=weight) if task == 'multiclass' else nn.BCEWithLogitsLoss(weight=weight)

    self.logits_process = (lambda logits: torch.argmax(logits, dim=1)) if task == 'multiclass' else (lambda logits: logits)
    self.labels_process = (lambda logits: logits) if task == 'multiclass' else (lambda logits: logits.float())

    self.lr = lr
    self.weight_decay = weight_decay
    assert lr_decay in (None, 'linear', 'cosine'), f"`lr_decay must be in `(None, 'linear', 'cosine')@, got `{lr_decay}`"
    self.lr_decay = lr_decay
    self.warmup_ratio = warmup_ratio
    
    self.save_hyperparameters(ignore=['model'])

  def forward(self, x):
    return self.model(x)

  def step(self, batch, stage):
    x, y = batch
    logits = self(x)
    y = self.labels_process(y)
    loss = self.loss(logits, y)
    self.log(f'{stage}/loss', loss, prog_bar=True)
    return {'loss': loss, 'logits': logits, 'y': y}

  def training_step(self, batch, batch_idx):
    return self.step(batch, 'train')['loss']

  def test_validation_step(self, batch, stage):
    output = self.step(batch, stage)
    preds = self.logits_process(output['logits'])
    for metric_name in ('accuracy', 'precision', 'recall', 'f1score', 'averageprecision'):
      metric = getattr(self, f'{stage}_{metric_name}')
      if metric_name == 'averageprecision':
        metric.update(output['logits'], output['y'].int())
      else:
        metric.update(preds, output['y'])
      self.log(f"{stage}/{metric_name}", metric, prog_bar=metric_name in ('accuracy', 'f1score'))
    return output['loss']

  def validation_step(self, batch, batch_idx):
    return self.test_validation_step(batch, 'val')

  def test_step(self, batch, batch_idx):
    return self.test_validation_step(batch, 'test')

  def predict_step(self, batch, batch_idx):
    # can't reuse self.step() because predict doesn't support logging
    x, y = batch
    return {'logits': self(x), 'y': y}

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    if self.warmup_ratio == 0 and self.lr_decay is None:
      return optimizer
    else:
      scheduler_kwargs = {'num_warmup_steps': round(self.trainer.estimated_stepping_batches*self.warmup_ratio)}
      if self.lr_decay is None:
        lr_sched_fn = get_constant_schedule_with_warmup
      else:
        scheduler_kwargs['num_training_steps'] = self.trainer.estimated_stepping_batches
        if self.lr_decay == 'linear':
          lr_sched_fn = get_linear_schedule_with_warmup
        elif self.lr_decay == 'cosine':
          lr_sched_fn = get_cosine_schedule_with_warmup
      return {
        'optimizer': optimizer,
        'lr_scheduler': {
          'scheduler': lr_sched_fn(optimizer, **scheduler_kwargs),
          'interval': 'step'
        }
      }


class LitTensorDataModule(pl.LightningDataModule):
  '''Data module for classification datasets where the input is already an embedding'''

  def __init__(
      self,
      train_embeds=None,
      train_labels=None,
      val_embeds=None,
      val_labels=None,
      test_embeds=None,
      test_labels=None,
      pred_embeds=None,
      pred_labels=None,
      batch_size=512
  ):
    super().__init__()
    self.train_embeds = train_embeds
    self.train_labels = train_labels
    self.val_embeds = val_embeds
    self.val_labels = val_labels
    self.test_embeds = test_embeds
    self.test_labels = test_labels
    self.pred_embeds = pred_embeds
    self.pred_labels = pred_labels
    self.batch_size = batch_size

  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
      self.ds_train = TensorDataset(self.train_embeds, self.train_labels)

    if stage in ("fit", "validate") or stage is None:
      self.ds_val = TensorDataset(self.val_embeds, self.val_labels)

    if stage == 'test' or stage is None:
      self.ds_test = TensorDataset(self.test_embeds, self.test_labels)

    if stage == 'predict':
      self.ds_pred = TensorDataset(self.pred_embeds, self.pred_labels)

  def train_dataloader(self):
      return DataLoader(
          self.ds_train,
          batch_size=self.batch_size,
          shuffle=True,
          generator=torch.Generator().manual_seed(42),
          num_workers=15
      )

  def val_dataloader(self):
      return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=15) # num workers suggested by running script

  def test_dataloader(self):
      return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=15) # num workers suggested by running script

  def predict_dataloader(self):
      return DataLoader(self.ds_pred, batch_size=self.batch_size, shuffle=False, num_workers=15)


def pl_train(
    lit_model_cls,
    dm_init,
    lit_model_kwargs={},
    dm_kwargs={},
    trainer_kwargs={},
    seed=12345678,
    project=None,
    name=None,
    id=None,
    log_model=False
):
  '''Trains classifier'''
  
  fit_kwargs = {}
  if project is not None and name is not None:
    resume_kwargs = {'id': id, 'resume': 'allow'} if id is not None else {}
    if id is not None and log_model:
        api = wandb.Api()
        artifact = api.artifact(f'patrickramos/{project}/model-{id}:latest')
    wandb_logger = WandbLogger(
        project=project,
        name=name,
        log_model=log_model,
        **resume_kwargs
    )
    trainer_kwargs['logger'] = wandb_logger
  pl.seed_everything(seed)
  dm = dm_init(**dm_kwargs)
  lit_model = lit_model_cls(**lit_model_kwargs)
  trainer = pl.Trainer(accelerator='auto', devices=1, num_nodes=1, **trainer_kwargs)
  trainer.fit(lit_model, dm, **fit_kwargs)
  if project is not None and name is not None:
    wandb.finish()
  return trainer

def pl_val_test_predict(
    fn,
    model=None,
    dm=None,
    trainer=None
):
  '''General function for handling predictions and  evaluation on validation and test sets'''
  
  assert fn in ('validate', 'test', 'predict'), f'`fn` must be ("validate", "test", "predict"), got {fn}'
  assert not (trainer is None and model is None and dm is None) and not (trainer is not None and model is not None and dm is not None), 'provide either `trainer` or both `model` and `dm`'
  assert (model is None) == (dm is None), 'provide both `model` and `dm` or neither'
  if model is None and dm is None:
    model = trainer.model
    dm = trainer.datamodule
  if trainer is None:
    trainer = pl.Trainer(accelerator='auto', devices=1, num_nodes=1)
  return getattr(trainer, fn)(model=model, dataloaders=dm)


# Create individual val, test, and predict functions
pl_val = partial(pl_val_test_predict, 'validate')
pl_test = partial(pl_val_test_predict, 'test')
pl_predict = partial(pl_val_test_predict, 'predict')


def create_mlp(num_layers, input_dim, num_classes, dropout):
  '''Creates an MLP'''
  
  layers = []
  for i in range(num_layers):
    if num_layers == 1:
     return nn.Linear(input_dim, num_classes)
    elif i == num_layers-1:
      layers.append(nn.Linear(input_dim//2, num_classes))
    else:
      if i == 0:
        layers.append(nn.Linear(input_dim, input_dim//2))
      else:
        layers.append(nn.Linear(input_dim//2, input_dim//2))
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      layers.append(nn.ReLU())
  return nn.Sequential(*layers)


def collate_fn(batch):
  new_batch = {k: [] for k in batch[0].keys()}
  for item in batch:
    for k in new_batch.keys():
      new_batch[k].append(item[k])
  return new_batch

  
@torch.no_grad
def extract_image_embeddings(model, dataset, processor, use_projection):
  '''Extracts image embeddings'''
  
  outputs = []
  for image in tqdm(KeyDataset(dataset, key='image')):
    inputs = processor(images=[image], return_tensors='pt')
    output = model(**{k: v.to(device) for k, v in inputs.items()})
    if use_projection:
      output = output.image_embeds.cpu()
    else:
      output = output.pooler_output.cpu()
    outputs.append(output)
  return torch.cat(outputs)

@torch.no_grad
def extract_image_text_embeddings(model, dataset, processor, use_projection):
  '''Extracts image and text embeddings'''
  
  image_outputs = []
  text_outputs = []

  dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
  for batch in tqdm(dataloader):
    inputs = processor(images=batch['image'], text=batch['caption'], padding=True, truncation=True, return_tensors='pt')

    output = model(**{k: v.to(device) for k, v in inputs.items()})
    if use_projection:
      image_output = output.image_embeds.cpu()
      text_output = output.text_embeds.cpu()
    else:
      image_output = output.vision_model_outputs.pooler_output.cpu()
      text_output = output.text_model_outputs.pooler_output.cpu()
      
    image_outputs.append(image_output)
    text_outputs.append(text_output)
  return torch.cat((torch.cat(image_outputs), torch.cat(text_outputs)), axis=1)


def extract_caption(filepath):
  with open(filepath) as f:
    caption = json.load(f)['caption']
  return caption

  
if __name__ == '__main__':

  lt.monkey_patch()

  os.environ['TOKENIZERS_PARALLELISM'] = "false"

  # command-line arguments
  parser = ArgumentParser(
    prog='classify_data',
    description='classify data'
  )
  parser.add_argument('--dataset', type=str, help='Which dataset to generate for')
  parser.add_argument('--classification-type', choices=['single', 'multi'], help='Type of classification to perform. Choose from ["single", "multi"]')
  parser.add_argument('--data-type', choices=['image', 'image+text'], help='Type of data to use. Choose from ["images", "images+text"]')
  parser.add_argument('--modes', choices=['train', 'eval', 'label'], nargs='+', type=str, help='Whether to train the  model, predict with it, or both.')
  parser.add_argument('--num-layers', type=int, choices=[1, 2, 3], help='Number of FC layers. Check code for how these are constructed.')
  parser.add_argument('--use-projection', action='store_true', help='Use the final projection layer of CLIP')
  parser.add_argument('--weight-classes', action='store_true', help='Use class weights')
  parser.add_argument('--load-embeds', type=str, help='Where to load tensors+labels from if loading labels. Look at this code to see how the file should be structured.')
  parser.add_argument('--save-embeds', type=str, help='Where to save tensors+labels if saving.')
  parser.add_argument('--checkpoint', help='Where to save or load checkpoint')
  parser.add_argument('--save-best', action='store_true', help='Save best checkpoint based on val loss')
  parser.add_argument('--save-dir', type=str, help='Where to save annotations')
  parser.add_argument('--caption-dir', type=str, help='Where to load text from when embedding images and text. Only used when data-type is "images+text". This is different from save-dir in the sense that it should only lead to the dataset, not the split. The script expects splits inside this directory.')
  parser.add_argument('--force-multi-eval-label', action='store_true', help='If model was trained with single label classification, this flag can force multilabel classification. Only one class will be predicted per input still, but the test labels will be multilabel.')
  parser.add_argument('--eval-label-split', default='test', help='Dataset split to evaluate and/or label')
  parser.add_argument('--wandb-project', type=str, help='Name of Weights & Biases project. `--wandb-project` and `wandb-name` must both be provided for wandb logging to occur.')
  parser.add_argument('--wandb-name', type=str, help='Name of Weights & Biases experiment. `--wandb-project` and `wandb-name` must both be provided for wandb logging to occur.')

  args = parser.parse_args()
    
  # load dataset
  class_labels, dataset = classify_with_labels_dataset_helper_registry[args.dataset]()
  class_labels = np.array(class_labels)

  # add captions if necessary
  if args.data_type == 'image+text':
    dataset = DatasetDict({
      split: _dataset.add_column('caption', [extract_caption(os.path.join(args.caption_dir, split, f'{image_id}.json')) for image_id in _dataset['image_id']])
      for split, _dataset
      in dataset.items()
    })

  if args.classification_type == 'single' and not args.force_multi_eval_label:
  
    dataset = DatasetDict({
      split: _dataset.select((np.sum(_dataset['labels'], axis=1) == 1).nonzero()[0])
      for split, _dataset
      in dataset.items()
    }).map(lambda item: {'labels': np.argmax(item['labels'])})
  
  print(dataset)

  # prepare model
  device = torch.device('cuda')
  processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
  if args.data_type == 'image+text':
    clip_cls = CLIPModel
  elif args.use_projection:
    clip_cls = CLIPVisionModelWithProjection
  else:
    clip_cls = CLIPVisionModel

  model = clip_cls.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)

  # prepare embeddings
  extract_embeddings = extract_image_embeddings if args.data_type == 'image' else extract_image_text_embeddings
  
  if args.load_embeds:
    embeds_labels = load_file(args.load_embeds)
  else:
    embeds_labels = {}
    train_embeds = extract_embeddings(model, dataset['train'], processor, args.use_projection)
    embeds_labels.update({'train_embeds': train_embeds, 'train_labels': torch.tensor(dataset['train']['labels'])})

    val_splits = [k for k in dataset.keys() if k.startswith('val')]
    for split in val_splits:
      val_embeds = extract_embeddings(model, dataset[split], processor, args.use_projection)
      embeds_labels.update({f'{split}_embeds': val_embeds, f'{split}_labels': torch.tensor(dataset[split]['labels'])})

    test_splits = [k for k in dataset.keys() if k.startswith('test')]
    for split in test_splits:
      test_embeds = extract_embeddings(model, dataset[split], processor, args.use_projection)
      embeds_labels.update({f'{split}_embeds': test_embeds, f'{split}_labels': torch.tensor(dataset[split]['labels'])})

  if args.save_embeds:
    save_file(embeds_labels, args.save_embeds)

  print('ALL DATA DONE')

  num_classes = len(class_labels)

  if args.use_projection:
    if args.data_type == 'image':
      input_dim = 512
    else:
      input_dim = 512+512
  else:
    if args.data_type == 'image':
      input_dim = 768
    else:
      input_dim = 768+512

  task = 'multiclass' if args.classification_type == 'single' else 'multilabel'

  assert not(args.weight_classes and args.classification_type == 'multi'), 'Cannot use class weighting in multilabel classification'
  if args.weight_classes and args.classification_type == 'single':
    _labels = embeds_labels['train_labels'].view(-1).numpy()
    weight = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(_labels), y=_labels)).float()
  else:
    weight = None

  embeds_labels['test_embeds'] = embeds_labels[f'{args.eval_label_split}_embeds']
  embeds_labels['test_labels'] = embeds_labels[f'{args.eval_label_split}_labels']
  embeds_labels['pred_embeds'] = embeds_labels[f'{args.eval_label_split}_embeds']
  embeds_labels['pred_labels'] = embeds_labels[f'{args.eval_label_split}_labels']
  if 'val_embeds' not in embeds_labels:
    embeds_labels['val_embeds'] = embeds_labels['val_classify_embeds']
    embeds_labels['val_labels'] = embeds_labels['val_classify_labels']
    
  dm_kwarg_names = [
    f'{split}_{item}'
    for split, item
    in product(['train', 'val', 'test', 'pred'], ['embeds', 'labels'])
  ]
  embeds_labels = {k: v for k, v in embeds_labels.items() if k in dm_kwarg_names}
  print(embeds_labels)

  # train / eval / label inputs
  if 'train' in args.modes:
    trainer_kwargs = {
      'max_epochs': 100,
      'callbacks': [LearningRateMonitor(logging_interval='step')]
    }
    if args.save_best:
      trainer_kwargs['callbacks'].append(ModelCheckpoint(
        dirpath=os.path.dirname(args.checkpoint),
        filename=f'{os.path.splitext(os.path.basename(args.checkpoint))[0]}'+'-epoch={epoch}-val_loss={val/loss}',
        monitor='val/loss',
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=False
      ))
      print(f'{os.path.splitext(os.path.basename(args.checkpoint))[0]}'+'-epoch={epoch}-val_loss={val/loss:.2f}')
      
    trainer = pl_train(
        lit_model_cls=LitModel,
        dm_init=LitTensorDataModule,
        lit_model_kwargs=dict(
          num_layers=args.num_layers,
          input_dim=input_dim,
          num_classes=num_classes,
          task=task,
          weight=weight
        ),
        dm_kwargs=embeds_labels,
        trainer_kwargs=trainer_kwargs,
        project=args.wandb_project,
        name=args.wandb_name
    )
    if args.checkpoint:
      trainer.save_checkpoint(args.checkpoint)
    

  if 'eval' in args.modes:
    if 'train' in args.modes:
      # if train also in args.modes, then we should have access to the trainer
      trainer.loggers = [] # weird thing about wandb run being finished
      print(pl_val(trainer=trainer))
      print(pl_test(trainer=trainer))
    else:
      # else, we'll have to load the model and create the data module manually

      if args.force_multi_eval_label:
        # use a model trained for single class classification to predict when data has multilabel labels
        model = LitModel.load_from_checkpoint(args.checkpoint, num_layers=args.num_layers, input_dim=input_dim, num_classes=num_classes, task=task, weight=weight)
        dm = LitTensorDataModule(**embeds_labels)
        trainer = pl.Trainer(accelerator='auto', devices=1, num_nodes=1)

        logits = trainer.predict(model=model, dataloaders=dm)
        logits = torch.cat([pred['logits'] for pred in logits])
        preds = torch.nn.functional.one_hot(torch.argmax(logits, dim=1), num_classes)

        print(preds)
        
        score_kwargs = dict(y_true=np.array(embeds_labels['test_labels']), y_pred=preds)
        average_kwargs= dict(average='macro')
        print({
            'accuracy': (preds == embeds_labels['test_labels']).float().mean().item(),
            'f1': f1_score(**score_kwargs, **average_kwargs),
            'precision': precision_score(**score_kwargs, **average_kwargs),
            'recall': recall_score(**score_kwargs, **average_kwargs),
            'ap': average_precision_score(y_true=np.array(embeds_labels['test_labels']), y_score=logits, **average_kwargs)
        })
      else:
        print(args)
        model = LitModel.load_from_checkpoint(args.checkpoint, num_layers=args.num_layers, input_dim=input_dim)
        dm = LitTensorDataModule(**embeds_labels)
        
        print(pl_val(model=model, dm=dm))
        print(pl_test(model=model, dm=dm))

  if 'label' in args.modes:
      
      # create folders
      os.makedirs(args.save_dir, exist_ok=True)

      # get predictions
      if 'train' in args.modes:
        # if train also in args.modes, then we should have access to the trainer
        trainer.loggers = [] # weird thing about wandb run being finished
        model = trainer.model
        preds = pl_predict(trainer=trainer)
      else:
        if 'eval' not in args.modes:
          # if eval was in args.modes, we'd already have the model and dm
          # task kwarg might not be necessary, does not affect predict_step
          model = LitModel.load_from_checkpoint(args.checkpoint, model=model_init(), task=task, num_classes=num_classes)
          dm = LitTensorDataModule(**embeds_labels)
        preds = pl_predict(model=model, dm=dm)
      
      preds = torch.cat([pred['logits'] for pred in preds])
      if args.classification_type == 'multi':
        preds = (preds > 0).int()
      else:
        preds = F.one_hot(torch.argmax(preds, dim=1))

      print(preds)
      dataset[args.eval_label_split] = dataset[args.eval_label_split].add_column('pred', preds.tolist())

      lr = model.lr
      weight_decay = model.weight_decay

      for item in tqdm(dataset[args.eval_label_split]):
        filename = os.path.basename(item['file_name'])
        # nonzero looks weird for single classification, 
        # which would more intuitively use softmax, 
        # but nonzero works here
        labels = class_labels[np.nonzero(item['pred'])].tolist() 

        with open(os.path.join(args.save_dir, f'{os.path.splitext(filename)[0]}.json'), 'w') as file:
          json.dump({'filename': filename, 'labels': labels, 'lr': lr, 'weight_decay': weight_decay}, file)
