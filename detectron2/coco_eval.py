import os, itertools, json
from functools import partial

from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate


class CustomIOUCOCOEvaluator(COCOEvaluator):
  '''COCO evaluator that supports custom IOUs
  Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py
  '''
  
  def __init__(self, *args, iou=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.iou = iou

  def _eval_predictions(self, predictions, img_ids=None):
      """
      Evaluate predictions. Fill self._results with the metrics of the tasks.
      """
      self._logger.info("Preparing results for COCO format ...")
      coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
      tasks = self._tasks or self._tasks_from_predictions(coco_results)

      # unmap the category ids for COCO
      if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
          dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
          all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
          num_classes = len(all_contiguous_ids)
          assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

          reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
          for result in coco_results:
              category_id = result["category_id"]
              assert category_id < num_classes, (
                  f"A prediction has class={category_id}, "
                  f"but the dataset only has {num_classes} classes and "
                  f"predicted class id should be in [0, {num_classes - 1}]."
              )
              result["category_id"] = reverse_id_mapping[category_id]

      if self._output_dir:
          file_path = os.path.join(self._output_dir, "coco_instances_results.json")
          self._logger.info("Saving results to {}".format(file_path))
          with PathManager.open(file_path, "w") as f:
              f.write(json.dumps(coco_results))
              f.flush()

      if not self._do_evaluation:
          self._logger.info("Annotations are not available for evaluation.")
          return

      self._logger.info(
          "Evaluating predictions with {} COCO API...".format(
              "unofficial" if self._use_fast_impl else "official"
          )
      )
      for task in sorted(tasks):
          assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
          coco_eval = (
              _evaluate_predictions_on_coco(
                  self._coco_api,
                  coco_results,
                  task,
                  kpt_oks_sigmas=self._kpt_oks_sigmas,
                  cocoeval_fn=partial(CustomIOUCOCOeval, iou=self.iou), # hard-code cocoeval_fn with custom fn
                  img_ids=img_ids,
                  max_dets_per_image=self._max_dets_per_image,
              )
              if len(coco_results) > 0
              else None  # cocoapi does not handle empty results very well
          )
          self.coco_eval = coco_eval

          res = self._derive_coco_results(
              coco_eval, task, class_names=self._metadata.get("thing_classes")
          )
          self._results[task] = res

  def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        if self.iou is not None:
          # not sure if new dim is -2 or -1 but I don't think it should matter
          precisions = np.expand_dims(coco_eval.iou_stats, -2)
        else:
          precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

import numpy as np
class CustomIOUCOCOeval(COCOeval):
  '''COCOeval but with the ability to specify a custom iou threshold in order to
  later on get per-class AP results with the custom iou

  If iou is not None, then the overall stats for that iou are stored in
  self.iou_stats

  Kinda hacky ngl
  '''
  def __init__(self, *args, iou=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou = iou
        self.iou_stats = None
        self.added_iou = False
        if self.iou not in self.params.iouThrs:
          print(f'Custom IOU {self.iou} not in IOU thresholds {self.params.iouThrs}, appending and sorting')
          self.params.iouThrs = np.sort(np.append(self.params.iouThrs, self.iou))
          self.added_iou = True

  def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
                if self.iou is not None and self.iou == iouThr:
                  self.iou_stats = s
                  # print('IOU', self.iou_stats.shape)
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            if self.added_iou:
              stats = np.append(stats, _summarize(1, iouThr=self.iou, maxDets=self.params.maxDets[2]))
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
