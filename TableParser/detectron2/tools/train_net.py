#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances

register_coco_instances("yearbooks-weak-tables-train2", {}, "datasets/yearbooks_processed_v2_coco/yearbooks_AUTOv1.json", "datasets/yearbooks_processed_v2")
register_coco_instances("yearbooks-weak-tables-train3", {}, "datasets/yearbooks_processed_v3_coco/yearbooks_AUTOv1.json", "datasets/yearbooks_processed_v3")
register_coco_instances("yearbooks-weak-tables-weaktrain3", {}, "datasets/yearbooks_v3_weak_train_coco/yearbooks_AUTOv1_objects.json", "datasets/yearbooks_v3_weak_train")

register_coco_instances("yearbooks-target-tables-train1", {}, "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/yearbooks_manual_train_coco_v1/yearbooks_GTJpprfinal_objects.json", "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/yearbooks_manual_train_v1")

register_coco_instances("yearbooks-target-tables-train-v2", {}, "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/manual_train_v2_with_L_multicells_coco/yearbooks_GTJ2postpr_objects.json", "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/manual_train_v2_with_L_multicells")
register_coco_instances("yearbooks-target-tables-val-v2", {}, "./detectron2/tools/datasets/manual_val_v2_with_L_multicells_coco/yearbooks_GTJ2postpr_objects.json", "./detectron2/tools/datasets/manual_val_v2_with_L_multicells")


register_coco_instances("Au-tables-v1-finetune", {}, "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Au_images_finetune_v1/weak_train_coco/Au_tables_automated_objects.json", "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Au_images_processed_finetune_v1")

register_coco_instances("Au-tables-v2-finetune_ada", {}, "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Au_images_finetune_v2/weak_train_coco/Au_tables_Ada_Ada-post_objects.json", "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Au_images_processed_finetune_v2")

register_coco_instances("Zh_images_test", {}, "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Zh_images_test/weak_train_coco/Zh_images_testAda-post_objects.json", "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Zh_images_processed_test")

register_coco_instances("Zh_images_test-auto", {}, "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Zh_images_test/weak_train_coco/Zh_images_testAda-post-auto_objects.json", "/mnt/ds3lab-scratch/raox/docparser_public/datasets/Zh_images_processed_test")

register_coco_instances("AuHu_images_test", {}, "/mnt/ds3lab-scratch/raox/docparser_public/datasets/AuHu_images_test/weak_train_coco/AuHu_images_testAda-post_objects.json", "/mnt/ds3lab-scratch/raox/docparser_public/datasets/AuHu_images_processed_test")
register_coco_instances("AuHu_images_test-auto", {}, "/mnt/ds3lab-scratch/raox/docparser_public/datasets/AuHu_images_test/weak_train_coco/AuHu_images_testAda-post-auto_objects.json", "/mnt/ds3lab-scratch/raox/docparser_public/datasets/AuHu_images_processed_test")

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
