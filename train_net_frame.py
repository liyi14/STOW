
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluators,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    MaskFormerInstanceDatasetMapper,
)

from stow import (
    StowSynMultiFrameMapper,
    StowSynMapper,
    add_maskformer2_frame_config,
    frameBinDatasetV2,
    build_combined_loader,
    # FrameInstanceSegEvaluator,
)

from stow import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)


from detectron2.evaluation import COCOEvaluator
import numpy as np
import random
# torch.backends.cudnn.deterministic=True
# torch.manual_seed(20)
# torch.cuda.manual_seed(20)
# np.random.seed(20)
# random.seed(20)
class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
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
            os.makedirs(output_folder, exist_ok=True)
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        # instance segmentation
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "ytvis":
            return YTVISEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.DATASETS.TRAIN[0].startswith('stow'):
            loader = []
            for ds in cfg.DATASETS.TRAIN:

                meta = MetadataCatalog.get(ds)
                h5_file, json_file = meta.image_root, meta.json_file
                # else:
                #     raise KeyError("Dataset not found")
                dataset = frameBinDatasetV2(json_file=json_file, h5_file=h5_file)
                if cfg.LOAD_DATASET_INTO_MEMORY:
                    dataset.load_dataset_into_memory()
                mapper = StowSynMapper(cfg, is_train=True)
                loader.append(build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, aspect_ratio_grouping=False))
            if len(loader) == 0:
                raise KeyError("Dataset or mapper not found")
            if len(loader) > 1:
                return build_combined_loader(cfg, loader, cfg.DATASETS.DATASET_RATIO)
        else:
            dataset = None
            mapper = None
            
        if dataset is None or mapper is None:
            raise KeyError("Dataset or mapper not found")

        return build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, aspect_ratio_grouping=False)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        # TODO: add support for multiple datasets
        dataset_name = cfg.DATASETS.TEST[0]
        # instance segmentation
        if evaluator_type == "coco":
            return super().build_test_loader(cfg, dataset_name)
        else:
            if 'real' in dataset_name:
                dataset = None 
                mapper = YTVISDatasetMapper(cfg, is_train=False)
            else:
                meta = MetadataCatalog.get(dataset_name)
                h5_file, json_file = meta.image_root, meta.json_file
                dataset = frameBinDatasetV2(json_file=json_file, h5_file=h5_file)
                mapper = StowSynMapper(cfg, is_train=False)
        
        if dataset:
            return build_detection_test_loader(cfg, dataset=dataset, mapper=mapper, dataset_name=None)
        else:
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if cfg.MODEL.FREEZE_WEIGHTS:
                    for k in cfg.MODEL.FREEZE_WEIGHTS:
                        if k in module_name:
                            value.requires_grad = False
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
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
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_frame_config(cfg)
    cfg.merge_from_file(args.config_file)

    output_folder = os.path.basename(args.config_file).split('.')[0]
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
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

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
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
