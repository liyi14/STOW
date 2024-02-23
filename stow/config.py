# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_maskformer2_frame_config(cfg):
    # Datset
    cfg.DATASETS.DATASET_RATIO = []

    # Training
    cfg.MODEL.FREEZE_WEIGHTS = []
    
    # Network
    cfg.MODEL.MASK_FORMER.MEMORY_FORWARD_STRATEGY = "top-k_lastest_frame:-1"
    cfg.MODEL.MASK_FORMER.TRACK_FEAT_TYPE = 'memory' # memory, zero, learned
    cfg.MODEL.MASK_FORMER.TRACK_POS_TYPE = 'zero' # memory, zero, learned
    cfg.MODEL.MASK_FORMER.REID_DIM = 256
    cfg.MODEL.MASK_FORMER.SELF_INIT_FRAME_ATTN = False
    
    # cfg.MODEL.MASK_FORMER.REID_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REID_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.INV_REID_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.REID_CONTRA_WEIGHT = -1.0
    cfg.MODEL.MASK_FORMER.REID_SOFTMAX_WEIGHT = -1.0
    cfg.MODEL.MASK_FORMER.REID_COEFF = 0.1
    cfg.MODEL.MASK_FORMER.INV_REID_COEFF = 0.0
    cfg.MODEL.MASK_FORMER.REID_LOSS_OVER = 'all' # macthed, all
    cfg.MODEL.MASK_FORMER.MASK_USE_DIFFERENT_FEATURE = False
    cfg.MODEL.MASK_FORMER.IGNORE_FIRST_OUTPUT = False
    cfg.MODEL.MASK_FORMER.IGNORE_FIRST_REID = False
    cfg.MODEL.MASK_FORMER.USE_LAYER_ATTNMASK = True
    cfg.MODEL.MASK_FORMER.REID_FRAME_LOSS_TYPE = 'sigmoid' # softmax, sigmoid, contrastive_cosdist, contrastive_cossim
    cfg.MODEL.MASK_FORMER.REID_SOFTMAX_TEMPERATURE = 1.0
    cfg.MODEL.MASK_FORMER.FRAME_SELFMASK = False
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_ALPHA = 0.02
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_SIGMA = 0.5
    cfg.MODEL.MASK_FORMER.FRAME_DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.ASSOC_DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.FEAT_DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.IGNORE_IOU_RANGE = [0.5, 0.5] # < first is negative, > second is positive, hungarian matched ones have value 1.0
    cfg.MODEL.MASK_FORMER.MAX_NUM_BG = 100
    cfg.MODEL.MASK_FORMER.MAX_NUM_FG = 100
    
    cfg.MODEL.REID = CN()
    cfg.MODEL.REID.ENCODER_TYPE = 'linear' # linear, mlp, 2frame-encoder
    cfg.MODEL.REID.REID_NORMALIZE = False # whether to normalize the reid embedding
    cfg.MODEL.REID.ASSOCIATOR = 'hungarian' # hungarian, decoder
    cfg.MODEL.REID.ENCODER_POS_TYPE = 'zero' # zero, fixed, learned, external
    cfg.MODEL.REID.INTERACTION_OVER = 'all' # matched, all
    cfg.MODEL.REID.ENCODER_DIM_FEEDFORWARD = 2048
    cfg.MODEL.REID.ENCODER_NUM_LAYERS = 1
    cfg.MODEL.REID.ENCODER_LAYER_STRUCTURE = ['full'] # self, cross, frame, ffn
    cfg.MODEL.REID.ENCODER_ACTIVATION = 'relu'
    cfg.MODEL.REID.ENCODER_NORMALIZE_BEFORE = False
    cfg.MODEL.REID.ENCODER_ONE_LAYER_AFTER_ATTENTION = False
    cfg.MODEL.REID.ENCODER_LAST_LAYER_BIAS = True
    cfg.MODEL.REID.ENCODER_UPDATE_CONFIDENCE = False
    cfg.MODEL.REID.ENCODER_PREDICT_MASK = False
    cfg.MODEL.REID.ENCODER_NORM_BEFORE_HEAD = False
    cfg.MODEL.REID.TEST_MATCH_THRESHOLD = 0.2
    
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Inference
    cfg.MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD = 0.6
    
    # Multi-GPU Training
    cfg.SOLVER.NUM_GPUS = 1
