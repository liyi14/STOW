import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import io
import torch
import cv2
import json
import os

import logging
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
import contextlib
from detectron2.data import MetadataCatalog
import pycocotools.mask as mask_util
from detectron2.structures import Boxes, BoxMode, PolygonMasks

logger = logging.getLogger(__name__)

__all__ = ['frameBinDatasetV2']

class frameBinDatasetV2(Dataset):
    def __init__(self, json_file, h5_file):
        self.dataset_dicts = self.load_ytvis_json(json_file, h5_file)

    def load_ytvis_json(self, json_file, image_root, dataset_name=None, extra_annotation_keys=None):
        from .ytvis_api.ytvos import YTVOS

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            ytvis_api = YTVOS(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        id_map = None
        if dataset_name is not None:
            meta = MetadataCatalog.get(dataset_name)
            cat_ids = sorted(ytvis_api.getCatIds())
            cats = ytvis_api.loadCats(cat_ids)
            # The categories in a custom json file may not be sorted.
            thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
            meta.thing_classes = thing_classes

            # In COCO, certain category ids are artificially removed,
            # and by convention they are always ignored.
            # We deal with COCO's id issue and translate
            # the category ids to contiguous ids in [0, 80).

            # It works by looking at the "categories" field in the json, therefore
            # if users' own json also have incontiguous ids, we'll
            # apply this mapping as well but print a warning.
            if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                if "coco" not in dataset_name:
                    logger.warning(
                        """
    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
    """
                    )
            id_map = {v: i for i, v in enumerate(cat_ids)}
            meta.thing_dataset_id_to_contiguous_id = id_map

        # sort indices for reproducible results
        vid_ids = sorted(ytvis_api.vids.keys())
        # vids is a list of dicts, each looks something like:
        # {'license': 1,
        #  'flickr_url': ' ',
        #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
        #  'height': 720,
        #  'width': 1280,
        #  'length': 36,
        #  'date_captured': '2019-04-11 00:55:41.903902',
        #  'id': 2232}
        vids = ytvis_api.loadVids(vid_ids)

        anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
        total_num_valid_anns = sum([len(x) for x in anns])
        total_num_anns = len(ytvis_api.anns)
        if total_num_valid_anns < total_num_anns:
            logger.warning(
                f"{json_file} contains {total_num_anns} annotations, but only "
                f"{total_num_valid_anns} of them match to images in the file."
            )

        vids_anns = list(zip(vids, anns))
        logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

        dataset_dicts = []

        ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

        num_instances_without_valid_segmentation = 0

        for (vid_dict, anno_dict_list) in vids_anns:
            record = {}
            record["image_root"] = image_root
            record["file_names"] = [vid_dict["file_names"][i] for i in range(vid_dict["length"])]
            record["height"] = vid_dict["height"]
            record["width"] = vid_dict["width"]
            record["length"] = vid_dict["length"]
            if "eval_idx" in vid_dict:
                record["eval_idx"] = vid_dict["eval_idx"]
            video_id = record["video_id"] = vid_dict["id"]

            video_objs = []
            for frame_idx in range(record["length"]):
                frame_objs = []
                for anno in anno_dict_list:
                    assert anno["video_id"] == video_id

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    _bboxes = anno.get("bboxes", None)
                    _segm = anno.get("segmentations", None)

                    if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                        continue

                    bbox = _bboxes[frame_idx]
                    segm = _segm[frame_idx]

                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYWH_ABS

                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    elif segm:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm
                    if id_map:
                        obj["category_id"] = id_map[obj["category_id"]]
                    # Mask2Former treat the background as the last class index
                    obj['category_id'] -= 1
                    frame_objs.append(obj)
                video_objs.append(frame_objs)
            record["annotations"] = video_objs
            dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. ".format(
                    num_instances_without_valid_segmentation
                )
                + "There might be issues in your dataset generation process. "
                "A valid polygon should be a list[float] with even length >= 6."
            )
        return dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        return self.dataset_dicts[idx]
    
    def load_dataset_into_memory(self):
        h5_file = None 
        h5_filename = None
        for record in self.dataset_dicts:
            if h5_filename != record["image_root"]:
                if h5_file is not None:
                    h5_file.close()
                h5_filename = record["image_root"]
                h5_file = h5py.File(record["image_root"], "r")
            images = []
            for fname in record["file_names"]:
                v, f, _ = fname.split("_")
                scene_id = int(v[1:])
                frame_id = int(f[1:])
                images.append(h5_file["data"][scene_id, frame_id])
            
            record['preload_image'] = []
            for image in images:
                record["preload_image"].append(image[:, :, :3]) # RGB format