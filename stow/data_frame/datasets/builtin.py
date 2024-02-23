
import os

from .stow import (
    register_stow_video_instances,
    register_stow_image_instances,
    register_stow_syn_video_instances,
    register_stow_videoperframe_instances,
    register_yuxiangds_instances,
    register_stow_tabletop_real_instances,
    _get_stow_instances_meta,
)

_PREDEFINED_SPLITS_STOW_BIN_REAL_VIDEO = {
    "stow_bin_real_test": ("stow2023/bin_real/images",
                           	  "stow2023/bin_real/video_instances.json"),
}

_PREDEFINED_SPLITS_STOW_BIN_SYN_VIDEO = {
    "stow_bin_syn_train": ("stow2023/bin_syn/train_shard_000000.h5",
                            	  "stow2023/bin_syn/train_shard_000000_coco.json"),
    "stow_bin_syn_val": ("stow2023/bin_syn/test_shard_000000.h5",
                            	"stow2023/bin_syn/test_shard_000000_coco.json"),
}

_PREDEFINED_SPLITS_STOW_TABLETOP_SYN = {
    "stow_tabletop_syn_train": ("stow2023/tabletop_syn/train_shard_000000.h5",
                            "stow2023/tabletop_syn/train_shard_000000_coco.json"),
    "stow_tabletop_syn_val": ("stow2023/tabletop_syn/test_shard_000000.h5",
                         "stow2023/tabletop_syn/test_shard_000000_coco.json"),
    "stow_tabletop_syn_mini_val": ("stow2023/tabletop_syn/mini_test.h5",
                              "stow2023/tabletop_syn/mini_test_coco.json"),
}

_PREDEFINED_SPLITS_STOW_TABLETOP_REAL = {
    "stow_tabletop_real_val": ("stow2023/tabletop_real/images",
                            "stow2023/tabletop_real/video_instances.json"),
    "stow_tabletop_real_val_oneframe": ("singleframe_camera/images",
                            "singleframe_camera/video_instances.json"),
}

def register_all_stow_bin_real_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_STOW_BIN_REAL_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_stow_video_instances(
            key,
            _get_stow_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_stow_bin_syn_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_STOW_BIN_SYN_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_stow_syn_video_instances(
            key,
            _get_stow_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_stow_tabletop_syn(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_STOW_TABLETOP_SYN.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_stow_syn_video_instances(
            key,
            _get_stow_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_stow_tabletop_real(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_STOW_TABLETOP_REAL.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_stow_tabletop_real_instances(
            key,
            _get_stow_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_stow_bin_real_video(_root)
    register_all_stow_bin_syn_video(_root)
    register_all_stow_tabletop_syn(_root)
    register_all_stow_tabletop_real(_root)
