# STOW: Discrete-Frame Segmentation and Tracking of Unseen Objects for Warehouse Picking Robots (CVPR 2023)

[Yi Li](https://yili.vision/), [Muru Zhang](https://nanami18.github.io/), [Markus Grotz](https://markusgrotz.github.io/), [Kaichun Mo](https://kaichun-mo.github.io/), [Dieter Fox](https://homes.cs.washington.edu/~fox/)


[[`arXiv`](https://arxiv.org/abs/2311.02337)] [[`Project`](https://sites.google.com/view/stow-corl23)] [[`BibTeX`](#CitingSTOW)]


![STOW Poster](DOCUMENTATION/stow_poster-1.jpg)

### Abstract
Segmentation and tracking of unseen object instances in discrete frames pose a significant challenge in dynamic industrial robotic contexts, such as distribution warehouses. Here, robots must handle object rearrangements, including shifting, removal, and partial occlusion by new items, and track these items after substantial temporal gaps. The task is further complicated when robots encounter objects beyond their training sets, thereby requiring the ability to segment and track previously unseen items. Considering that continuous observation is often inaccessible in such settings, our task involves working with a discrete set of frames separated by indefinite periods, during which substantial changes to the scene may occur. This task also translates to domestic robotic applications, such as table rearrangement. To address these demanding challenges, we introduce new synthetic and real-world datasets that replicate these industrial and household scenarios. Furthermore, we propose a novel paradigm for joint segmentation and tracking in discrete frames, alongside a transformer module that facilitates efficient inter-frame communication. Our approach significantly outperforms recent methods in our experiments.

## Installation

See [installation instructions](./DOCUMENTATION/INSTALL.md).

## Getting Started
Follow the instruction of how to use STOW.

See [Getting Started](GETTINGSTARTED.md)

## <a name="CitingSTOW"></a>Citing STOW

If you use STOW in your research or wish to refer to the baseline results, please use the following BibTeX entry.

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@misc{li2023stow,
      title={STOW: Discrete-Frame Segmentation and Tracking of Unseen Objects for Warehouse Picking Robots}, 
      author={Yi Li and Muru Zhang and Markus Grotz and Kaichun Mo and Dieter Fox},
      year={2023},
      eprint={2311.02337},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

```

## Credits and Acknowledgements

Release:
[Sebastian Gabriel](https://github.com/sgabriel92) 

Code is based on:
- [MaskFormer](https://github.com/facebookresearch/MaskFormer)
- [Mask2Former](https://bowenc0221.github.io/mask2former/)



