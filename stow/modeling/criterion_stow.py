
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def reid_cosdist_loss(pred_reid_flatten: torch.Tensor, pred_match_flatten: torch.Tensor, 
              ref_reid_flatten: torch.Tensor, ref_match_flatten: torch.Tensor,
              alpha: float, sigma: float):
    
    reid_label = pred_match_flatten @ ref_match_flatten.T
    pos_map = (reid_label>0).detach()
    neg_neg_map = (pred_match_flatten.max(dim=1)[0].unsqueeze(1)==0).float() \
                @ (ref_match_flatten.max(dim=1)[0].unsqueeze(0)==0).float()
    neg_map = (reid_label==0) & (neg_neg_map==0)
    cosine_sim = F.cosine_similarity(pred_reid_flatten.unsqueeze(2), ref_reid_flatten.T.unsqueeze(0), dim=1)
    d = 0.5*(1-cosine_sim)
    pos_map = (((d-alpha)>0)*pos_map).float().detach()
    neg_map = (((sigma-d)>0)*neg_map).float().detach()
    num_pos = max(pos_map.sum().item(), 1)
    num_hard_neg = max(neg_map.sum().item(), 1)
    intra_loss = (1/num_pos*(pos_map*d*d))
    inter_loss = 1/num_hard_neg*torch.pow(torch.clamp(sigma-d, min=0), 2)*neg_map
    del neg_neg_map
    del reid_label
    return intra_loss.sum(), inter_loss.sum()
    
reid_cosdist_loss_jit = torch.jit.script(
    reid_cosdist_loss
)  # type: torch.jit.ScriptModule

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class SetCriterionV5(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, reid_loss_over,
                 reid_loss_type, use_temp, alpha, sigma):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.reid_loss_over = reid_loss_over
        self.reid_loss_type = reid_loss_type
        self.use_temp = use_temp
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/0.07)]))
        self.alpha = alpha 
        self.sigma = sigma

    def loss_labels(self, outputs, targets, positive_indices, negative_indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        target_classes = torch.full(
            src_logits.shape[:2], -100, dtype=torch.int64, device=src_logits.device
        )
        pos_idx = self._get_src_permutation_idx(positive_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, positive_indices)])
        target_classes[pos_idx] = target_classes_o
        neg_idx = self._get_src_permutation_idx(negative_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, negative_indices)])
        target_classes[neg_idx] = self.num_classes

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-100)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_reid(self, outputs, targets, pos_indices, neg_indices, matching_table, ref_outputs=None, ref_pos_indices=None, ref_neg_indices=None, ref_matching_table=None):
        loss_contra = 0
        loss_softmax = 0
        num_frames = len(outputs)
        assert num_frames == len(targets)
        batch_size, num_queries, _, _ = outputs[0]['pred_masks'].shape
        if ref_outputs == None:
            ref_outputs = outputs
            ref_pos_indices = pos_indices
            ref_neg_indices = neg_indices
            ref_matching_table = matching_table
        for b in range(batch_size):
            pred_reid = [outputs[f]['pred_reid'][b] for f in range(num_frames)]
            ref_reid = [ref_outputs[f]['pred_reid'][b] for f in range(num_frames)]
            pred_match = matching_table[b]
            ref_match = ref_matching_table[b]
            if self.reid_loss_over == "matched":
                indices = [pos_indices[f][b] for f in range(num_frames)]
                ref_indices = [ref_pos_indices[f][b] for f in range(num_frames)]
            elif self.reid_loss_over == "all":
                indices = []
                ref_indices = []
                for f in range(num_frames):
                    pos_query_idx, pos_gt_idx = pos_indices[f][b]
                    neg_query_idx, neg_gt_idx = neg_indices[f][b]
                    indices.append((torch.cat([pos_query_idx, neg_query_idx], dim=0), torch.cat([pos_gt_idx, neg_gt_idx], dim=0)))
                    ref_pos_query_idx, ref_pos_gt_idx = ref_pos_indices[f][b]
                    ref_neg_query_idx, ref_neg_gt_idx = ref_neg_indices[f][b]
                    ref_indices.append((torch.cat([ref_pos_query_idx, ref_neg_query_idx], dim=0), torch.cat([ref_pos_gt_idx, ref_neg_gt_idx], dim=0)))
            pred_reid = [pred_reid[frame_idx][query_idx] for frame_idx, (query_idx, gt_idx) in enumerate(indices)]
            pred_match = [pred_match[frame_idx][query_idx] for frame_idx, (query_idx, gt_idx) in enumerate(indices)]
            ref_reid = [ref_reid[frame_idx][query_idx] for frame_idx, (query_idx, gt_idx) in enumerate(ref_indices)]
            ref_match = [ref_match[frame_idx][query_idx] for frame_idx, (query_idx, gt_idx) in enumerate(ref_indices)]

            if '+' in self.reid_loss_type:
                loss_type = self.reid_loss_type.split('+')
            else:
                loss_type = self.reid_loss_type
            if loss_type == 'contrastive_cosdist' or 'contrastive_cosdist' in loss_type:
                pred_reid_flatten = torch.cat(pred_reid, dim=0)
                ref_reid_flatten = torch.cat([ref_reid[i] for i in range(num_frames)], dim=0)
                pred_match_flatten = torch.cat(pred_match, dim=0).float().detach()
                ref_match_flatten = torch.cat([ref_match[i] for i in range(num_frames)], dim=0).float().detach()
                intra_loss, inter_loss = reid_cosdist_loss(pred_reid_flatten, pred_match_flatten, ref_reid_flatten, ref_match_flatten, 0.02, 0.5)
                loss_contra += intra_loss.sum() + inter_loss.sum()
            if loss_type == 'triplet':
                raise NotImplementedError
            if loss_type == 'sigmoid':
                raise NotImplementedError
            if loss_type == 'softmax' or 'softmax' in loss_type:
                if self.use_temp:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = 1.
                for frame_idx in range(num_frames):
                    cur_reid = pred_reid[frame_idx]
                    other_reid = torch.cat([ref_reid[i] for i in range(num_frames) if i != frame_idx], dim=0)
                    cur_match = pred_match[frame_idx].float().detach()
                    other_match = torch.cat([ref_match[i] for i in range(num_frames) if i != frame_idx], dim=0).float().detach()
                    
                    if other_match.shape[0] != 0 and cur_match.shape[0] != 0:
                        similarity_map = torch.einsum("nc,mc->nm", other_reid, cur_reid)
                        match_map = other_match @ cur_match.T
                        match_value, match_label = torch.max(match_map, dim=1)
                        match_label[match_value==0] = -100
                        if match_label.max() < 0:
                            # print(cur_match, other_match)
                            continue
                        loss_softmax += F.cross_entropy(similarity_map*logit_scale, match_label)
                        
                        # if torch.isnan(loss_reid):
                        #     print(match_map.shape, match_label.shape, similarity_map.shape, match_label)

                        self_similarity_map = torch.einsum("nc,mc->nm", cur_reid, cur_reid)
                        match_map = cur_match @ cur_match.T
                        match_value, match_label = torch.max(match_map, dim=1)
                        match_label[match_value==0] = -100
                        loss_softmax += F.cross_entropy(self_similarity_map*logit_scale, match_label)
                        # if torch.isnan(loss_reid):
                        #     print(match_map.shape, match_label.shape, similarity_map.shape, match_label)
                    else:
                        print(other_match.shape, cur_match.shape)
                        continue
            
        return {'loss_contra': loss_contra/batch_size,
                'loss_softmax': loss_softmax/batch_size}
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)
    
    def check_lists_equal(list1, list2):
        n = min(len(list1), len(list2))
        for i in range(n):
            if list1[i] != list2[i]:
                return False
        return True

    def create_matching_table(self, outputs, targets, positive_indices):
        batch_size, num_queries, _, _ = outputs[0]['pred_masks'].shape
        num_frames = len(outputs)
        # create matching table
        matching_table = []
        for b in range(batch_size):
            t = [targets[f][b] for f in range(num_frames)]
            all_gt_uid = torch.unique(torch.cat([x['unique_indices'] for x in t])).tolist()
            max_num_gt = len(all_gt_uid)
            cur_matching_table = torch.zeros((num_frames, num_queries, max_num_gt), dtype=torch.bool, device=outputs[0]['pred_masks'].device)
            for f in range(num_frames):
                # # check consistency
                # for idx, uid in enumerate(t[f]['unique_indices']):
                #     if idx not in mapping_table:
                #         mapping_table[idx] = uid
                #     else:
                #         assert mapping_table[idx] == uid, "unique indices are not consistent"
                        
                query_indices, gt_idx = positive_indices[f][b]
                uid_cur_frame = t[f]['unique_indices'][gt_idx]
                uid_indices = torch.tensor([all_gt_uid.index(x) for x in uid_cur_frame], dtype=query_indices.dtype, device=query_indices.device)
                cur_matching_table[f, query_indices, uid_indices] = True
                
            matching_table.append(cur_matching_table)
        return matching_table
    
    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # clip logit_scale
        if self.logit_scale.data > torch.log(torch.tensor(100.0, device=self.logit_scale.device)):
            self.logit_scale.data = torch.log(torch.tensor(100.0, device=self.logit_scale.device))
        num_frames = len(targets)
        # Retrieve the matching between the outputs of the last layer and the targets
        positive_indices_last_layer = []
        negative_indices_last_layer = []
        losses = {}
        for frame_idx in range(num_frames):
            outputs_cur_frame = outputs[frame_idx]
            targets_cur_frame = targets[frame_idx]
            outputs_without_aux = {k: v for k, v in outputs_cur_frame.items() if k != "aux_outputs"}
            positive_indices, negative_indices = self.matcher(outputs_without_aux, targets_cur_frame)
            positive_indices_last_layer.append(positive_indices)
            negative_indices_last_layer.append(negative_indices)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_masks = sum(len(t["labels"]) for t in targets_cur_frame)
            num_masks = torch.as_tensor(
                [num_masks], dtype=torch.float, device=next(iter(outputs_without_aux.values())).device
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                if loss == 'reid':
                    continue
                if loss == 'masks':
                    l_dict = self.loss_masks(outputs_without_aux, targets_cur_frame, positive_indices, num_masks)
                if loss == 'labels':
                    l_dict = self.loss_labels(outputs_without_aux, targets_cur_frame, positive_indices, negative_indices)
                
                l_dict = {f"frame_{frame_idx}_" + k: v/num_frames for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'reid' in self.losses:
            matching_table_last_layer = self.create_matching_table(outputs, targets, positive_indices_last_layer)
            losses.update(self.loss_reid(outputs, targets, positive_indices_last_layer, negative_indices_last_layer, matching_table_last_layer))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs[0]:
            num_aux = len(outputs[0]["aux_outputs"])
            for i in range(num_aux):
                positive_indices_all = []
                negative_indices_all = []
                outputs_current_layer = [outputs[f]["aux_outputs"][i] for f in range(num_frames)]
                for frame_idx in range(num_frames):
                    aux_outputs = outputs_current_layer[frame_idx]
                    targets_cur_frame = targets[frame_idx]
                    positive_indices, negative_indices = self.matcher(aux_outputs, targets_cur_frame)
                    positive_indices_all.append(positive_indices)
                    negative_indices_all.append(negative_indices)
                    
                    # Compute the average number of target boxes accross all nodes, for normalization purposes
                    num_masks = sum(len(t["labels"]) for t in targets_cur_frame)
                    num_masks = torch.as_tensor(
                        [num_masks], dtype=torch.float, device=next(iter(outputs_without_aux.values())).device
                    )
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(num_masks)
                    num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
                    for loss in self.losses:
                        if loss == 'reid':
                            continue
                        if loss == 'masks':
                            l_dict = self.loss_masks(aux_outputs, targets_cur_frame, positive_indices, num_masks)
                        if loss == 'labels':
                            l_dict = self.loss_labels(aux_outputs, targets_cur_frame, positive_indices, negative_indices)
                        l_dict = {f"frame_{frame_idx}_" + k + f"_{i}": v/num_frames for k, v in l_dict.items()}
                        losses.update(l_dict)
                if 'reid' in self.losses:
                    if outputs_current_layer[0]['pred_reid'] == None or outputs_current_layer[1]['pred_reid'] == None:
                        continue
                    matching_table = self.create_matching_table(outputs_current_layer, targets, positive_indices_all)
                    l_dict = self.loss_reid(outputs_current_layer, targets, positive_indices_all, negative_indices_all, matching_table, 
                                            outputs, positive_indices_last_layer, negative_indices_last_layer, matching_table_last_layer)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                
        return losses, None

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
