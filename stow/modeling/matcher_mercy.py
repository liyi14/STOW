
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def batch_reid_loss(inputs: torch.Tensor, targets: torch.Tensor, gt_matched_track_indices: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
                QxC
        targets: A float tensor with the same shape as inputs. Stores the reid
                embedding prediction from last frame
                QxC
        gt_matched_track_indices: A int tensor indicating the index of queries in last frame matched 
                with objects in current frame, -1 for novel objects
    """
    similarity = torch.einsum("nc,mc->nm", inputs, targets)
    loss_over_all = -F.log_softmax(similarity, dim=1)
    zeros_for_novel = torch.zeros_like(loss_over_all[:,0])
    loss_over_matched = torch.stack([loss_over_all[:,i] if i>=0 else zeros_for_novel for i in gt_matched_track_indices], dim=1)
    # loss_over_matched = loss_over_all[:, matched_track_indices]
    return loss_over_matched

batch_reid_loss_jit = torch.jit.script(
    batch_reid_loss
)  # type: torch.jit.ScriptModule

def batch_inv_reid_loss(inputs: torch.Tensor, targets: torch.Tensor, gt_matched_track_indices: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
                QxC
        targets: A float tensor with the same shape as inputs. Stores the reid
                embedding prediction from last frame
                QxC
        gt_matched_track_indices: A int tensor indicating the index of queries in last frame matched 
                with objects in current frame, -1 for novel objects
    """
    similarity = torch.einsum("nc,mc->nm", inputs, targets)
    loss_over_all = -F.log_softmax(similarity, dim=0)
    zeros_for_novel = torch.zeros_like(loss_over_all[:,0])
    loss_over_matched = torch.stack([loss_over_all[:,i] if i>=0 else zeros_for_novel for i in gt_matched_track_indices], dim=1)
    # loss_over_matched = loss_over_all[:, matched_track_indices]
    return loss_over_matched

batch_inv_reid_loss_jit = torch.jit.script(
    batch_inv_reid_loss
)  # type: torch.jit.ScriptModule

class MercyMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, 
                 cost_class: float = 1,
                 cost_mask: float = 1, 
                 cost_dice: float = 1,
                 num_points: int = 0,
                 ignore_low: float = 0.5,
                 ignore_high: float = 0.5, 
                 max_num_fg: int = 100,
                 max_num_bg: int = 100):
        """Creates the matcher

        Params:
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.ignore_low = ignore_low
        self.ignore_high = ignore_high
        self.max_num_fg = max_num_fg
        self.max_num_bg = max_num_bg
        
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        positive_indices = []
        negative_indices = []
        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                IoU = 1-cost_dice
                
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            hungarian_matching = linear_sum_assignment(C)
            for query_index, gt_index in zip(*hungarian_matching):
                # make sure each gt has at least one matched query
                IoU[query_index] = 0
                IoU[query_index, gt_index] = 1.
            for thresh_pos in torch.arange(self.ignore_high, 1.02, 0.02):
                pos = (IoU >= thresh_pos).nonzero()
                if pos.shape[0] <= self.max_num_fg:
                    break
                
            neg = (IoU < self.ignore_low).nonzero()
            if neg.shape[0] > self.max_num_bg:
                neg = neg[torch.randperm(neg.shape[0])[:self.max_num_bg]]
            # print(IoU.shape, pos[:, 1])
            positive_indices.append((pos[:, 0], pos[:, 1]))
            negative_indices.append((neg[:, 0], neg[:, 1]))            

        positive_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in positive_indices
        ]
        negative_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in negative_indices
        ]
        return positive_indices, negative_indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
