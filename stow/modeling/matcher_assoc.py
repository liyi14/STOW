
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


# batch_reid_loss_jit = torch.jit.script(
#     batch_reid_loss
# )  # type: torch.jit.ScriptModule

class HungarianMatcherAssoc(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class_assoc: float = 1, 
                 cost_reid_assoc: float = 1, 
                 reid_seq_loss_type = 'sigmoid'):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class_assoc = cost_class_assoc
        self.cost_reid_assoc = cost_reid_assoc

        assert cost_class_assoc != 0 or cost_reid_assoc != 0, "all costs cant be 0"

        self.reid_loss_type = reid_seq_loss_type

    def batch_reid_loss(self, pred_reid_assoc: torch.Tensor, pred_reid_frame: torch.Tensor, matching_table_frame: torch.Tensor):
        """
        Args:
            pred_reid_assoc: num_queries x channel
            pred_reid_frame: num_frame x num_matched x channel
            matching_table_frame: num_frame x num_matched x num_gt
        Returns:
            per_query_loss: A float tensor of shape GxQ
        """
        num_queries = pred_reid_assoc.shape[0]
        num_frame = len(matching_table_frame)
        num_gt = matching_table_frame[0].shape[-1]
        if '_' in self.reid_loss_type:
            loss_type = self.reid_loss_type.split('_')[0]
            metric = self.reid_loss_type.split('_')[1]
        else:
            loss_type = self.reid_loss_type
        if loss_type == 'contrastive':
            if metric=='cosdist':
                per_query_loss_reid = torch.zeros(num_queries, num_gt, device=pred_reid_assoc.device)
                for f in range(num_frame):
                    similarity = F.cosine_similarity(pred_reid_assoc.unsqueeze(2), # num_queries x channel x 1
                                                    pred_reid_frame[f].T.unsqueeze(0), # 1 x channel x num_matched
                                                    dim=1)
                    distance = 0.5*(1-similarity)
                    per_query_loss_reid += torch.einsum("nm, mg->ng", distance, matching_table_frame[f].float())
                return per_query_loss_reid
            elif metric == 'cossim':
                raise NotImplementedError
        elif loss_type == 'triplet':
            raise NotImplementedError
        elif loss_type == 'sigmoid':
            raise NotImplementedError
        elif loss_type == 'softmax':
            per_query_loss_reid = torch.zeros(num_queries, num_gt, device=pred_reid_assoc.device)
            for f in range(num_frame):
                similarity_map = torch.einsum("nc,mc->nm", pred_reid_assoc.float(), pred_reid_frame[f])
                logits_map = torch.einsum("nm, mg->ng", similarity_map, matching_table_frame[f].float().detach())
                per_query_loss_reid += -logits_map.softmax(-1) # num_queries x num_gt
            return per_query_loss_reid
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def memory_efficient_forward(self, pred_reid_assoc, pred_class_assoc, pred_reid_frame, matching_table_frame, table_class):
        """More memory-friendly matching"""
        """
            pred_reid_assoc: batch_size, num_queries x channel
            pred_class_assoc: batch_size, num_queries x num_classes+1)
            pred_reid_frame: batch_size, num_frame x num_queries x channel
            matching_table_frame: batch_size, num_frame x num_queries x num_gt
            table_class: batch_size, num_gt
        """
        bs = len(pred_reid_assoc)
        num_queries = pred_reid_assoc[0].shape[0]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = pred_class_assoc[b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = torch.tensor(table_class[b], device=out_prob.device).detach()

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            with autocast(enabled=False):
                if self.cost_reid_assoc>0:
                    cost_reid = self.batch_reid_loss(pred_reid_assoc[b],
                                                     pred_reid_frame[b],
                                                     matching_table_frame[b])
                                                            
                else:
                    cost_reid = 0
            # Final cost matrix
            C = ( self.cost_class_assoc * cost_class
                + self.cost_reid_assoc * cost_reid
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, pred_reid_assoc, pred_class_assoc, pred_reid_frame, matching_table_frame, table_class):
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
        return self.memory_efficient_forward(pred_reid_assoc, pred_class_assoc, pred_reid_frame, matching_table_frame, table_class)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class_sequence: {}".format(self.cost_class),
            "cost_reid_sequence: {}".format(self.cost_reid),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
