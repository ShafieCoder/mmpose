import torch
import torch.nn as nn
from mmpose.models import LOSSES
import torch.nn.functional as f
from ..builder import LOSSES


def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel()>0
    
    norm_pred = f.normalize(pred, p = 1, dim =1)
    losses = torch.mul(-1,torch.sum(torch.mul(norm_pred , target), dim = 1))
    loss = torch.mean(losses)
    return loss

@LOSSES.register_module()
class MyLoss(nn.Module):
    """ Likelihood loss for heatmaps.
    
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """
    def __init__(self):
        super(MyLoss,self).__init__()
        #self.criterion = my_loss()
        #self.use_target_weight = use_target_weight

    def my_loss(self,pred, target):
        assert pred.size() == target.size() and target.numel()>0
    
        norm_pred = f.normalize(pred, p = 1, dim =1,eps=1e-12)
        print(norm_pred)
        print(target)
        losses = torch.mul(-1,torch.sum(torch.mul(norm_pred , target), dim = 1))
        loss = torch.mean(losses)
        return loss
       
        

    
    def forward(self, output, target):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            #if self.use_target_weight:
            #    loss += self.my_loss(heatmap_pred * target_weight[:, idx],
            #                           heatmap_gt * target_weight[:, idx])
            
            loss += self.my_loss(heatmap_pred, heatmap_gt)

        return loss / num_joints 
 
        