import torch.nn as nn
import torch, warnings
from monai.metrics import compute_hausdorff_distance

class SegementationMetric(nn.Module):
    def __init__(self, w1=0.6, w2=0.4):
        """
        w1 corresponds to weightage of hausdorff distance
        w2 corresponds to weightage of dice coefficent score
        """

        assert w1 + w2 == 1, 'sum of all weights must be equal to 1'
        super(SegementationMetric, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.bce_loss = nn.BCELoss()

    
    def hausdorff_distance(self, pred, target, max_dist=224):
        #pred shape: (N, C, H, W)
        #target shape: (N, C, H, W)
        pred = pred.detach().cpu().round()
        target = target.cpu()

        warnings.filterwarnings("ignore")

        #compute the hausdorff distance for each batch
        batch_distances = compute_hausdorff_distance(pred, target, distance_metric='euclidean', include_background=True)

        batch_distances[batch_distances == torch.inf] = max_dist
        batch_distances[batch_distances != batch_distances] = 0

        #get max distance for each batch and compute mean
        batch_distances = batch_distances.max(dim=1)
        dist = batch_distances[0].mean()
        dist = dist / max_dist
        return dist


    def dice_coeff(self, prediction, ground_truth, epsilon=1e-5):
        #input shape: N, C, H, W
        prediction = prediction.cpu().round()
        ground_truth = ground_truth.cpu()
        intersection = torch.sum(torch.abs(ground_truth * prediction), dim=(2, 3))
        denominator = ground_truth.sum(dim=(2, 3)) + prediction.sum(dim=(2, 3))
        dice_coeff = ((2 * intersection + epsilon) / (denominator + epsilon)).mean(dim=(0, 1))
        return dice_coeff.item()

      
    def metric_values(self, prediction, ground_truth):
        #input shape: N, C, H, W
        hausdorff_distance = self.hausdorff_distance(prediction, ground_truth)
        dice_coeff = self.dice_coeff(prediction, ground_truth)
        bce_loss = self.bce_loss(prediction, ground_truth)
        return hausdorff_distance, dice_coeff, bce_loss

    
    def forward(self, prediction, ground_truth):
        #input shape: N, C, H, W
        hausdorff_distance, dice_coeff, bce_loss = self.metric_values(prediction, ground_truth)
        hausdorff_distance = torch.tensor(hausdorff_distance)

        acc_score = (self.w1 * (1 - hausdorff_distance)) + (self.w2 * dice_coeff)
        loss = (1 + bce_loss) * (1 - acc_score).requires_grad_(True)
        return loss
    
    