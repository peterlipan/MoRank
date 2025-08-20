import torch
import torch.nn as nn
import torch.nn.functional as F


class RankConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(RankConsistencyLoss, self).__init__()
        self.weight = weight
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True) # needs to take log on both the inputs and targets
    
    def forward(self, stu_logits, tch_logits):
        # ensure the logits have the same size
        assert stu_logits.size() == tch_logits.size(), "Logits must have the same size"

        # for deepsurv, which returns a scalar risk, we directly estimate the soft ranking
        # for deephit/discrete, which returns a distribution on time bins, we estimate intra-bin ranking
        stu_rank = F.log_softmax(stu_logits, dim=0)
        tch_rank = F.log_softmax(tch_logits, dim=0)
        return self.weight * self.kl(stu_rank, tch_rank)
