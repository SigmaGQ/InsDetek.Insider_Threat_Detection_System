import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        """
        Args:
            input: [Batchsize N, Num_class C, Seq_len L]
        """
        N, C, L = input.size(0), input.size(1), input.size(2)
        input = input.transpose(1,2)    # N,C,L => N,L,C
        input = input.contiguous().view(N*L, C)   # N,L,C => [N*L,C]
        target = target.view(N*L,1)   # N,L => [N*L,1]

        logpt = input.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.view(N,L)
        return loss