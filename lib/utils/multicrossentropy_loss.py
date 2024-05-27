import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

__all__ = ['MultiCrossEntropyLoss']

class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """

    def __init__(self, num_classes, weight, alpha=0.17):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.mse = nn.MSELoss(reduction='none')
        self.classes = num_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        :param logits:  [batch_size, seq_len, logits]
        :param targets: [batch_size, seq_len]
        """

        logits = einops.rearrange(logits, 'batch_size seq_len logits -> batch_size logits seq_len')
        loss = { }

        # Frame level classification
        loss['loss_ce'] = self.ce(
            einops.rearrange(logits, "batch_size logits seq_len -> (batch_size seq_len) logits"),
            einops.rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")
        )

        # Neighbour frames should have similar values
        loss['loss_mse'] = torch.mean(torch.clamp(self.mse(
            F.log_softmax(logits[:, :, 1:], dim=1),
            F.log_softmax(logits.detach()[:, :, :-1], dim=1)
        ), min=0.0, max=160.0))

        loss['loss_total'] = loss['loss_ce'] + self.alpha * loss['loss_mse']
        return loss

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)
            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)
