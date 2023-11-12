import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss()

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _dot_similarity(self, x, y):
        return torch.matmul(x, y.t())

    def _cosine_similarity(self, x, y):
        return F.cosine_similarity(x, y, dim=2)

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # Filter out the scores from the positive samples
        positives = torch.diagonal(similarity_matrix, offset=self.batch_size, dim1=0, dim2=1)
        negatives = similarity_matrix[~torch.eye(2 * self.batch_size, dtype=bool)].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

