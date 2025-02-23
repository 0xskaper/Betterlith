import torch
import torch.nn as nn
import torch.Tensor as tensor


class DeepFMLayer(nn.Module):
    def __init__(self, numFeatures: int, embeddingDimension: int, hiddenLayers: List[int]):
        super().__init__()
        self.numFeatures = numFeatures
        self.embeddingDimension = embeddingDimension

        self.firstOrder = nn.Embedding(numFeatures, 1)

        self.secondOrder = nn.Embedding(numFeatures, embeddingDimension)

        deepLayers = []
        inputDimension = numFeatures * embeddingDimension
        for hiddenDimension in hiddenLayers:
            deepLayers.extend([
                nn.Linear(inputDimension, hiddenDimension),
                nn.ReLU(),
                nn.BatchNorm1d(hiddenDimension),
                nn.DropOut()
            ])
            inputDimension = hiddenDimension
        deepLayers.append(nn.Linear(inputDimension, 1))
        self.deepLayers = nn.Sequential(*deepLayers)

    def forward(self, featureIds: tensor, featureValues: tensor) -> tensor:
        firstOrderWeights = self.firstOrder(featureIds).squeeze()
        firstOrder = torch.sum(firstOrderWeights * featureValues, dim=1)

        embeddings = self.secondOrder(featureIds)
        squareSum = torch.sum(embeddings, dim=1).pow(2)
        sumSquare = torch.sum(embeddings.pow(2), dim=1)
        secondOrder = 0.5 * torch.sum(squareSum - sumSquare, dim=1)

        deepInput = embeddings.reshape(-1,
                                       self.numFeatures * self.embeddingDimension)
        deep = self.deepLayers(deepInput).squeeze()

        return torch.sigmoid(firstOrder + secondOrder + deep)
