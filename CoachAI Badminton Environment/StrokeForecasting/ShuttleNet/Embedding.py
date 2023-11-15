import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0

class EmbeddingLayer(nn.Module):
    def __init__(self, area_dim, shot_dim, player_dim, shot_num, player_num):
        super().__init__()

        # dim is 2 * d
        self.areaEmbedding = nn.Linear(2, area_dim)
        # dim is shot * shot
        self.shotEmbedding = nn.Embedding(shot_num, shot_dim, padding_idx = PAD)
        # dim is playerCount * playerCount
        self.playerEmbedding = nn.Embedding(player_num, player_dim, padding_idx = PAD)

    def forward(self, area, shot, player):
        embeddedArea = F.relu(self.areaEmbedding(area))
        embeddedShot = self.shotEmbedding(shot)
        embeddedAreaPlayer = self.playerEmbedding(player)

        return embeddedArea, embeddedShot, embeddedAreaPlayer