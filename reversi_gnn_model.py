import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# config.pyやconfig_loader.pyなどで読み込んだ設定値をimportする想定
from config import MODEL_HIDDEN_CHANNELS, MODEL_OUTPUT_DIM


class ReversiGNN(nn.Module):
    def __init__(self): 
        super(ReversiGNN, self).__init__()
        self.conv1 = GCNConv(1, MODEL_HIDDEN_CHANNELS)
        self.conv2 = GCNConv(MODEL_HIDDEN_CHANNELS, MODEL_HIDDEN_CHANNELS)
        self.fc = nn.Linear(
            MODEL_HIDDEN_CHANNELS, MODEL_OUTPUT_DIM
        )  # 各ノードの特徴 → 出力次元

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)  # [ノード数, 出力次元] の形状
        x = x.view(-1)  # 1次元テンソルに変換（例: [64]）
        return x
