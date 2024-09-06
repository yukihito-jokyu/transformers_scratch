import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Attributes
    ----------
    self.d_k : torch.Tensor
        次元数

    method
    ----------
    forward(Xs: torch.Tensor) -> torch.Tensor
        attention後の値を返す
    """

    def __init__(self, d_k) -> None:
        """
        説明
        ----------
        attentionの計算を行う

        Parameters
        ----------
        d_k : int
            次元数
        """

        self.d_k = d_k

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        説明
        ----------
        attentionの計算を行う

        Parameters
        ----------
        Q: torch.Tensor
            クエリー
        K: torch.Tensor
            キー
        V: torch.Tensor
            バリュー

        Returns
        ----------
        torch.Tensor
            Attention後の値
        """

        qk = torch.matmul(Q, torch.transpose(K, 1, 2))
        scaler = np.sqrt(self.d_k)

        attention_weight = F.softmax(qk / scaler, dim=2)

        result = torch.matmul(attention_weight, V)

        return result
