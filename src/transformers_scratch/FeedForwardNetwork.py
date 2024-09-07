import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    Attributes
    ----------
    self.W_1 : torch.Tensor
        重み1
    self.W_2 : torch.Tensor
        重み2

    method
    ----------
    forward(X: torch.Tensor) -> torch.Tensor
        ffnの計算を行う
    """

    def __init__(self, d_model: int) -> None:
        """
        説明
        ----------
        Feed Forward Networkの計算を行う

        Parameters
        ----------
        d_model : int
            次元数
        """

        super(FeedForwardNetwork, self).__init__()

        self.W_1 = nn.Linear(in_features=d_model, out_features=4 * d_model)
        self.W_2 = nn.Linear(in_features=4 * d_model, out_features=d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        ffnの計算を行う

        Parameters
        ----------
        X : torch.Tensor
            入力される値

        Returns
        ----------
        torch.Tensor
            ffn後の値
        """

        # 線形変換
        z = self.W_2(F.gelu(self.W_1(X)))

        return z
