import torch
import torch.nn as nn


class Norm(nn.Module):
    """
    Attributes
    ----------
    self.r : torch.Tensor
        重み
    self.b : torch.Tensor
        重み
    self.eps : float
        0で割らないようにするための数値

    method
    ----------
    forward(X: torch.Tensor) -> torch.Tensor
        層正規化の処理を行う
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """
        説明
        ----------
        層正規化の処理を行う

        Parameters
        ----------
        d_model : int
            次元数
        """

        super(Norm, self).__init__()

        self.r = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        層正規化の処理を行う

        Parameters
        ----------
        X : torch.Tensor
            入力される値

        Returns
        ----------
        torch.Tensor
            層正規化後の値
        """

        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)

        norm_X = self.r * (X - mean) / (std + self.eps) + self.b

        return norm_X
