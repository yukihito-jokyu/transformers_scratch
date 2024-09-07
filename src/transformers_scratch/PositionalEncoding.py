import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Attributes
    ----------
    self.PE_vector : torch.tensor
        PEの値を保存する変数

    method
    ----------
    forward(inputs: torch.Tensor) -> torch.Tensor
        単語ベクトルについてPEを足し合わせる。
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        """
        説明
        ----------
        PEの処理を行う

        Parameters
        ----------
        max_len : int
            最大単語入力数
        d_model : int
            単語ベクトルの次元数
        """

        self.max_len = max_len
        self.d_model = d_model

        self.PE_vector = self._init_PE_vector()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        PEの値の初期化を行う

        Parameters
        ----------
        X : torch.Tensor
            入力される値

        Returns
        ----------
        torch.Tensor
            PE値を足した後の値
        """

        seq_len = X.size(1)

        return X + self.PE_vector[:seq_len, :].unsqueeze(0)

    def _init_PE_vector(self) -> torch.Tensor:
        """
        説明
        ----------
        PEの値の初期化を行う

        Returns
        ----------
        torch.Tensor
            PEの値
        """

        # PEの値を作成する
        PE_list = []
        for pos in range(1, self.max_len + 1):
            pe_list = []
            for i in range(self.d_model):
                pe_cost = pos / 10000 ** ((2 * i) / self.d_model)
                if i % 2 == 0:
                    pe = np.sin(pe_cost)
                else:
                    pe = np.cos(pe_cost)
                pe_list.append(pe)
            PE_list.append(pe_list)

        return torch.Tensor(PE_list).float()
