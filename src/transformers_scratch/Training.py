from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Training(nn.Module):
    """
    Attributes
    ----------
    self.net: nn.Module
        学習用のネットワーク

    method
    ----------
    fit(train_loader: DataLoader) -> List[float]
        1epoch分の学習を行う
    """

    def __init__(self, net: nn.Module, optimizer) -> None:
        """
        説明
        ----------
        学習について管理するclass

        Parameters
        ----------
        net: nn.Module
            学習に使用するネットワーク
        """

        super(Training, self).__init__()

        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        説明
        ----------
        予測とlossの計算を行う

        Parameters
        ----------
        train_loader: DataLoader
            学習用のデータローダー

        Returns
        ----------
        Tuple[torch.Tensor, float]
            予測されたoutputとloss値
        """

        # 予測
        output = self.net.forward(src, tgt)

        # lossの計算に使うデータの取り出し
        output = output[:, :-1, :]
        tgt = tgt[:, 1:]

        # lossを計算するためにbatch_sizeとseq_sizeを結合する
        # output:[batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        output = output.contiguous().view(-1, output.size(-1))
        # tgt:[batch_size, seq_size] -> [, batch_size * seq_size]
        tgt = tgt.contiguous().view(-1)

        # lossの算出
        loss = self.criterion(output, tgt)

        # 学習
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output, loss

    def fit(self, train_loader: DataLoader) -> List[float]:
        """
        説明
        ----------
        DataLoaderを受け取り、1epoch分学習を行う。
        学習lossのリストを返す。

        Parameters
        ----------
        train_loader: DataLoader
            学習用のデータローダー

        Returns
        ----------
        List[float]
            学習のlossを保存しているリスト
        """

        train_loss_list = []

        # 学習
        for src, tgt in train_loader:
            _, train_loss = self.train_step(src, tgt)

            train_loss_list.append(train_loss.item())

        return train_loss_list
