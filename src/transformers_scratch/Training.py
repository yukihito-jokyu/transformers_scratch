from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .Bleu import BleuScore


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

    def __init__(
        self, net: nn.Module, optimizer, device: torch.device, bleu_score: BleuScore
    ) -> None:
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

        self.net = net.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.device = device
        self.bleu_score = bleu_score

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
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

        src = src.to(self.device)
        tgt = tgt.to(self.device)

        # 予測
        output = self.net.forward(src, tgt)

        # lossの計算に使うデータの取り出し
        output = output[:, :-1, :]
        tgt = tgt[:, 1:]

        # lossを計算するためにbatch_sizeとseq_sizeを結合する
        # output:[batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        outputs = output.contiguous().view(-1, output.size(-1))
        # tgt:[batch_size, seq_size] -> [, batch_size * seq_size]
        tgts = tgt.contiguous().view(-1)

        # lossの算出
        loss = self.criterion(outputs, tgts)

        _, output_ids = torch.max(output, dim=-1)

        # scoreの算出
        score = self.bleu_score.calculation_bleu_score(output=output_ids, tgt=tgt)

        # 学習
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output, loss, score

    def fit(self, train_loader: DataLoader) -> Tuple[List[float], List[float]]:
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
        train_score_list = []

        i = 0

        # 学習
        with tqdm(total=len(train_loader)) as pbar:
            for src, tgt in train_loader:
                _, train_loss, score = self.train_step(src, tgt)

                train_loss_list.append(train_loss.item())
                train_score_list.append(score)

                pbar.update(1)
                i += 1
                if i == 10:
                    break

        return train_loss_list, train_score_list
