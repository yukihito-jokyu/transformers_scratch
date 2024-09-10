from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .Token import Token


class CostumDataset(Dataset):
    """
    Attributes
    ----------
    src_list : List[List[str]]
        入力のデータセットを文章ごとにlistにし、空白で分割した
    tgt_list : List[List[str]]
        出力のデータセットを文章ごとにlistにし、空白で分割した
    src_vocab : Token
        入力の辞書
    src_vocab : Token
        出力の辞書

    method
    ----------
    """

    def __init__(
        self,
        src_vocab_path: str,
        tgt_vocab_path: str,
        src_dataset_path: str,
        tgt_dataset_path: str,
        max_seq: int,
    ) -> None:
        """
        説明
        ----------
        データの管理を行う

        Parameters
        ----------
        src_vocab_path : str
            入力の語彙データが保存されているjsonファイルのpath
        tgt_vocab_path : str
            出力の語彙データが保存されているjsonファイルのpath
        src_dataset_path : str
            入力のデータセットが保存されているtxtファイルのpath
        tgt_dataset_path : str
            出力のデータセットが保存されているtxtファイルのpath
        max_seq : int
            最大トークン数
        """

        self.src_list: List[List[str]] = []
        self.tgt_list: List[List[str]] = []

        # データセットからデータを抽出
        with open(src_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line_list = line.split()
                self.src_list.append(line_list)
        with open(tgt_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line_list = line.split()
                self.tgt_list.append(line_list)

        # 入力出力の辞書の作成
        self.src_vocab = Token(src_vocab_path, max_seq=max_seq)
        self.tgt_vocab = Token(tgt_vocab_path, max_seq=max_seq)

    def __len__(self) -> int:
        return len(self.src_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        src_data = torch.Tensor(
            self.src_vocab.to_id(self.src_list[idx], bos=False, eos=False)
        ).long()
        tgt_data = torch.Tensor(self.tgt_vocab.to_id(self.tgt_list[idx])).long()
        return src_data, tgt_data
