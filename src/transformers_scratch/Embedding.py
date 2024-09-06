import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Attributes
    ----------
    self.embedding_layer : nn.Embedding
        単語の辞書を保存するjsonファイルのpath

    method
    ----------
    forward(inputs: torch.Tensor) -> torch.Tensor
        単語ベクトルをバッチごとに返すメソッド
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0) -> None:
        """
        説明
        ----------
        embeddingの処理を行う

        Parameters
        ----------
        vocab_size : int
            語彙数
        d_model : int
            単語ベクトルの次元数
        padding_idx : int
            パディング用idx
        """

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        embeddingの処理を行う

        Parameters
        ----------
        inputs : torch.Tensor
            中身が単語idのTensor型
        """

        embedded = self.embedding_layer(inputs)

        return embedded
