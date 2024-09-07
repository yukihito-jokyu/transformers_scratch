import torch
import torch.nn as nn

from .Attention import MultiHeadAttention
from .Embedding import Embedding
from .FeedForwardNetwork import FeedForwardNetwork
from .Norm import Norm
from .PositionalEncoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Attributes
    ----------
    self.multi_head_attention : MultiHeadAttention
        Multi-Head Attentionの処理を行うレイヤ
    self.attention_norm : Norm
        Attention後の層正規化の処理を行うレイヤ
    self.feed_forward_network : FeedForwardNetwork
        Feed-Forward Networkの処理を行うレイヤ
    self.feed_forward_network_norm : Norm
        FFN後の層正規化の処理を行うレイヤ

    method
    ----------
    forward(X: torch.Tensor) -> torch.Tensor
        Encoderの処理を行う
    """

    def __init__(self, d_model: int, head: int) -> None:
        """
        説明
        ----------
        Encoderの一つのレイヤーの処理を記述する

        Parameters
        ----------
        d_model : int
            次元数
        head : int
            head数
        """

        super(EncoderLayer, self).__init__()

        # 各レイヤの定義
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, head=head)
        self.attention_dropout = nn.Dropout(p=0.1)
        self.attention_norm = Norm(d_model=d_model)
        self.feed_forward_network = FeedForwardNetwork(d_model=d_model)
        self.feed_forward_network_dropout = nn.Dropout(p=0.1)
        self.feed_forward_network_norm = Norm(d_model=d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        encoder内の計算を行う

        Parameters
        ----------
        X : torch.Tensor
            入力される値

        Returns
        ----------
        torch.Tensor
            encoder計算後の値
        """

        # レイヤの計算
        X_attention = self.multi_head_attention.forward(q=X, k=X, v=X)
        X_attention_dropout = self.attention_dropout(X_attention)
        X_attention_norm = self.attention_norm.forward(X_attention_dropout + X)
        X_ffn = self.feed_forward_network.forward(X_attention_norm)
        X_ffn_dropout = self.feed_forward_network_dropout(X_ffn)
        X_ffn_norm = self.feed_forward_network_norm.forward(
            X_ffn_dropout + X_attention_norm
        )

        return X_ffn_norm


class Encoder(nn.Module):
    """
    Attributes
    ----------
    self.embedding : Embedding
        Embeddingレイヤ
    self.positional : PositionalEncoding
        PositionalEncodingレイヤ
    self.encoder_laysers : nn.ModuleList
        EncoderLayerをN個保管しているリスト

    method
    ----------
    forward(src: torch.Tensor) -> torch.Tensor
        Encoderの処理を行う
    """

    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, head: int, N: int
    ) -> None:
        """
        説明
        ----------
        Encoderの処理を記述する

        Parameters
        ----------
        vocab_size : int
            語彙数
        max_len : int
            最大入力数
        d_model : int
            次元数
        head : int
            head数
        N : int
            encoderの処理の回数
        """

        super(Encoder, self).__init__()

        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.positional = PositionalEncoding(max_len=max_len, d_model=d_model)

        self.encoder_laysers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, head=head) for _ in range(N)]
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        encoderの全ての処理を行う

        Parameters
        ----------
        src : torch.Tensor
            単語idが入った値

        Returns
        ----------
        torch.Tensor
            encoder計算後の値
        """

        x = self.embedding.forward(src)
        x = self.positional.forward(x)

        # encoding
        for encoder in self.encoder_laysers:
            x = encoder.forward(x)

        return x
