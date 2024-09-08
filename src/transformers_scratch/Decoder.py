import torch
import torch.nn as nn

from .Attention import MultiHeadAttention
from .Embedding import Embedding
from .FeedForwardNetwork import FeedForwardNetwork
from .Norm import Norm
from .PositionalEncoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Attributes
    ----------
    self.tgt_attention_layer : MultiHeadAttention
        tgtのMulti-Head Attentionの処理を行うレイヤ
    self.tgt_norm_layer : Norm
        tgtのAttention後の層正規化の処理を行うレイヤ
    self.src_tgt_attention_layer : MultiHeadAttention
        tgtとsrcのMulti-Head Attentionの処理を行うレイヤ
    self.src_tgt_norm_layer : Norm
        tgtとsrcのAttention後の層正規化の処理を行うレイヤ
    self.ffn_layer : FeedForwardNetwork
        FFNの処理を行うレイヤ
    self.ffn_norm_layer : Norm
        FFN後の層正規化の処理を行うレイヤ

    method
    ----------
    forward(self, src: torch.Tensor, tgt: torch.Tensor, src_tgt_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor
        Decoderの処理を行う
    """

    def __init__(self, d_model: int, head: int) -> None:
        """
        説明
        ----------
        Decoderの一つのレイヤーの処理を記述する

        Parameters
        ----------
        d_model : int
            次元数
        head : int
            head数
        """

        super(DecoderLayer, self).__init__()

        # 各レイヤの定義
        self.tgt_attention_layer = MultiHeadAttention(d_model=d_model, head=head)
        self.tgt_norm_layer = Norm(d_model=d_model)
        self.src_tgt_attention_layer = MultiHeadAttention(d_model=d_model, head=head)
        self.src_tgt_norm_layer = Norm(d_model=d_model)
        self.ffn_layer = FeedForwardNetwork(d_model=d_model)
        self.ffn_norm_layer = Norm(d_model=d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_tgt_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        説明
        ----------
        decoder内の計算を行う

        Parameters
        ----------
        src : torch.Tensor
            Encoder処理後のsrc値
        tgt : torch.Tensor
            入力されるtgt
        src_tgt_mask : torch.Tensor
            Crossed Multi-Head Attentionで使われるmask
        tgt_mask : torch.Tensor
            Masked Multi-Head Attentionで使われるmask

        Returns
        ----------
        torch.Tensor
            decoder計算後の値
        """

        # 各レイヤの計算
        tgt_attention_output = self.tgt_attention_layer.forward(
            q=tgt, k=tgt, v=tgt, mask=tgt_mask
        )
        tgt_norm_output = self.tgt_norm_layer.forward(tgt_attention_output + tgt)
        src_tgt_attention_output = self.src_tgt_attention_layer.forward(
            q=tgt_norm_output, k=src, v=src, mask=src_tgt_mask
        )
        src_tgt_norm_output = self.src_tgt_norm_layer.forward(
            src_tgt_attention_output + tgt_norm_output
        )
        ffn_output = self.ffn_layer.forward(src_tgt_norm_output)
        ffn_norm_output = self.ffn_norm_layer.forward(ffn_output)

        return ffn_norm_output


class Decoder(nn.Module):
    """
    Attributes
    ----------
    self.embedding : Embedding
        Embeddingレイヤ
    self.positional : PositionalEncoding
        PositionalEncodingレイヤ
    self.decoder_layers : nn.ModuleList
        DecoderLayerをN個保管しているリスト

    method
    ----------
    forward(src: torch.Tensor, tgt: torch.Tensor, src_tgt_mask: torch.Tensor | None = None, tgt_mask: torch.Tensor | None = None) -> torch.Tensor
        Decoderの処理を行う
    """

    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, head: int, N: int
    ) -> None:
        """
        説明
        ----------
        Decoderの処理を記述する

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

        super(Decoder, self).__init__()

        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.positional = PositionalEncoding(max_len=max_len, d_model=d_model)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, head=head) for _ in range(N)]
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_tgt_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        説明
        ----------
        decoder内の計算を行う

        Parameters
        ----------
        src : torch.Tensor
            Encoder処理後のsrc値
        tgt : torch.Tensor
            入力されるtgt
        src_tgt_mask : torch.Tensor
            Crossed Multi-Head Attentionで使われるmask
        tgt_mask : torch.Tensor
            Masked Multi-Head Attentionで使われるmask

        Returns
        ----------
        torch.Tensor
            decoder計算後の値
        """

        tgt = self.embedding.forward(tgt)
        tgt = self.positional.forward(tgt)

        # decoder
        for decoder in self.decoder_layers:
            tgt = decoder.forward(src, tgt, src_tgt_mask, tgt_mask)

        return tgt
