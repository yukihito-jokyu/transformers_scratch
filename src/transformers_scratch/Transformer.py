import torch
import torch.nn as nn

from .Decoder import Decoder
from .Encoder import Encoder


class Transformer(nn.Module):
    """
    Attributes
    ----------
    self.encoder : Encoder
        Encoderレイヤ

    method
    ----------
    _padding_mask(src: torch.Tensor) -> torch.Tensor
        paddingに対してのmaskを作成する
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int,
        d_model: int,
        head: int,
        N: int,
    ) -> None:
        """
        説明
        ----------
        Transformerを定義したclass

        Parameters
        ----------
        src_vocab_size : int
            srcの語彙数
        tgt_vocab_size : int
            tgtの語彙数
        max_len : int
            最大入力数
        d_model : int
            次元数
        head : int
            head数
        N : int
            encoderの処理の回数
        """
        super(Transformer, self).__init__()

        # レイヤの定義
        self.encoder = Encoder(
            vocab_size=src_vocab_size, max_len=max_len, d_model=d_model, head=head, N=N
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size, max_len=max_len, d_model=d_model, head=head, N=N
        )

        # 最終層
        self.linear = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # mask処理
        src_mask = self._padding_mask(src)

        tgt_padding_mask = self._padding_mask(tgt)
        tgt_decoder_mask = self._decoder_mask(tgt)

        tgt_mask = torch.logical_or(tgt_padding_mask, tgt_decoder_mask)

        # output
        src_output = self.encoder.forward(src=src, mask=src_mask)
        tgt_output = self.decoder.forward(
            src=src_output, tgt=tgt, src_tgt_mask=src_mask, tgt_mask=tgt_mask
        )

        output = self.linear(tgt_output)

        return output

    def _padding_mask(self, X: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        paddingに対してmaskを作成する

        Parameters
        ----------
        X: torch.Tensor
            単語idが保管されたもの

        Returns
        ----------
        torch.Tensor
            mask
        """

        seq_len = X.size(1)

        # 値が0の場合はTrueそれ以外の場合はFalse [batch, seq_len]
        mask = X.eq(0)

        # [batch, seq_len] -> [batch, 1, seq_len]
        mask = mask.unsqueeze(1)

        # seq_lenだけ複製する [batch, 1, seq_len] -> [batch, seq_len, seq_len]
        mask = mask.repeat(1, seq_len, 1)

        return mask

    def _decoder_mask(self, X: torch.Tensor) -> torch.Tensor:
        """
        説明
        ----------
        decoderに使う要素に対してmaskを作成する

        Parameters
        ----------
        X: torch.Tensor
            単語idが保管されたもの

        Returns
        ----------
        torch.Tensor
            mask
        """

        batch_size = X.size(0)
        seq_len = X.size(1)

        # mask用の全てが1の変数を用意する
        mask = torch.ones(batch_size, seq_len, seq_len)

        # 対角線よりも上の要素を0にする
        mask = torch.tril(mask)

        # 0の要素はTrue,それ以外はFalseにする
        mask = mask.eq(0)

        return mask
