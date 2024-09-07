import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Attributes
    ----------
    self.d_k : torch.Tensor
        次元数

    method
    ----------
    forward(Xs: torch.Tensor) -> torch.Tensor
        attention後の値を返す
    """

    def __init__(self, d_k) -> None:
        """
        説明
        ----------
        attentionの計算を行う

        Parameters
        ----------
        d_k : int
            次元数
        """

        self.d_k = d_k

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        説明
        ----------
        attentionの計算を行う

        Parameters
        ----------
        Q: torch.Tensor
            クエリー
        K: torch.Tensor
            キー
        V: torch.Tensor
            バリュー

        Returns
        ----------
        torch.Tensor
            Attention後の値
        """

        qk = torch.matmul(Q, torch.transpose(K, 1, 2))
        scaler = np.sqrt(self.d_k)

        attention_weight = F.softmax(qk / scaler, dim=2)

        result = torch.matmul(attention_weight, V)

        return result


class MultiHeadAttention(nn.Module):
    """
    Attributes
    ----------
    self.d_model : int
        次元数
    self.head : int
        head数
    self.d_k : int
        head分割時の次元数
    self.W_q : torch.Tensor
        重み
    self.W_k : torch.Tensor
        重み
    self.W_v : torch.Tensor
        重み
    self.scaled_dot_product_attention : ScaledDotProductAttention
        Attentionを計算する機構

    method
    ----------
    forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor
        クエリ―、キー、バリューからAttentionの計算を行う
    """

    def __init__(self, d_model: int, head: int) -> None:
        """
        説明
        ----------
        multi-head attentionの計算を行う

        Parameters
        ----------
        d_model : int
            次元数
        head : int
            head数
        """

        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.head = head

        # 次元数がhead数で割り切れなかったらエラーとする
        if self.d_model % self.head != 0:
            raise Exception("d_model / head で割り切れません！")

        self.d_k = self.d_model // self.head

        # 重みの作成
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Attention計算用
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k=self.d_k)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        説明
        ----------
        Multi-Head Attentionの計算を行う

        Parameters
        ----------
        q: torch.Tensor
            クエリー
        k: torch.Tensor
            キー
        v: torch.Tensor
            バリュー

        Returns
        ----------
        torch.Tensor
            Multi-Head Attention後の値
        """

        # バッチサイズとシーケンスサイズの取得
        batch_size, seq_len = q.size(0), q.size(1)

        # d_modelをheadとd_kに分ける [batch_size, seq_len, d_model] -> [batch_size, seq_len, head, d_k] (d_model = head * d_k)
        q = self.W_q(q).view(batch_size, seq_len, self.head, self.d_k)
        k = self.W_k(k).view(batch_size, seq_len, self.head, self.d_k)
        v = self.W_v(v).view(batch_size, seq_len, self.head, self.d_k)

        # seq_lenとheadを逆にする [batch_size, seq_len, head, d_k] -> [batch_size, head, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 各ヘッドに対して独立したAttentinoをするため、バッチとして定義する [batch_size, head, seq_len, d_k] -> [batch_size * head, seq_len, d_k]
        q = q.reshape(self.head * batch_size, seq_len, self.d_k)
        k = k.reshape(self.head * batch_size, seq_len, self.d_k)
        v = v.reshape(self.head * batch_size, seq_len, self.d_k)

        # Attention後の値を取得 [batch_size * head, seq_len, d_k]
        attention_output = self.scaled_dot_product_attention.forward(q, k, v)

        # head分だけ分割する Tuple[[batch_size, seq_len, d_k], ...]
        attention_chunk = torch.chunk(attention_output, self.head, dim=0)

        # 分割したheadを繋げる [batch_size, seq_len, d_model]
        output = torch.cat(attention_chunk, dim=2)

        return output
