import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .Token import Token


class BleuScore:
    def __init__(self, tgt_vocab_path, max_seq) -> None:
        self.token = Token(path=tgt_vocab_path, max_seq=max_seq)
        self.smoothie = SmoothingFunction().method4

    def calculation_bleu_score(self, output: torch.Tensor, tgt: torch.Tensor):
        # バッチ内の各文に対して BLEU スコアを計算する
        batch_bleu_scores = []

        # tgt と output は (バッチサイズ, シーケンス長) の形式なので、それぞれのバッチごとに処理
        for i in range(tgt.size(0)):  # バッチ内の各サンプルを個別に処理
            tgt_text = self.token.to_token(tgt[i].tolist())  # 参照翻訳
            out_text = self.token.to_token(output[i].tolist())  # 生成された翻訳

            # tgt_text をリストのリストとして渡す
            bleu_score = sentence_bleu(
                [tgt_text], out_text, smoothing_function=self.smoothie
            )
            batch_bleu_scores.append(bleu_score)

        # バッチ全体の平均 BLEU スコアを返す
        return sum(batch_bleu_scores) / len(batch_bleu_scores)
