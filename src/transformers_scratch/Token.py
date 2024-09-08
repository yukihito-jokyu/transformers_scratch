import json
from typing import Dict, List


class Token:
    """
    Attributes
    ----------
    self.path : str
        単語の辞書を保存するjsonファイルのpath
    self.max_seq : int
        最大トークン数
    self.token2id : Dict[str, int]
        単語とそれに対する単語IDが保存されている。
    self.id2token : Dict[int, str]
        単語IDとそれに対する単語が保存されている。

    method
    ----------
    make_vocab(path: str)
        単語の辞書を作成する関数。
        トークンに分割されたtextファイルから作成される。
    save()
        単語と単語IDを保存する関数。
    to_id(list[str]) -> list[int]
        単語から単語IDに変換する関数。
    to_toekn(list[int]) -> list[str]
        単語IDから単語に変換する関数。
    """

    def __init__(self, path: str, max_seq: int) -> None:
        """
        説明
        ----------
        インスタンス時に単語の辞書が保存されているpath入れる。

        Parameters
        ----------
        path : str
            vocabを保存するディレクトリ
        max_seq : int
            最大トークン数
        """

        self.path = path
        self.max_seq = max_seq

        # 単語の辞書の読み込み
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                vocab_data = json.load(file)
            self.token2id: Dict[str, str] = vocab_data.get("token2id", {})
            self.id2token: Dict[str, str] = vocab_data.get("id2token", {})
        except FileNotFoundError:
            print(f"Warning: The file '{self.path}' was not found.")
            with open(path, "w", encoding="utf-8") as file:
                json.dump({}, file, ensure_ascii=False, indent=4)

    def make_vocab(self, path: str) -> None:
        """
        説明
        ----------
        単語辞書を作成するメソッド

        Parameters
        ----------
        path : str
            tokenに分割されたtextファイル
        """

        # ファイルの読み込み
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Warning: The file '{path}' was not found.")
            return

        # 辞書が空の時、特殊トークンを入れる
        special_tokens = ["PAD", "BOS", "EOS", "UNK"]
        if len(self.token2id) == 0:
            for token in special_tokens:
                if token not in self.token2id:
                    idx = str(len(self.token2id))
                    self.token2id[token] = idx
                    self.id2token[idx] = token

        word_list = text.split()

        # 単語辞書の作成
        for word in word_list:
            if word not in self.token2id:
                idx = str(len(self.token2id))
                self.token2id[word] = idx
                self.id2token[idx] = word

    def save(self) -> None:
        """
        説明
        ----------
        単語辞書をjsonに保存する方法
        """

        # 保存
        save_data = {"token2id": self.token2id, "id2token": self.id2token}
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(save_data, file, ensure_ascii=False, indent=4)

    def to_id(self, words: List[str]) -> List[int]:
        """
        説明
        ----------
        単語から単語IDに変換するメソッド。

        Parameters
        ----------
        words : List[str]
            単語のリスト

        Returns
        ----------
        List[int]
            単語IDのリスト
        """

        # 初めと終わりにBOS,EOSを追加
        words = ["BOS"] + words + ["EOS"]

        # パディングを追加する
        words += ["PAD"] * (self.max_seq - len(words))

        return [int(self.token2id.get(word, 3)) for word in words]

    def to_token(self, ids: List[str]) -> List[str]:
        """
        説明
        ----------
        単語IDから単語に変換するメソッド。

        Parameters
        ----------
        ids : List[str]
            単語IDのリスト

        Returns
        ----------
        List[str]
            単語のリスト
        """

        return [self.id2token.get(idx, "<UNK>") for idx in ids]
