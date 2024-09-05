import json


class Token:
    """
    method
    ----------
    save()
        単語と単語IDを保存する関数。
    to_id(list[str]) -> list[int]
        単語から単語IDに変換する関数。
    to_toekn(list[int]) -> list[str]
        単語IDから単語に変換する関数。

    Attributes
    ----------
    self.path : str
        単語の辞書を保存するjsonファイルのpath
    self.token2id : dict[str, int]
        単語とそれに対する単語IDが保存されている。
    self.id2token : dict[int, str]
        単語IDとそれに対する単語が保存されている。
    """

    def __init__(self, path: str) -> None:
        """
        説明
        ----------
        インスタンス時に単語の辞書が保存されているpath入れる。

        Parameters
        ----------
        path : str
            vocabを保存するディレクトリ
        """

        self.path = path

        # 単語の辞書の読み込み
        try:
            with open(self.path, "r") as file:
                vocab_data = json.load(file)
            self.token2id = vocab_data.get("token2id", {})
            self.id2token = vocab_data.get("id2token", {})
        except FileNotFoundError:
            print(f"Warning: The file '{self.path}' was not found.")
            self.token2id = {}
            self.id2token = {}

    def make_vocab(self) -> None:
        pass
