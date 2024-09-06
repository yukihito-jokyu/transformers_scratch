import json


class Token:
    """
        Attributes
    ----------
    self.path : str
        単語の辞書を保存するjsonファイルのpath
    self.token2id : dict[str, int]
        単語とそれに対する単語IDが保存されている。
    self.id2token : dict[int, str]
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
            self.token2id: dict[str, str] = vocab_data.get("token2id", {})
            self.id2token: dict[str, str] = vocab_data.get("id2token", {})
        except FileNotFoundError:
            print(f"Warning: The file '{self.path}' was not found.")
            with open(path, "w") as file:
                json.dump({}, file)

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

        word_list = text.split(" ")

        # 単語辞書の作成
        for word in word_list:
            if word not in self.token2id:
                idx = str(len(self.token2id))
                self.token2id[word] = idx
                self.id2token[idx] = word
