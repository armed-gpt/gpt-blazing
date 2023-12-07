from typing import Sequence, Tuple
from sentencepiece import SentencePieceProcessor


class Baichuan2Tokenizer:

    def __init__(self, model_file: str) -> None:
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(model_file)

        self.bos_token_id = 1
        self.eos_token_id = 2

        self.user_token_id = 195
        self.assistant_token_id = 196

    def tokenize(self, text: str) -> Sequence[int]:
        return self.sp_model.tokenize(text)  # type: ignore

    def chat_tokenize(self, rounds: Sequence[Tuple[str, str]]):
        input_ids = [self.bos_token_id]

        if rounds[0][0] == 'system':
            text = rounds[0][1]
            input_ids.extend(self.tokenize(text))
            rounds = rounds[1:]

        for role, text in rounds:
            if role == 'user':
                input_ids.append(self.user_token_id)
            elif role == 'assistant':
                input_ids.append(self.assistant_token_id)
            else:
                raise NotImplementedError()
            input_ids.extend(self.tokenize(text))

        assert rounds[-1][0] == 'user'
        input_ids.append(self.assistant_token_id)

        return input_ids

    def decode(self, token: int) -> str:
        return self.sp_model.decode(token)  # type: ignore
