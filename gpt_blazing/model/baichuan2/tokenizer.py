from typing import Sequence, Tuple

from sentencepiece import SentencePieceProcessor

from gpt_blazing.model.interface import Role


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

    def chat_tokenize(self, rounds: Sequence[Tuple[Role, str]]):
        input_ids = [self.bos_token_id]

        system = None
        if rounds[0][0] == Role.SYSTEM:
            system = rounds[0][1]
            input_ids.extend(self.tokenize(system))
            rounds = rounds[1:]

        num_system_tokens = len(input_ids)

        for role, text in rounds:
            if role == Role.USER:
                input_ids.append(self.user_token_id)
            elif role == Role.ASSISTANT:
                input_ids.append(self.assistant_token_id)
            else:
                raise NotImplementedError()
            input_ids.extend(self.tokenize(text))

        assert rounds[-1][0] == Role.USER
        input_ids.append(self.assistant_token_id)

        return input_ids, system, num_system_tokens

    def decode(self, tokens: Sequence[int]) -> str:
        return self.sp_model.decode(tokens)  # type: ignore
