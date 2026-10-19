import torch
from transformers import PreTrainedTokenizerBase
from yachalk import chalk
from rich.console import Console


def print_tokens(tokens: torch.LongTensor, tokenizer: PreTrainedTokenizerBase):
    # gen = tokenizer.batch_decode(generated_ids)
    # print(gen)

    num_tokens = len(tokens)

    for i in range(num_tokens):
        token = tokenizer.decode(tokens[i])
        token = token.replace("\n", "\\n")
        if i % 2 == 0:
            print(chalk.bg_white.white_bright(token), end="")
        else:
            print(chalk.bg_gray.white(token), end="")
    print("")


def print_token_diff(old_tokens: torch.LongTensor, new_tokens: torch.LongTensor, tokenizer: PreTrainedTokenizerBase):
    string = ""

    console_width = Console().width - 4
    current_line_len = 0

    length = min(len(old_tokens), len(new_tokens))
    for i in range(length):
        old_token = tokenizer.decode(old_tokens[i])
        new_token = tokenizer.decode(new_tokens[i])

        if current_line_len + len(new_token) > console_width:
            string += "\n"
            current_line_len = 0

        current_line_len += len(new_token)

        old_token = old_token.replace("\n", "\\n")
        new_token = new_token.replace("\n", "\\n")

        if old_token == new_token:
            if i % 2 == 0:
                string += chalk.bg_white.white_bright(new_token)
            else:
                string += chalk.bg_gray.white(new_token)
        else:
            if i % 2 == 0:
                string += chalk.bg_red.white_bright(new_token)
            else:
                string += chalk.bg_red_bright.white(new_token)

    return string
