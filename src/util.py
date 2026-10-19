import torch
from transformers import PreTrainedTokenizerBase
from yachalk import chalk


def print_tokens(tokens: torch.LongTensor, tokenizer: PreTrainedTokenizerBase):
    # gen = tokenizer.batch_decode(generated_ids)
    # print(gen)

    num_tokens = len(tokens)

    for i in range(num_tokens):
        token = tokenizer.decode(tokens[i], skip_special_tokens=True)
        token = token.replace("\n", "\\n")
        if i % 2 == 0:
            print(chalk.bg_white.white_bright(token), end="")
        else:
            print(chalk.bg_gray.white(token), end="")
