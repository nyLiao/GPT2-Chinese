# -*- coding: utf-8 -*-
# Repeatedly generating paragraph texts from inputs.
#
# by nyLiao, 2020

import os
import argparse
from tqdm import trange
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from tokenizations import tokenization_bert


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def gen_paragraph(model, contex, contex_past, length, temperature, topk, topp, device):
    prev, past = contex, contex_past
    generate = []
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=topk, top_p=topp)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate, past


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='trained model')
    parser.add_argument('--tokenizer_path', default='model/final_model/vocab.txt', type=str, required=False, help='tokenizer')
    parser.add_argument('--inputs', default='[CLS][MASK]', type=str, required=False, help='beginning of generated text')
    parser.add_argument('--length', default=100, type=int, required=False, help='generated length')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='temperature of generating freedom')
    parser.add_argument('--topk', default=8, type=int, required=False, help='top-k filtering')
    parser.add_argument('--topp', default=0, type=float, required=False, help='top-p filtering')
    # parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx
    length = args.length if args.length > 0 else n_ctx

    past = None             # pre-computed hidden-states
    while True:
        para_tokens = []    # generated tokens
        # inputs = args.inputs
        inputs = input("In: ")
        if inputs == "":
            inputs = "[CLS][MASK]"

        context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs))
        context_tensor = torch.LongTensor(context).view(1, -1).to(device)
        _, past = model(context_tensor[:, :-1], past)[:2]   # prepare past
        prev = context_tensor[:, -1].view(1, -1)            # minimize context to speed up
        para_tokens += context

        generate, past = gen_paragraph(model, prev, past,
            length=length, temperature=args.temperature,
            topk=args.topk, topp=args.topp, device=device)

        para_tokens += generate
        para_word = tokenizer.convert_ids_to_tokens(para_tokens)

        # for i, item in enumerate(para_word):
        #     if item == '[MASK]' or item == '[UNK]':
        #         para_word[i] = ''
        #     elif item == '[CLS]':
        #         para_word[i] = '\n\n'
        #     elif item == '[SEP]':
        #         para_word[i] = '\n'

        para_text = ''.join(para_word).strip()
        print(para_text)


if __name__ == '__main__':
    main()
