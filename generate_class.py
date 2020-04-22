# -*- coding: utf-8 -*-
# Class for for generating paragraph texts from inputs.
#
# by nyLiao, 2019

# import os
# import argparse
# from tqdm import trange
# import time
from random import randint

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from tokenizations import tokenization_bert


MODEL7_PATH = './model/model_epoch7'
TOKEN7_PATH = './model/model_epoch7/vocab.txt'


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Params:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
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


class genModel(object):
    """API Class for generating texts."""

    def __init__(self, model_path=MODEL7_PATH, tokenizer_path=TOKEN7_PATH):
        """Init model with given path."""
        super(genModel, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.n_ctx = self.model.config.n_ctx
        self.past = None             # pre-computed hidden-states

    def clear(self):
        self.past = None

    def gen_ph(self, inputs="", length=100, topk=4, topp=1, temperature=1.5):
        """Generate a paragraph with input beginning and params.
            Params:
                inputs: text string as beginning of generated paragraph
                length: max length of generated paragraph
                topk, topp: params for top_k_top_p_filtering()
                temperature: generation temperature
            Retrun:
                para_text: generated text string
        """
        def gen_paragraph(model, contex, contex_past, device, length, topk, topp, temperature):
            prev, past = contex, contex_past
            generate = []
            with torch.no_grad():
                for i in range(length):
                    print(str(int(i / length * 100)) + "=i", end='')
                    try:
                        output = model(prev, past=past)
                        output, past = output[:2]
                        output = output[-1].squeeze(0) / temperature
                        filtered_logits = top_k_top_p_filtering(output, top_k=topk, top_p=topp)
                        next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
                        generate.append(next_token.item())
                    except Exception as e:
                        # Some past states cause predict index out of range
                        next_token = torch.tensor(randint(200, 7000))  # or 4638
                        past = None
                        # raise e
                    prev = next_token.view(1, 1)
            return generate, past

        def post_process(para_word):
            last_end = -1
            for i, item in enumerate(para_word):
                # Find where to end
                if item in ['[SEP]', '[CLS]', '。', '，', '；', '.']:
                    last_end = i
                # Replace words
                if item == '[MASK]' or item == '[UNK]':
                    para_word[i] = '[]'
                elif item == '[CLS]':
                    para_word[i] = '\n'
                elif item == '[SEP]':
                    para_word[i] = ''

            # End paragraph at last_end
            para_word[last_end] = '。'
            para_text = ''.join(para_word[:last_end+1]).strip()
            return para_text

        # print(length, temperature, topk, topp)
        # for i in range(300):
        #     print(str(int(i / 300 * 100)) + "=i", end='')
        #     time.sleep(0.01)
        # return "test"
        # -----
        # Data pre-processing
        length = length if length > 0 else self.n_ctx
        if inputs == "":        # as text beginning
            inputs = "[CLS][MASK]"
        if len(inputs) == 1:    # too short will cause pre-calculating bugs
            inputs = "[CLS][MASK]" + inputs
        para_tokens = []    # generated tokens

        # Pre-calculating to boost generation speed
        context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(inputs))
        context_tensor = torch.LongTensor(context).view(1, -1).to(self.device)
        _, self.past = self.model(context_tensor[:, :-1], self.past)[:2]   # prepare past
        prev = context_tensor[:, -1].view(1, -1)            # minimize context to speed up
        para_tokens += context

        # Generate and process to chars
        generate, self.past = gen_paragraph(self.model, prev, self.past,
            device=self.device, length=length,
            topk=topk, topp=topp, temperature=temperature)
        para_tokens += generate
        para_word = self.tokenizer.convert_ids_to_tokens(para_tokens)
        para_text = post_process(para_word)
        return para_text


if __name__ == '__main__':
    # main for command line running test
    model = genModel()
    inputs = ''
    while inputs != 'e':
        inputs = input("In: ")
        text = model.gen_ph(inputs=inputs)
        print(text)
