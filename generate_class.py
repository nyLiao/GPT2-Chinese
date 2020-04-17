# -*- coding: utf-8 -*-
# Class for for generating paragraph texts from inputs.
#
# by nyLiao, 2019

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


class genModel(object):
    """API Class for generating texts."""

    def __init__(self, model_path='./model/model_epoch7', tokenizer_path='./model/model_epoch7/vocab.txt'):
        super(genModel, self).__init__()

        print(os.getcwd())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.n_ctx = self.model.config.n_ctx
        self.past = None             # pre-computed hidden-states

    def _clear(self):
        self.past = None             # pre-computed hidden-states

    def gen_ph(self, inputs="", length=100, topk=4, topp=1, temperature=1.5):
        """Generate a paragraph with input beginning and params."""
        def gen_paragraph(model, contex, contex_past, device, length, topk, topp, temperature):
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

        def post_process(para_word):
            for i, item in enumerate(para_word):
                if item == '[MASK]':
                    para_word[i] = '[]'
                elif item == '[CLS]':
                    para_word[i] = '\n'
                elif item == '[SEP]' or item == '[UNK]':
                    para_word[i] = ''

            para_text = ''.join(para_word).strip()
            return(para_text)

        length = length if length > 0 else self.n_ctx
        if inputs == "":
            inputs = "[CLS][MASK]"
        para_tokens = []    # generated tokens

        context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(inputs))
        context_tensor = torch.LongTensor(context).view(1, -1).to(self.device)
        _, self.past = self.model(context_tensor[:, :-1], self.past)[:2]   # prepare past
        prev = context_tensor[:, -1].view(1, -1)            # minimize context to speed up
        para_tokens += context

        generate, self.past = gen_paragraph(self.model, prev, self.past,
            device=self.device, length=length,
            topk=topk, topp=topp, temperature=temperature)
        para_tokens += generate
        para_word = self.tokenizer.convert_ids_to_tokens(para_tokens)
        para_text = post_process(para_word)
        return para_text
