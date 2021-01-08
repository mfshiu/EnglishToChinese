import sys
import os
import argparse

import torch

import data
import model
from torch.autograd import Variable


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 32


def parse_arguments():
    p = argparse.ArgumentParser(description='Translate Sentence')
    p.add_argument(
        '-model',
        type=str,
        default=model.DEFAULT_MODEL,
        help='pre-trained model [' + model.DEFAULT_MODEL + ']')
    p.add_argument(
        'sentence',
        type=str,
        default="",
        help='Sentence to be translated')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    assert torch.cuda.is_available()

    if not os.path.exists(args.model):
        print("model %s not exists." % (args.model))
        sys.exit(-1)

    seq2seq = torch.load(args.model)
    seq2seq = seq2seq.cuda()

    data.EnglishText.vocab = data.load_vocab("models/english.vocab")
    data.ChineseText.vocab = data.load_vocab("models/chinese.vocab")

    with open("data/test.txt") as fp:
        lines = fp.readlines()

    out_lines = []
    sentences = lines[:2]
    size = len(sentences)
    for index, sentence in enumerate(sentences):
        s = [data.EnglishText.vocab.stoi[w] for w in data.EnglishText.preprocess(sentence)]
        s.append(EOS_token)
        src = torch.LongTensor(s).view(-1, 1)

        t = [EOS_token for i in range(MAX_LENGTH)]
        trg = torch.LongTensor(t).view(-1, 1)

        seq2seq.eval()
        src, trg = src.cuda(), trg.cuda()
        with torch.no_grad():
            x = seq2seq(src, trg, teacher_forcing_ratio=0.0)

        x = torch.argmax(x.squeeze(1), dim=1).cpu()
        words = [data.ChineseText.vocab.itos[i] for i in x]

        translated = "".join(words)
        print("[%d/%d] Source: %s" % (index, size, sentence))
        print("[%d/%d] Result: %s", (index, size, translated))

        out_lines.append("%s\t%s\n" % (sentence, translated))

    with open("data/output.txt", "w") as fp:
        lines = fp.readlines() #
