"""
Step 1: data path [train|dev|test] expected
    - convert all to lower case 
"""

import sys

from rmai.utils import daba
import sentencepiece as spm

def spm_trainer(c_input, model_name, vocab_size, model_type, tokens):
    """ """
    if(model_type == "bpe"):
        spm.SentencePieceTrainer.train(
            input=c_input,
            model_prefix=model_name,
            vocab_size=vocab_size,
            user_defined_symbols=list(set(tokens)),
            hard_vocab_limit=False,
            model_type='bpe',
            split_digits=True)
    else:
        spm.SentencePieceTrainer.train(
            input=c_input,
            model_prefix=model_name,
            vocab_size=vocab_size,
            user_defined_symbols=list(set(tokens)),
            hard_vocab_limit=False,
            model_type=model_type,
            split_digits=True)

def daba_base_model_create(source, target, bam2fr=True, model_type="bpe", vocab_size=50000):
    util = daba.DabaUtils()
    content1 = [i.strip() for i in open(source).readlines()]
    content2 = [i.strip() for i in open(target).readlines()]

    print(f"Tokenizing file {source} using Daba")

    if(bam2fr):
        tokens = util.tokenize_line(content1)
    else:
        tokens = []

    spm_trainer(source, "source", vocab_size, model_type, tokens)
    spm_trainer(target, "target", vocab_size, model_type, []) 

if __name__ == '__main__':
    """ """
    args = sys.argv
    source = args[1] # source path
    target = args[2] # target path
    vocab = args[3] # vocab size
    mode_type = args[4]

    daba_base_model_create(
        source, target,
        bam2fr=None, model_type=mode_type,
        vocab_size=vocab)
