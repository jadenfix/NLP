"""Small offline example; results are not benchmark measurements."""
import argparse
import json
from ngram import tokenize, get_vocab_freq, get_corpus_with_unk, train_regular

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', help='Optional UTF-8 training file, one sentence per line')
    parser.add_argument('--text', default='the agent reads the code', help='Sentence to score')
    args = parser.parse_args()
    if args.train:
        with open(args.train, encoding='utf-8') as source:
            lines = source.readlines()
    else:
        lines = ['the agent reads the code', 'the agent writes the code',
                 'the agent tests the code', 'the agent improves the system',
                 'the system reads the data', 'the system tests the data']
    if not lines or not any(line.strip() for line in lines):
        parser.error('training corpus must contain a sentence')
    corpus = tokenize(lines)
    corpus = get_corpus_with_unk(corpus, get_vocab_freq(corpus))
    print(json.dumps({'dataset': args.train or 'built-in toy corpus', 'text': args.text,
                     'perplexity': {str(order): train_regular(corpus, order).perplexity(tokenize([args.text]))
                                    for order in (1, 2, 3)}}, indent=2))

if __name__ == '__main__':
    main()
