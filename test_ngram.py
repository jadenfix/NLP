import math
import unittest
from ngram import *

class NGramTests(unittest.TestCase):
    def test_start_marker_is_recognized_by_value(self):
        marker = ''.join(['<STA', 'RT>'])
        self.assertFalse(unknown(1, marker, 3))
        self.assertFalse(unknown(1, ''.join(['<ST', 'OP>']), 3))

    def test_unseen_word_without_unk_training_occurrences(self):
        model = train_regular(tokenize(["known known"]), 1)
        self.assertTrue(math.isfinite(model.perplexity(tokenize(["new"]))))

    def test_probability_mass_and_seen_sentence_perplexity(self):
        corpus = tokenize(['one two', 'one three', 'one two', 'one four'])
        corpus = get_corpus_with_unk(corpus, get_vocab_freq(corpus), 2)
        unigram = train_regular(corpus, 1)
        self.assertAlmostEqual(sum(unigram.vocab_prob.values()), 1)
        for order in (1, 2, 3):
            score = train_regular(corpus, order).perplexity(corpus)
            self.assertTrue(math.isfinite(score))
            self.assertGreaterEqual(score, 1)

if __name__ == '__main__':
    unittest.main()
