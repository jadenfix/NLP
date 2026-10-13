import math
import unittest

from ngram import Bigram, START_TOKEN, tokenize, train_regular, train_interpolation_trigram


class BigramInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = tokenize(["a b", "a c"])
        self.unigram = train_regular(self.corpus, 1)

    def smoothed(self, l1=.3, l2=.7):
        model = Bigram()
        model.set_unigram(self.unigram)
        model.train_bigram(self.corpus, smoothing=True, l1=l1, l2=l2)
        return model

    def test_training_and_seen_probabilities(self):
        model = self.smoothed()
        self.assertAlmostEqual(model.vocab_prob[(START_TOKEN, "a")], .8)
        self.assertAlmostEqual(model.vocab_prob[("a", "b")], .4)

    def test_unseen_transition_uses_unigram_component(self):
        model = self.smoothed()
        self.assertAlmostEqual(model.MLE(("a", "a"), True, .3, .7), .1)
        # P(a|START)=.8, P(a|a)=.1, P(STOP|a)=.1 => perplexity=5.
        self.assertAlmostEqual(model.perplexity(tokenize(["a a"])), 5)

    def test_observed_context_probability_mass(self):
        model = self.smoothed()
        for context in (START_TOKEN, "a", "b", "c"):
            mass = sum(model.MLE((context, word), True, .3, .7)
                       for word in self.unigram.vocab_prob)
            self.assertAlmostEqual(mass, 1)

    def test_unsmoothed_scores_preserved(self):
        model = train_regular(self.corpus, 2)
        self.assertAlmostEqual(model.perplexity(tokenize(["a b"])), 2 ** (1 / 3))
        interpolated = self.smoothed(0, 1)
        for sentence in ("a b", "a a", "unseen a"):
            self.assertAlmostEqual(model.perplexity(tokenize([sentence])),
                                   interpolated.perplexity(tokenize([sentence])))

    def test_unknown_context_and_word_do_not_raise(self):
        model = self.smoothed()
        self.assertTrue(math.isfinite(model.perplexity(tokenize(["unseen a"]))))
        self.assertAlmostEqual(model.MLE(("not-a-context", "a"), True, .3, .7), .1)

    def test_missing_unigram_rejected_clearly(self):
        with self.assertRaisesRegex(ValueError, "trained unigram"):
            Bigram().train_bigram(self.corpus, smoothing=True, l1=.3, l2=.7)

    def test_invalid_mixture_weights_rejected(self):
        for l1, l2 in ((-.1, 1.1), (.3, .3), (float("nan"), .7), (.3, float("inf"))):
            with self.subTest(l1=l1, l2=l2):
                with self.assertRaises(ValueError):
                    self.smoothed(l1, l2)

    def test_retraining_discards_old_probability_cache(self):
        model = self.smoothed()
        corpus = tokenize(["x y"])
        model.set_unigram(train_regular(corpus, 1))
        model.train_bigram(corpus, smoothing=True, l1=.3, l2=.7)
        self.assertNotIn(("a", "b"), model.vocab_prob)

    def test_trigram_shared_interpolation_path_preserved(self):
        bigram = train_regular(self.corpus, 2)
        model = train_interpolation_trigram(self.corpus, bigram, .3, .3, .4)
        self.assertAlmostEqual(model.perplexity(tokenize(["a b"])),
                               (.8 * .4 * .8) ** (-1 / 3))


if __name__ == "__main__":
    unittest.main()
