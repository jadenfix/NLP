"""Notebook models exposed without downloading the original benchmark corpus."""
from collections import Counter
import math
from random import sample
UNK_TOKEN = "<UNK>"
STOP_TOKEN = "<STOP>"
START_TOKEN = "<START>"

#Adding the dataset
def load_data(filename):
  text = []
  with open(filename, "r") as f:
    text = [sent for sent in f]
  return text
#Tokenizing the data by splitting words on space (" ")
def tokenize(raw_corpus):
  return [[START_TOKEN] + sent.split() + [STOP_TOKEN] for sent in raw_corpus]
#Getting vocab
def get_vocab_freq(corpus):
  counter = Counter([token for sent in corpus for token in sent])
  return dict(counter)
#Creating <UNK> tokens for words with low frequency
def get_corpus_with_unk(corpus, vocab_freq, freq_constraint=3):
  return [[(token if not unknown(vocab_freq[token], token, freq_constraint) else UNK_TOKEN) for token in sent] for sent in corpus]
#Word checker for <UNK>
def unknown(freq, token, freq_constraint):
  return (token != START_TOKEN and token != STOP_TOKEN and freq < freq_constraint)

class Unigram:
    def __init__(self):
        self.vocab_freq = {}
        self.vocab_prob = {}
        self.smoothing = None
        self.l1 = 1

    @property
    def vocab_size(self):
        return len(self.vocab_freq.keys()) - 1

    @property
    def total_words_num(self):
        # excluding the <START> token
        return sum(self.vocab_freq.values()) - self.vocab_freq.get(
            START_TOKEN, 0
        )

    def perplexity(self, corpus):
        m = sum([len(sent) - 1 for sent in corpus])  # ignore the <START> token
        entropy = 0
        for sent in corpus:
            probs = [
                self.vocab_prob.get(self.convert_unk(word), 0)
                for word in sent
                if word != START_TOKEN
            ]
            entropy += self.log_joint_probs(probs)
        l = entropy / m
        return pow(2, -l)

    # for calculating log sum of joint probability and handle unseen words (in Unigram, just considering it as unknown)
    def log_joint_probs(self, probs):
        log_probs = []
        for prob in probs:
            log_probs.append(math.log(prob if prob else self.handle_zero_prob(), 2))
        return sum(log_probs)

    def handle_zero_prob(self):
        return pow(10, -10)

    def train_unigram(self, train_corpus, smoothing=None, l1=1):
        self.smoothing = smoothing
        self.l1 = l1

        self.vocab_freq = get_vocab_freq(train_corpus)
        for token in self.vocab_freq.keys():
            # ignore start token
            if token in [START_TOKEN]:
                continue
            # cache the probability for all unigram
            self.vocab_prob[token] = self.MLE(token, smoothing, l1)

    def MLE(self, word, smoothing=None, l1=1):
        # three different way for estimating Maximum Likelihood
        if smoothing:  
            return self.word_prob_with_interpolation(word, l1)
        else:
            return self.word_prob(word)

    # convert word to UNK token if not in the vocabulary
    def convert_unk(self, word):
        if word != START_TOKEN and word not in self.vocab_prob.keys():
            return UNK_TOKEN
        return word

    # without any smoothing
    def word_prob(self, word):
        return self.vocab_freq[word] / self.total_words_num

class Bigram:
  def __init__(self):
    # for storing pre-trained unigram
    self.unigram = None
    # frequency count for bigram only
    self.vocab_freq = {}
    # probability cache for bigram only
    self.vocab_prob = {}
    # hyperparameters
    self.smoothing = None
    self.l1 = 0
    self.l2 = 1
  # set the pre-trained unigram
  def set_unigram(self, unigram):
    self.unigram = unigram
  def perplexity(self, corpus):
    m = sum([len(sent) - 1 for sent in corpus])  # ignore the <START> token
    
    processed_corpus = [[self.unigram.convert_unk(word) for word in sent] for sent in corpus] #For <UNK>
    entropy = 0
    for sent in processed_corpus:
      # if not exist(unseen bigram), just give 0, and we will handle it in the log_joint_probs function
      probs = [self.vocab_prob.get(bigram, 0) for bigram in self.get_bigrams(sent)]
      entropy += self.log_joint_probs(probs)
    l = entropy / m
    return pow(2, -l)
  def log_joint_probs(self, probs):
    log_probs = []
    for prob in probs:
      log_probs.append(math.log(prob if prob else self.handle_zero_prob(), 2))
    return sum(log_probs)
  def handle_zero_prob(self):
    return pow(10, -10)
  def train_bigram(self, train_corpus, smoothing=None, l1=0, l2=1):
      self.smoothing = smoothing
      self.l1 = l1
      self.l2 = l2
      self.vocab_freq = get_vocab_freq([self.get_bigrams(sent) for sent in train_corpus])
      for bigram in self.vocab_freq.keys():
        self.vocab_prob[bigram] = self.MLE(bigram, smoothing, l1, l2)
  def MLE(self, bigram, smoothing=None, l1=0, l2=1):
    if smoothing : 
      return self.bigram_prob_with_interpolation(bigram, l1, l2)
    else:
      return self.bigram_prob(bigram)
  def get_bigrams(self, sent):
    # get all bigrams in given sentence
    return [(sent[i - 1], sent[i]) for i in range(1, len(sent))]
  def vocab_size(self):
    # excluding the <START> token
    return len(self.unigram.vocab_freq.keys()) - 1
  def bigram_prob(self, bigram):
    return (self.vocab_freq.get(bigram, 0) / self.unigram.vocab_freq[bigram[0]])
  def bigram_prob_interpolation(self, bigram, l1=0, l2=1):
    return l2 * self.bigram_prob(bigram) + l1 * self.unigram.word_prob(bigram[1])


class Trigram:
  def __init__(self):
    self.bigram = None
    self.vocab_freq = {}
    self.vocab_prob = {}
    self.smoothing = None
    self.l1 = 0
    self.l2 = 0
    self.l3 = 1

  def set_bigram(self, bigram):
    self.bigram = bigram

  def perplexity(self, corpus):
    m = sum([len(sent) - 1 for sent in corpus])  # ignore the <START> token
    processed_corpus = [[self.bigram.unigram.convert_unk(word) for word in sent] for sent in corpus]
    entropy = 0
    for sent in processed_corpus:
      entropy += self.log_joint_probs(self.MLE_log_with_sentence(sent))
    l = entropy / m
    return pow(2, -l)

  def log_joint_probs(self, probs):
    log_probs = []
    for prob in probs:
      log_probs.append(math.log(prob if prob else self.handle_zero_prob(), 2))
    return sum(log_probs)

  def handle_zero_prob(self):
    return pow(10, -10)

  def train_trigram(self, train_corpus, smoothing=False, l1=0, l2=0, l3=1):
    self.vocab_freq = get_vocab_freq([self.get_trigrams(sent) for sent in train_corpus])
    self.smoothing = smoothing
    self.l1 = l1
    self.l2 = l2
    self.l3 = l3

    for trigram in self.vocab_freq.keys():
      self.vocab_prob[trigram] = self.MLE(trigram)

  def MLE_log_with_sentence(self, sent):
    trigrams = self.get_trigrams(sent)
    first_trigram = trigrams[0]
    # need to calculate the P(w1 | <START>)
    w1_start_bigram = (first_trigram[0], first_trigram[1])
    bigram_prob = 1
    if self.smoothing: 
      bigram_prob = self.bigram_prob_interpolation(w1_start_bigram, self.l1, self.l2, self.l3)
    else:
      bigram_prob = self.bigram.bigram_prob(w1_start_bigram)
    probs = [bigram_prob]
    for trigram in trigrams:
      prob = self.vocab_prob.get(trigram, self.MLE(trigram))
      probs.append(prob)
    return probs

  def MLE(self, trigram):
    if self.smoothing: 
      return self.trigram_prob_interpolation(trigram, self.l1, self.l2, self.l3)
    else:
      return self.trigram_prob(trigram)

  def get_trigrams(self, sent):
    return [(sent[i - 2], sent[i - 1], sent[i]) for i in range(2, len(sent))]
  def vocab_size(self):
    return len(self.bigram.unigram.vocab_freq.keys()) - 1

  def trigram_prob(self, trigram):
    if (trigram[0], trigram[1]) not in self.bigram.vocab_freq.keys():
      return pow(10, -10)
    return (self.vocab_freq.get(trigram, 0)/self.bigram.vocab_freq[(trigram[0], trigram[1])])
  def bigram_prob_interpolation(self, bigram, l1=0, l2=0, l3=1):
    return self.bigram.bigram_prob_interpolation(bigram, l1, l2 + l3)

  def trigram_prob_interpolation(self, trigram, l1=0, l2=0, l3=1):
    return (l3 * self.trigram_prob(trigram) + l2 * self.bigram.vocab_prob.get((trigram[1], trigram[2]), 0) + l1 * self.bigram.unigram.vocab_prob.get(trigram[2], 0))

def train_regular(corpus, n_gram=1):
  unigram = Unigram()
  unigram.train_unigram(corpus)
  if n_gram == 1:
      return unigram
  bigram = Bigram()
  bigram.set_unigram(unigram)
  bigram.train_bigram(corpus)
  if n_gram == 2:
    return bigram
  trigram = Trigram()
  trigram.set_bigram(bigram)
  trigram.train_trigram(corpus)
  if n_gram == 3:
    return trigram
  return None
def train_interpolation_trigram(corpus, pretrained_bigram, l1=0.3, l2=0.3, l3=0.4):
  trigram_interpolation = Trigram()
  trigram_interpolation.set_bigram(pretrained_bigram)
  trigram_interpolation.train_trigram(corpus, smoothing=True, l1=l1, l2=l2, l3=l3)
  return trigram_interpolation
def get_pretrained_bigram(corpus):
  return train_regular(corpus, n_gram=2)
