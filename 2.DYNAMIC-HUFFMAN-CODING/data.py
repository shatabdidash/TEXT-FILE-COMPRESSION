import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, file):
        self.dictionary = Dictionary()
        self.ids = self.tokenize(file)
        self.train = self.ids

    """Tokenizes a text file."""
    def tokenize(self, path):
        def split(word):
            return [char for char in word]
        assert os.path.exists(path)
        # start symbol
        self.dictionary.add_word('<s>')
        # Add words to the dictionary
        with open(path, encoding="ascii", errors="surrogateescape") as f:
            for line in f:
                words = split(line) + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize file content
        with open(path, encoding="ascii", errors="surrogateescape") as f:
            idss = []
            # start line
            ids = []
            # window size
            k = 10
            for i in range(k):
                ids.append(self.dictionary.word2idx['<s>'])
            idss.append(torch.tensor(ids).type(torch.int64))
            for line in f:
                # remove strip
                # words = split(line.strip()) + ['<eos>']
                words = split(line) + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids


class Context(object):
    def __init__(self, dictionary=Dictionary()):
        self.dictionary = dictionary

    def context_tokenize(self, context):
        idss = []
        ids = []
        for word in context:
            if word in self.dictionary.word2idx:
                ids.append(self.dictionary.word2idx[word])
        idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)
        return ids
