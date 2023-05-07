import train
import os
import heapq
import collections
import operator
import sys
import time
from data import Corpus, Context, Dictionary
import torch
from model import RNNModel
import argparse
import torch.optim as optim
import torch.nn as nn


def read_characters(file):
    chars = []
    vocab = set()
    count = 0
    with open(file, encoding='ascii', errors="surrogateescape") as f:
        while True:
            c = f.read(1)
            chars.append(c)
            vocab.add(c)
            count += 1
            if not c:
                break
    return chars, vocab


def check():
    chars, vocab = read_characters('dickens.txt')
    print(chars[:1000])
    counts = {i: chars.count(i) for i in set(chars)}
    print(counts)
    from math import log2
    # entropy
    def shannon(boe):
        total = sum(boe.values())
        return sum(freq / total * log2(total / freq) for freq in boe.values())
    print(shannon(counts))


class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return other.freq > self.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.vocab = {}

    @staticmethod
    def make_frequency_dict(text):
        # fixed Huffman tree (not adaptive)
        # we will use this to encode unknowns
        counted = dict(collections.Counter(text))
        sort = collections.OrderedDict(
            sorted(
                counted.items(),
                key=operator.itemgetter(1),
                reverse=False))
        return sort

    @staticmethod
    def make_context_frequency_dict(context, model, context_map, device, threshold):
        model.eval()
        context_data = train.batchify(context_map.context_tokenize(context), 1, device)
        data, targets = train.get_batch(280, context_data, 0)
        with torch.no_grad():
            hidden = model.init_hidden(bsz=1)
            output, hidden = model(data, hidden)
            # model returns log softmax
            preds = output[-1].squeeze().exp().cpu().tolist()
            hidden = train.repackage_hidden(hidden)
        assert len(context_map.dictionary.idx2word) == len(preds)
        probs = {key: prob for key, prob in zip(context_map.dictionary.idx2word, preds)}
        if int(threshold):
            # threshold of number of nodes (leafs)
            p_sorted = sorted(probs, key=lambda x: probs[x], reverse=True)
            filtered = {k: probs[k] for k in p_sorted[:int(threshold)]}
            probs = filtered
            pass
        else:
            # threshold of probabilities
            probs = {key: prob for key, prob in zip(context_map.dictionary.idx2word, preds)
                     if prob > threshold}
            pass
        sort = collections.OrderedDict(
            sorted(
                probs.items(),
                key=operator.itemgetter(1),
                reverse=False))
        return sort

    # make a heap queue from node
    def make_heap_node(self, freq_dict):
        for key in freq_dict:
            anode = HeapNode(key, freq_dict[key])
            self.heap.append(anode)

    # build tree
    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merge = HeapNode(None, node1.freq + node2.freq)
            merge.left = node1
            merge.right = node2
            heapq.heappush(self.heap, merge)

    # actual coding happens here
    def encode_helper(self, root, current_code):
        if root is None:
            return
        if root.char is not None:
            self.codes[root.char] = current_code
            return
        self.encode_helper(root.left, current_code + "0")
        self.encode_helper(root.right, current_code + "1")

    def encode(self):
        root = heapq.heappop(self.heap)
        current_code = ''
        self.encode_helper(root, current_code)

    def get_encoded_word(self, word):
        encoded_text = ''
        encoded_text += self.codes[word]
        return encoded_text


class HuffmanLSTMCompressor:
    def __init__(self, args):
        self.args = args
        self.fname = args.file
        self.device = args.device

    

    @staticmethod
    def pad_encoded_text(encoded_text):
        # get the extra padding of encoded text
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += '0'
        # merge the "info" of extra padding in "string/bit" with encoded text
        # so we know how to truncate it later
        padded_info = "{0:08b}".format(extra_padding)
        new_text = padded_info + encoded_text
        return new_text

    @staticmethod
    def to_byte_array(padded_encoded_text):
        if len(padded_encoded_text) % 8 != 0:
            print('not padded properly')
            exit(0)
        b = bytearray()
        for i in range(
                0, len(padded_encoded_text), 8):  # loop every 8 character
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))  # base 2
        return b

    def train_model(self):
        args = self.args
        # Load data
        corpus = Corpus(args.file)
        train_data = train.batchify(corpus.train, args.batch_size, self.device)
        # Build the model
        ntokens = len(corpus.dictionary)
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid,
                               args.nlayers, args.dropout, args.tied).to(self.device)
        # criterion = nn.NLLLoss()
        # criterion = nn.MSELoss()
        criterion = self.args.criterion
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Training code
        # Loop over epochs.
        lr = args.lr
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train.train(train_data, args, model, optimizer, criterion,
                            corpus, epoch, lr, self.device)
                print('-' * 89)
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        return model

    def load_model(self):
        checkpoint = self.args.model_file
        with open(checkpoint, 'rb') as f:
            model = torch.load(f).to(self.device)
        model.eval()
        return model

    def compress(self, filename, load_model=False):
        start = time.time()
        data, vocab = read_characters(self.fname)
        comp_text = ''
        dictionary = Corpus(filename).dictionary
        context_map = Context(dictionary)
        fixed_huffman = HuffmanCoding()
        counts = fixed_huffman.make_frequency_dict(data)
        fixed_huffman.make_heap_node(counts)
        fixed_huffman.merge_nodes()
        fixed_huffman.encode()
        # window size
        k = 10
        # actual window size (we predict last character based on previous k)
        k += 1
        # start symbols
        data = ['<s>'] * (k - 1) + data
        # generator
        g = (data[t:t+k] for t in range(len(data) - k + 1))
        i = 0
        pct = 0
        count_unks = 0
        count_original_size = ''
        if load_model:
            model = self.load_model()
        else:
            model = self.train_model()
            model.eval()
        for window in g:
            i += 1
            if i > self.args.limit and not self.args.limit == -1:
                # print('unk ratio:', count_unks / i)
                print()
                print('original size: ', len(count_original_size.encode('ascii')))
                break
            # predict last char in window based on previous 10
            context = window[:-1]
            char = window[-1]
            count_original_size += char
            huffman = HuffmanCoding()
            prob = huffman.make_context_frequency_dict(context, model, context_map,
                                                       self.device, threshold=self.args.threshold)
            huffman.make_heap_node(prob)
            huffman.merge_nodes()
            huffman.encode()
            if char in prob:
                # survived the threshold
                encoding = '0' + huffman.get_encoded_word(char)
            else:
                # unknown, we use fixed huffman
                encoding = '1' + fixed_huffman.get_encoded_word(char)
                count_unks += 1
            comp_text += encoding
            # print progress
            if self.args.limit == -1:
                current_pct = i / len(data) * 100
            else:
                current_pct = i / self.args.limit * 100
            if int(current_pct) > pct:
                pct += 1
                sys.stdout.write('\r')
                sys.stdout.write("[%-100s] %d%%" % ('=' * pct, pct))
                sys.stdout.flush()
        padded_encoded_text = self.pad_encoded_text(comp_text)
        byte_array_huff = self.to_byte_array(padded_encoded_text)
        filename_split = filename.split('.')
        compressed_filename = filename_split[0] + "_compressed.bin"
        # write compressed file
        torch.save({
            'word2idx': dictionary.word2idx,
            'idx2word': dictionary.idx2word,
            'model_state_dict': model.state_dict(),
            'fixed_huffman_counts': counts,
            'bytes': bytes(byte_array_huff),
        }, compressed_filename)
        with open('temporal.bin', 'wb') as file:
            file.write(bytes(byte_array_huff))
        # true compressed size in bytes
        print('compressed bytes: ', os.path.getsize('temporal.bin'))
        # print(os.path.getsize(compressed_filename))
        # MISC
        print('Compression Done!')
        get_original_filesize = os.path.getsize(filename)
        get_compressed_filesize = os.path.getsize(compressed_filename)
        percentage = (get_compressed_filesize / get_original_filesize) * 100
        print(round(percentage, 3), "%")
        end = time.time()
        print(round((end - start), 3), "s")

    @staticmethod
    def remove_padding(padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)
        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-extra_padding]
        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ''
        decoded_text = ''
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ''
        return decoded_text

    def decompress(self, compressedfile):
        start = time.time()
        filename_split = compressedfile.split('_')
        checkpoint = torch.load(compressedfile, map_location=self.device)
        body = checkpoint['bytes']
        dictionary = Dictionary()
        dictionary.word2idx = checkpoint['word2idx']
        dictionary.idx2word = checkpoint['idx2word']
        context_map = Context(dictionary)
        ntokens = len(dictionary)
        model = RNNModel('LSTM', ntokens, 200, 200, 2, dropout=0.2, tie_weights=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        bit_string = ''
        join_body = list(body)
        for i in join_body:
            bit_string += "{0:08b}".format(i)
        encoded_text = self.remove_padding(bit_string)
        # decompress start here
        current_code = ''
        decoded_text = ''
        # we define an initial context
        # then we predict the initial huffman tree
        # read bits until we get to a leaf
        # convert the leaf to a char and add it to decompressed text
        # update the context and repeat the process
        context = ['<s>'] * 10
        def tree_from_context(context):
            huffman = HuffmanCoding()
            prob = huffman.make_context_frequency_dict(context, model, context_map,
                                                       self.device, threshold=self.args.threshold)
            huffman.make_heap_node(prob)
            huffman.merge_nodes()
            huffman.encode()
            huffman.reverse_mapping = {v: k for k, v in huffman.codes.items()}
            return huffman
        huffman = tree_from_context(context)
        fixed_huffman = HuffmanCoding()
        counts = checkpoint['fixed_huffman_counts']
        fixed_huffman.make_heap_node(counts)
        fixed_huffman.merge_nodes()
        fixed_huffman.encode()
        fixed_huffman.reverse_mapping = {v: k for k, v in fixed_huffman.codes.items()}
        flag = None
        for bit in encoded_text:
            if flag == '0':
                current_code += bit
                if current_code in huffman.reverse_mapping:
                    next_char = huffman.reverse_mapping[current_code]
                    decoded_text += next_char
                    current_code = ''
                    context = context[1:] + [next_char]
                    huffman = tree_from_context(context)
                    flag = None
                continue
            elif flag == '1':
                current_code += bit
                if current_code in fixed_huffman.reverse_mapping:
                    next_char = fixed_huffman.reverse_mapping[current_code]
                    decoded_text += next_char
                    current_code = ''
                    context = context[1:] + [next_char]
                    huffman = tree_from_context(context)
                    flag = None
                continue
            else:
                flag = bit
        # write decompressed file
        with open(filename_split[0] + "_decompressed.txt", 'w') as f:
            f.writelines(decoded_text)
        print('Decompression Done!')
        end = time.time()
        print(round((end - start), 3), "s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        help='compress or decompress')
    parser.add_argument('file', type=str, default='dickens.txt',
                        help='location of the file')
    parser.add_argument('--device', type=str,
                        help='GPU device')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--criterion', type=str, default='CE',
                        help='loss function - CE, L1 or L2')
    parser.add_argument('--bptt', type=int, default=280,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--threshold', type=float, default=1e-2,
                        help='pruning threshold (int or float)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='number of characters to compress/decompress. -1 for no limit')
    parser.add_argument('--model_file', type=str, default=None,
                        help='trained model file')

    args = parser.parse_args()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    huffmanLSTM = HuffmanLSTMCompressor(args)
    if args.task == 'compress':
        if args.model_file:
            huffmanLSTM.compress(args.file, load_model=True)
        else:
            huffmanLSTM.compress(args.file, load_model=False)
    elif args.task == 'decompress':
        huffmanLSTM.decompress(args.file)
    else:
        print("command not found")
        exit(0)
    pass
