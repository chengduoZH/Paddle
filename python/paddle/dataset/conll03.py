# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.dataset.common
import collections
import re
import string
import paddle.fluid as fluid

__all__ = ['build_dict', 'convert', 'train', 'test']

place = fluid.CPUPlace()


def build_dict(path, cutoff, pos):
    word_freq = collections.defaultdict(int)
    f = open(path, 'r')
    for line in f:
        words = line.split()
        if len(words) == 2:
            word_freq[words[pos]] += 1

    f.close()
    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())

    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, xrange(len(words))))
    if pos != 1:
        word_idx['<unk>'] = len(words)
    return word_idx


def build_char_dict(path, cutoff, pos):
    word_freq = collections.defaultdict(int)
    f = open(path, 'r')
    for line in f:
        words = line.split()
        if len(words) == 2:
            tmp_words = words[pos]
            for char in tmp_words:
                word_freq[char] += 1

    f.close()
    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())

    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx


def reader_creator(path, word_idx, label_idx, char_idx):
    UNK = word_idx['<unk>']
    INS = []

    def load(path, out):
        ln = -1
        f = open(path, 'r')
        out_tmp_words = []
        out_tmp_labels = []
        out_tmp_word_char = []
        out_tmp_sent_char = []
        for line in f:
            ln += 1
            if ln < 2:
                continue
            if line.strip() != '':
                words = line.split()
                out_tmp_words.append(word_idx.get(words[0], UNK))
                out_tmp_labels.append(label_idx[words[1]])
                for char in words[0]:
                    out_tmp_word_char.append(char_idx.get(char, UNK))

                out_tmp_sent_char.append(out_tmp_word_char)
                out_tmp_word_char = []
            else:
                out.append((out_tmp_words, out_tmp_labels, out_tmp_sent_char))
                out_tmp_words = []
                out_tmp_labels = []
                out_tmp_sent_char = []
        f.close()

    load(path, INS)
    print INS[0]

    def reader():
        for doc, label, char in INS:
            yield doc, label, char

    return reader


def reader_creator_word(path, word_idx, label_idx):
    UNK = word_idx['<unk>']
    INS = []

    def load(path, out):
        ln = -1
        f = open(path, 'r')
        out_tmp_words = []
        out_tmp_labels = []
        for line in f:
            ln += 1
            if ln < 2:
                continue
            if line.strip() != '':
                words = line.split()
                out_tmp_words.append(word_idx.get(words[0], UNK))
                out_tmp_labels.append(label_idx[words[1]])
            else:
                out.append((out_tmp_words, out_tmp_labels))
                out_tmp_words = []
                out_tmp_labels = []
        f.close()

    load(path, INS)
    print INS[0]

    def reader():
        for doc, label in INS:
            yield doc, label

    return reader


def load_dict(path):
    f = open(path, 'r')
    words = []
    for line in f:
        line = line.strip()
        words.append(line)
    f.close()
    word_idx = dict(zip(words, xrange(len(words))))
    return word_idx


def train(word_idx, label_idx, char_idx):
    return reader_creator('/root/.cache/paddle/dataset/conll03/eng.train.bioes',
                          word_idx, label_idx, char_idx)


def test(word_idx, label_idx, char_idx):
    return reader_creator('/root/.cache/paddle/dataset/conll03/eng.testb.bioes',
                          word_idx, label_idx, char_idx)


def valid(word_idx, label_idx, char_idx):
    return reader_creator('/root/.cache/paddle/dataset/conll03/eng.testa.bioes',
                          word_idx, label_idx, char_idx)


def train_word(word_idx, label_idx):
    return reader_creator_word(
        '/root/.cache/paddle/dataset/conll03/eng.train.bioes', word_idx,
        label_idx)


def test_word(word_idx, label_idx):
    return reader_creator_word(
        '/root/.cache/paddle/dataset/conll03/eng.testb.bioes', word_idx,
        label_idx)


def valid_word(word_idx, label_idx):
    return reader_creator_word(
        '/root/.cache/paddle/dataset/conll03/eng.testa.bioes', word_idx,
        label_idx)


def test_debug(word_idx, label_idx, char_idx):
    return reader_creator('/root/.cache/paddle/dataset/conll03/eng.testb.debug',
                          word_idx, label_idx, char_idx)


def word_dict():
    #word_idx = load_dict(
    #    '/home/disk3/wangsheng07/book-develop/09.lstm_crf/emb_data/word_dict.vocab'
    #)
    #return word_idx
    return build_dict('/root/.cache/paddle/dataset/conll03/eng.train.bioes', 0,
                      0)


def char_dict():
    char_dict = build_char_dict(
        '/root/.cache/paddle/dataset/conll03/eng.train.bioes', 0, 0)
    #print char_dict
    return char_dict


def label_dict():
    label_dict = {
        'E-PER': 10,
        'S-ORG': 7,
        'E-MISC': 14,
        'I-PER': 9,
        'I-LOC': 1,
        'B-ORG': 4,
        'O': 16,
        'S-MISC': 15,
        'S-PER': 11,
        'B-PER': 8,
        'B-MISC': 12,
        'I-MISC': 13,
        'E-ORG': 6,
        'S-LOC': 3,
        'I-ORG': 5,
        'B-LOC': 0,
        'E-LOC': 2
    }
    return label_dict


if __name__ == '__main__':
    #train(word_dict(),label_dict(),char_dict())
    train_word(word_dict(), label_dict())
    #test(word_dict(),label_dict())
