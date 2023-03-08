import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def input_transpose(sents, pad_token):
    """
    Pad list of sentences according to the longest sentence in the batch, and transpose the resulted sentences.

    Args:
        sents: (list[list[str]]): list of tokenized sentences, where each sentence
                                    is represented as a list of words
        pad_token: (str): padding token

    Returns:
        sents_padded: (list[list[str]]): list of padded and transposed sentences, where each element in this list
                                            should be a list of length len(sents), containing the ith token in each
                                            sentence. Sentences shorter than the max length sentence are padded out with
                                            the pad_token, such that each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### WRITE YOUR CODE HERE (~5 lines)
    # Get maximum sentence length
    max_sent_length = len(max(sents, key=len))
    # Create list of padded lists
    for idx, sent in enumerate(sents):
        sents_padded[idx] = sent + [pad_token] * (max_sent_length - len(sent))
    # Transpose it
    sents_padded = np.array(sents_padded).T.tolist()
    ### END OF YOUR CODE HERE

    return sents_padded


def read_corpus(file_path, source):
    data = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            sent = line.strip().split(' ')
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                sent = ['<s>'] + sent + ['</s>']
            data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """

    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)

        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss


if __name__ == '__main__':
    sents = [
        [10, 145, 252, 767, 24, 11, 2399, 1382, 2342, 31, 752, 8, 47, 23874, 200, 4, 1618, 16, 62, 381, 10770, 773, 16,
         684, 4, 54, 3649, 3, 19, 11, 3205, 4, 54, 59, 129, 333, 4, 19, 11, 8, 27, 3, 32, 5],
        [8, 248, 85, 247, 4, 29, 35, 32, 20, 3, 23975, 4, 20, 3963, 2972, 4, 20, 14340, 3, 81, 681, 838, 4, 140, 2688,
         19, 111, 1614, 3, 3, 30],
        [1360, 288, 1158, 634, 20, 3, 1470, 4, 22, 11, 3, 59, 333, 43, 904, 217, 1891, 4, 36, 8, 79, 37, 300, 497, 44,
         1704, 93, 3395, 930, 5],
        [199, 120, 25, 4, 117, 14, 4, 157, 3, 157, 4, 516, 1336, 324, 23733, 15, 924, 63, 20, 444, 260, 4, 183, 4952,
         4625, 6800, 1803, 5],
        [442, 19, 48, 52, 673, 120, 281, 4, 29, 14, 100, 4, 22, 13, 1246, 4, 89, 34, 48, 2775, 52, 4, 93, 2318, 16, 224,
         5],
        [1844, 4, 130, 1061, 12, 11514, 4848, 4, 6, 14, 171, 5409, 69, 1196, 6, 4299, 203, 4, 314, 14, 18, 3916, 33,
         4211, 371, 5],
        [6, 100, 8, 4, 155, 419, 26, 11, 1227, 2303, 189, 15824, 4, 130, 1034, 3, 4, 83, 519, 65, 10, 246, 8, 21, 5113,
         5], [100, 8, 4, 120, 11, 118, 4, 7, 10, 87, 569, 117, 145, 4, 12, 4, 22, 10, 50, 552, 1471, 50, 123, 5],
        [44, 1889, 26309, 7, 10350, 41, 7, 21924, 6, 596, 2258, 4, 15, 25806, 4, 66, 13, 18, 4409, 6093, 4, 9, 3, 5]]
    pad_token = 0
    input_transpose(sents, pad_token)
