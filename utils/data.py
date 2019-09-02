import torch
import torch.utils.data as data_util
import numpy as np
from scipy.stats import truncnorm
import jieba


class Datasets():
    def __init__(self, config):
        self.train_text = self._get_datasets_text(config.filename_train_src)
        self.train_summary = self._get_datasets_summary(config.filename_train_tgt)
        self.valid_text = self._get_datasets_text(config.filename_valid_src)
        self.valid_summary = self._get_datasets_summary(config.filename_valid_tgt)
        self.test_text = self._get_datasets_text(config.filename_test_src)
        self.test_summary = self._get_datasets_summary(config.filename_test_tgt)

    def _get_datasets_text(self, filename):
        text = []
        sent = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    text.append(sent)
                    sent = []
                else:
                    line = line.strip()
                    sent.append(line)
        text.append(sent)
        return text

    def _get_datasets_summary(self, filename):
        summary = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                summary.append(line)
        return summary


# save pt
def get_trimmed_datasets_src(datasets, word2idx, max_length):
    data = np.zeros([len(datasets), max_length])
    mask = np.zeros([len(datasets), 3, max_length])
    k = 0
    for sents in datasets:
        line = []
        start = 0
        end = 0
        m = np.zeros([3, max_length])
        for i in range(0, len(sents)):
            sent = list(sents[i])
            line = line + sent
            end = end + len(sent)
            if i <= 2:
                m[i][start:end] = 1
                start = end
        m[-1][start:end] = 1
        sen = np.zeros(max_length, dtype=np.int32)
        for i in range(max_length):
            if i == len(line):
                sen[i] = word2idx['<eos>']
                break
            else:
                flag = word2idx.get(line[i])
                if flag is None:
                    sen[i] = word2idx['<unk>']
                else:
                    sen[i] = word2idx[line[i]]
        data[k] = sen
        mask[k] = m
        k += 1
    data = torch.from_numpy(data).type(torch.LongTensor)
    mask1 = torch.from_numpy(mask[:, 0, :]).type(torch.FloatTensor)
    mask2 = torch.from_numpy(mask[:, 1, :]).type(torch.FloatTensor)
    mask3 = torch.from_numpy(mask[:, 2, :]).type(torch.FloatTensor)
    return data, mask1, mask2, mask3


# save pt
def get_trimmed_datasets_tgt(datasets, word2idx, max_length):
    data = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        line = line.strip()
        line = list(line)
        sen = np.zeros(max_length, dtype=np.int32)
        for i in range(max_length):
            if i == len(line):
                sen[i] = word2idx['<eos>']
                break
            else:
                flag = word2idx.get(line[i])
                if flag is None:
                    sen[i] = word2idx['<unk>']
                else:
                    sen[i] = word2idx[line[i]]
        data[k] = sen
        k += 1
    data = torch.from_numpy(data).type(torch.LongTensor)
    return data


def save_data(text, summary, word2idx, t_len, s_len, filename):
    text, mask1, mask2, mask3 = get_trimmed_datasets_src(text, word2idx, t_len)
    summary = get_trimmed_datasets_tgt(summary, word2idx, s_len)
    data = data_util.TensorDataset(text, mask1, mask2, mask3, summary)
    print('data save at ', filename)
    torch.save(data, filename)


def get_embeddings(config, vocab):
    embeddings = np.zeros((config.vocab_size, config.embedding_dim))
    flag = list(np.arange(0, 4000))
    with open(config.filename_embedding, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            if word in vocab.word2idx.keys():
                flag.remove(vocab.word2idx[word])
                embedding = [float(x) for x in line[1:]]
                embeddings[vocab.word2idx[word]] = embedding
    for i in flag:
        np.random.seed(i)
        embedding = truncnorm.rvs(-2, 2, size=config.embedding_dim)
        embeddings[i] = embedding
    embeddings = torch.from_numpy(embeddings)
    torch.save(embeddings, config.filename_trimmed_embedding)
    print('embeddings save at:', config.filename_trimmed_embedding)


def data_load(filename, batch_size, shuffle):
    data = torch.load(filename)
    data_loader = data_util.DataLoader(data, batch_size, shuffle=shuffle, num_workers=2)
    return data_loader


def get_datasets_clean(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            result.append(line)
    return result