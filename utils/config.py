class Config():
    def __init__(self):
        # dataset
        self.filename_train_src = 'DATA/seg_data/src-train.txt'
        self.filename_train_tgt = 'DATA/raw_data/tgt-train.txt'
        self.filename_valid_src = 'DATA/seg_data/src-valid.txt'
        self.filename_valid_tgt = 'DATA/raw_data/tgt-valid.txt'
        self.filename_test_src = 'DATA/seg_data/src-test.txt'
        self.filename_test_tgt = 'DATA/raw_data/tgt-test.txt'

        # trimmed data LCSTS
        self.filename_trimmed_train = 'DATA/data/train.pt'
        self.filename_trimmed_valid = 'DATA/data/valid.pt'
        self.filename_trimmed_test = 'DATA/data/test.pt'

        # vocab
        self.filename_word2idx = 'DATA/data/word2index.pkl'
        self.filename_idx2word = 'DATA/data/index2word.pkl'
        self.vocab_size = 8250

        # bos eos
        self.bos = 2
        self.eos = 3

        # sequence length
        self.src_len = 150
        self.tgt_len = 50


        # filename
        #################################################
        self.filename_model = 'result/model/'
        self.filename_data = 'result/data/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        #################################################
        self.filename_gold = 'result/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 32
        self.embedding_dim = 512
        self.hidden_size = 512
        self.beam_size = 5
        self.n_layer = 2
        self.dropout = 0
        self.bidirectional = True
        self.optimzer = 'Adam'

        # settings
        self.attn_flag = True
