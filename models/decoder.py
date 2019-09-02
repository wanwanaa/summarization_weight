import torch
import torch.nn as nn
from models import *


class Decoder(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds

        self.rnn = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout,
        )

        self.attention = Luong_Attention(config)
        self.linear_hidden = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_attn = nn.Linear(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, encoder_output, sents, masks):
        """
        :param x: (batch, 1) decoder input
        :param h: (batch, n_layer, hidden_size)
        :param encoder_output: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  out (batch, 1, hidden_size) decoder output
                  h (batch, n_layer, hidden_size) decoder hidden state
        """
        e = self.embeds(x).unsqueeze(1) # (batch, 1, embedding_dim)
        out, h = self.rnn(e, h)
        attn_weights, context = self.attention(out, encoder_output, sents, masks)
        p = self.sigmoid(self.linear_hidden(context) + self.linear_hidden(out))
        out = p * context + (1-p) * out
        return attn_weights, out, h