import torch
import torch.nn as nn
from models import *


class Encoder(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.n_layer = config.n_layer
        self.hidden_size = config.hidden_size

        self.rnn = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )

    def forward(self, x):
        """
        :param x:(batch, t_len)
        :return: gru_h(n_layer, batch, hidden_size) lstm_h(h, c)
                  out(batch, t_len, hidden_size)
        """
        e = self.embeds(x)
        # out (batch, time_step, hidden_size*bidirection)
        # h (batch, n_layers*bidirection, hidden_size)
        encoder_out, h = self.rnn(e)
        encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        h = (h[0][::2].contiguous(), h[1][::2].contiguous())
        return h, encoder_out


class Encoder_Sent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_out, masks):
        mask = masks[0].unsqueeze(1).transpose(1, 2).type(torch.FloatTensor)
        # print(mask.size())
        # print(encoder_out.size())
        sent1 = encoder_out * masks[0].unsqueeze(1).transpose(1, 2)
        sent2 = encoder_out * masks[1].unsqueeze(1).transpose(1, 2)
        sent3 = encoder_out * masks[2].unsqueeze(1).transpose(1, 2)
        num = encoder_out.size(1)
        sent1 = (torch.sum(sent1, dim=1) / num).unsqueeze(1)
        sent2 = (torch.sum(sent2, dim=1) / num).unsqueeze(1)
        sent3 = (torch.sum(sent3, dim=1) / num).unsqueeze(1)
        sents = torch.cat((sent1, sent2), dim=1)
        sents = torch.cat((sents, sent3), dim=1)
        return sents