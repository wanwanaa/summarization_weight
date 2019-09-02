import torch
import torch.nn as nn
import math


class Luong_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer

        self.linear_in = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.linear_out = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output, encoder_out, sents, masks):
        """
        :param output: (batch, 1, hidden_size) decoder output
        :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  output (batch, 1, hidden_size) attention vector
        """
        out = self.linear_in(output) # (batch, 1, hidden_size)
        out = out.transpose(1, 2) # (batch, hidden_size, 1)
        attn_weights = torch.bmm(encoder_out, out).squeeze() # (batch, t_len)
        # print(sents.size())
        sents_weight = torch.bmm(sents, out).squeeze() # (batch, 3)

        sents_weight = self.softmax(sents_weight)
        weights = 0.0
        # print(sents_weight[:, 0].size())
        for i in range(3):
            weights = weights + self.softmax(masks[i]*attn_weights)*(sents_weight[:, i].unsqueeze(1))

        weights = weights.unsqueeze(1)
        context = torch.bmm(weights, encoder_out) # (batch, 1, hidden_size)
        output = self.linear_out(torch.cat((output, context), dim=2))

        return attn_weights, output