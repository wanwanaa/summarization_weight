from models.attention import *
from models.encoder import *
from models.decoder import *
from models.seq2seq import *
from models.embedding import *


def build_model(config):
    embeds = Embeds(config, config.vocab_size)
    # if config.attn_flag == 'multi':
    #     encoder = Encoder_multi(embeds, config)
    # else:
    encoder = Encoder(embeds, config)
    encoder_sent = Encoder_Sent()
    decoder = Decoder(embeds, config)
    model = Seq2seq(encoder, encoder_sent, decoder, config)
    return model


def load_model(config, filename):
    model = build_model(config)
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)