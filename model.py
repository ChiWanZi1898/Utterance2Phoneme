import torch
from torch import nn
from collections import OrderedDict
from torch.nn.utils.rnn import *


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, use_gpu=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        self.input_layers = nn.Sequential(OrderedDict([
            ("conv_1d", nn.Conv1d(input_size, hidden_size * 2, 5, stride=1, padding=2)),
            ('relu', nn.ReLU())
        ]))

        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False
        )

        self.output_layer = nn.Sequential(OrderedDict([
            ("linear_0", nn.Linear(hidden_size * 2, 256)),
            ("relu_0", nn.ReLU()),
            ("linear_1", nn.Linear(256, output_size)),
        ]))

    def __call__(self, x, x_len):
        if self.use_gpu:
            x = x.cuda()

        x = self.input_layers(x.permute(1, 2, 0)).permute(2, 0, 1)
        packed_X = pack_padded_sequence(x, x_len, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        out = self.output_layer(out).log_softmax(2)

        return out, out_lens


