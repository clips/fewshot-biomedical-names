import torch.nn as nn
import torch


######################################################
######################################################
##################      FNN      #####################
######################################################
######################################################

class FNNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, nonlinear=True, seed=0, average=True):

        super(FNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.average = average

        self.final = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        if not self.num_layers:
            assert hidden_size == input_size
            self.hidden = []
            return

        # first hidden layers
        if self.seed:
            torch.manual_seed(self.seed)
        self.hidden = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.hidden_size)])

        if nonlinear:
            self.hidden.extend([nn.ReLU()])
        self.hidden.extend([nn.Dropout(self.dropout_rate)])

        # optional deep layers
        for k in range(1, self.num_layers):
            if self.seed:
                torch.manual_seed(self.seed)
            self.hidden.extend([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)])
            if nonlinear:
                self.hidden.extend([nn.ReLU()])
            self.hidden.extend([nn.Dropout(self.dropout_rate)])

        # output linear function (readout)
        if self.seed:
            torch.manual_seed(self.seed)  # make difference in seed between hidden layer and final layer!

        print('Input size: {} / Hidden size: {} / Output size: {} / # hidden layers: {}'.format(
            self.input_size, self.hidden_size, self.output_size, self.num_layers))

    def forward(self, x):

        y = x
        for i in range(len(self.hidden)):
            y = self.hidden[i](y)

        out = self.final(y)

        if self.average:
            out = torch.mean(torch.stack([out, x]), dim=0)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
