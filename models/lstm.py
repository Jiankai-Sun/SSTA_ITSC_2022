import torch
import torch.nn as nn
import models.convolutional_rnn as crnn 

Conv2dLSTMCell = crnn.Conv2dLSTMCell
Conv3dLSTMCell = crnn.Conv3dLSTMCell


class DeterministicConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=2, args=None, conv_type='3d', ):
        super(DeterministicConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.nf = nf = hidden_channels
        self.conv_type = conv_type
        self.filter_size = 3  # args.filter_size
        self.padding = self.filter_size // 2
        self.width = args.img_width
        if self.conv_type == '3d':
            self.embed = nn.Sequential(
                nn.Conv3d(in_channels, nf, self.filter_size, 1,
                          self.padding),
                nn.LayerNorm([nf, 1, self.width, self.width]),
            )
            self.lstm = nn.ModuleList([Conv3dLSTMCell(nf, nf, 3) for i in range(self.n_layers)])
            self.output = nn.Sequential(
                nn.LayerNorm([nf, 1, self.width, self.width]),
                nn.Conv3d(nf, out_channels, self.filter_size, 1, self.padding),
            )
        else:
            self.embed = nn.Conv2d(in_channels, nf, 3, 1, 1)
            self.lstm = nn.ModuleList([Conv2dLSTMCell(nf, nf, 3) for i in range(self.n_layers)])
            self.output = nn.Sequential(
                    nn.Conv2d(nf, out_channels, 3, 1, 1),
                    nn.ReLU(True),
                    )

    def init_hidden(self, shape, device):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(shape, device=device),
                           torch.zeros(shape, device=device)))
        return hidden

    def forward(self, input, hidden=None):
        embedded = self.embed(input)
        h_in = embedded

        if hidden is None:
            hidden = self.init_hidden(h_in.shape, h_in.device)

        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        out = self.output(h_in)

        return out, hidden


class GaussianConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=1, batch_size=16, multiplier=1, conv_type='3d'):
        super(GaussianConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.nf = nf = hidden_channels*multiplier
        self.conv_type = conv_type
        if self.conv_type == '3d':
            self.embed = nn.Conv3d(in_channels, nf, 3, 1, 1)
            self.lstm = nn.ModuleList([Conv3dLSTMCell(nf, nf, 3) for i in range(self.n_layers)])
            self.mu_net = nn.Sequential(nn.Conv3d(nf, out_channels, 3, 1, 1))
            self.logvar_net = nn.Sequential(nn.Conv3d(nf, out_channels, 3, 1, 1))
        else:
            self.embed = nn.Conv2d(in_channels, nf, 3, 1, 1)
            self.lstm = nn.ModuleList([Conv2dLSTMCell(nf, nf, 3) for i in range(self.n_layers)])
            self.mu_net = nn.Sequential(nn.Conv2d(nf, out_channels, 3, 1, 1))
            self.logvar_net = nn.Sequential(nn.Conv2d(nf, out_channels, 3, 1, 1))

    def init_hidden(self, bs, h, w, device):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(bs, self.nf, h, w, device=device),
                           torch.zeros(bs, self.nf, h, w, device=device)))
        return hidden

    def reparameterize(self, mu, logvar):
        sigma = logvar.mul(0.5).exp_()
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)

    def forward(self, input, hidden=None):
        embedded = self.embed(input)
        h_in = embedded

        if hidden is None:
            hidden = self.init_hidden(h_in.size(0), h_in.size(2), h_in.size(3), h_in.device)

        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, hidden
