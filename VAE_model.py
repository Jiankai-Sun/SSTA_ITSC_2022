import torch
import torch.nn as nn

class VanillaVAE(nn.Module):
    def __init__(self, input_dim, h_units, act, args):
        super(VanillaVAE, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2
        width = args.img_width
        latent_dim = args.vae_latent_dim
        modules = []
        in_channels = input_dim

        modules.append(
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels=h_units[0],
                          kernel_size=self.filter_size, stride=1,
                          padding=self.padding),
                # [args.num_past, self.filter_size, self.filter_size]
                # [0, self.padding, self.padding]
                # nn.LayerNorm([h_units[0], 1, width, width]),
                nn.LeakyReLU())
        )
        in_channels = h_units[0]
        for h_dim in h_units[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=self.filter_size, stride=1, padding=self.padding),
                    # nn.LayerNorm([h_dim, 1, width, width]),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Conv3d(h_units[-1], out_channels=latent_dim,
                              kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.fc_var = nn.Conv3d(h_units[-1], out_channels=latent_dim,
                              kernel_size=self.filter_size, stride=1, padding=self.padding)

        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, h_units[-1] * 4)

        h_units.reverse()
        in_channels = latent_dim
        if len(h_units) == 1:
            for i in range(len(h_units)):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(in_channels,
                                           h_units[i],
                                           kernel_size=self.filter_size,
                                           stride=1,
                                           padding=self.padding),
                        # nn.LayerNorm([h_units[i+1], 1, width, width]),
                        nn.LeakyReLU())
                )
                in_channels = h_units[i]
        else:
            for i in range(len(h_units) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(in_channels,
                                           h_units[i+1],
                                           kernel_size=self.filter_size,
                                           stride=1,
                                           padding=self.padding),
                        # nn.LayerNorm([h_units[i+1], 1, width, width]),
                        nn.LeakyReLU())
                )
                in_channels = h_units[i+1]

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=3,
                      kernel_size=self.filter_size, padding=self.padding),
            nn.Sigmoid())

    def __call__(self, x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list):
        pred_x_tp1, message, memory, h_t = self.forward(x)
        return pred_x_tp1, message, memory, h_t

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input.permute(0, 4, 1, 2, 3)
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = z  # self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        eps = torch.ones_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        output = output.permute(0, 2, 3, 4, 1)
        # message.shape:  (10, 1, 1, 128, 128), (10, 1, 128, 128, 1)
        return  [output, z.permute(0, 2, 3, 4, 1), mu, log_var]

    def get_message(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z.permute(0, 2, 3, 4, 1)