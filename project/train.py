import os
import anndata
import pandas as pd
import torch
import torch.nn as nn
from anndata.experimental.pytorch import AnnLoader
from torch.distributions import Normal, kl_divergence, Gamma
from torch.distributions.independent import Independent

EPSILON = float("1e-08")


class NetworkGaussian(nn.Module):
    """ Multilayer perceptron with last two layers encoding mean and log variance of a gaussian"""

    def __init__(self, sizes):
        super(NetworkGaussian, self).__init__()
        layers = []
        for inp_size, out_size in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(inp_size, out_size))
            layers.append(nn.Sigmoid())
        self.mu_layer = nn.Linear(sizes[-2], sizes[-1])
        self.sigma_layer = nn.Linear(sizes[-2], sizes[-1])
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        y = self.seq.forward(x)
        return self.mu_layer(y), torch.exp(self.sigma_layer(y))


class NetworkGamma(nn.Module):
    """ Multilayer perceptron with last two layers encoding shape and rate of gamma distribution"""

    def __init__(self, sizes):
        super(NetworkGamma, self).__init__()
        layers = []
        for inp_size, out_size in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(inp_size, out_size))
            layers.append(nn.Sigmoid())
        self.alpha_layer = nn.Linear(sizes[-2], sizes[-1])
        self.beta_layer = nn.Linear(sizes[-2], sizes[-1])
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        y = self.seq(x)
        return torch.exp(self.alpha_layer(y)), torch.exp(self.beta_layer(y))


class EncoderGaussian(nn.Module):
    """ Gaussian encoder with standard normal prior"""

    def __init__(self, sizes):
        super().__init__()
        self.network = NetworkGaussian(sizes)

    def forward(self, x):
        return self.network.forward(x)

    @staticmethod
    def kl_loss(mu, sigma):
        latent_distribution = Independent(Normal(mu, sigma), 1)
        prior_mu, prior_sigma = torch.zeros_like(mu), torch.ones_like(sigma)
        prior_distribution = Independent(Normal(prior_mu, prior_sigma), 1)
        return torch.sum(kl_divergence(latent_distribution, prior_distribution))


class Decoder(nn.Module):
    """ VAE decoder, which can model gaussian or gamma distribution"""

    def __init__(self, sizes, is_gaussian=True):
        super().__init__()
        if is_gaussian:
            self.network = NetworkGaussian(sizes)
        else:
            self.network = NetworkGamma(sizes)
        self.is_gaussian = is_gaussian

    def forward(self, x):
        return self.network.forward(x)

    def log_prob(self, x, params):
        if self.is_gaussian:
            posterior = Independent(Normal(params[0], params[1]), 1)
        else:
            posterior = Gamma(params[0], params[1])
        return torch.sum(posterior.log_prob(x + EPSILON))


class VAE(nn.Module):
    """VAE with a gaussian encoder"""

    def __init__(self, encoder, decoder, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.normal(0, 1, (mu.shape[0], self.latent_size))
        sampled_vector = mu + eps * sigma
        decoder_params = self.decoder(sampled_vector)
        return mu, sigma, decoder_params

    def elbo(self, x, beta=1):
        batch_size = len(x)
        mu, sigma, decoder_params = self.forward(x)
        rec_loss = self.decoder.log_prob(x, decoder_params) / batch_size
        kl_loss = beta * self.encoder.kl_loss(mu, sigma) / batch_size
        return rec_loss, kl_loss


def get_loader(data_path, batch_size):
    data = anndata.read_h5ad(data_path)
    return AnnLoader([data], batch_size=batch_size, shuffle=True)


def train(vae, train_data_path, test_data_path, epochs, batch_size, learning_rate, beta=1):
    """ Train and test vae on normalised annotation data from a given path"""
    train_regloss, test_regloss, train_recloss, test_recloss = [], [], [], []
    train_loader, test_loader = get_loader(train_data_path, batch_size), get_loader(test_data_path, batch_size)
    optimiser = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_recloss = 0
        epoch_regloss = 0
        batch_nr = 0
        # train
        for batch_idx, data in enumerate(train_loader):
            optimiser.zero_grad()
            rec_loss, reg_loss = vae.elbo(data.X, beta=beta)
            epoch_regloss += reg_loss.item()
            epoch_recloss += rec_loss.item()
            loss = -rec_loss + reg_loss
            loss.backward()
            optimiser.step()
            batch_nr += 1
        train_recloss.append(epoch_recloss / batch_nr)
        train_regloss.append(epoch_regloss / batch_nr)
        # test
        batch_nr = 0
        epoch_recloss = 0
        epoch_regloss = 0
        for batch_idx, data in enumerate(test_loader):
            rec_loss, reg_loss = vae.elbo(data.X, beta)
            epoch_regloss += reg_loss.item()
            epoch_recloss += rec_loss.item()
            batch_nr += 1
        test_recloss.append(epoch_recloss / batch_nr)
        test_regloss.append(epoch_regloss / batch_nr)
        print(f'====> Epoch: {epoch}')
    return vae, train_recloss, train_regloss, test_recloss, test_regloss


def run_analysis(train_data_path, test_data_path, output_folder_path, encoder_sizes, decoder_distribution,
                 decoder_sizes, epochs, batch_size, learning_rate, beta=1):
    os.makedirs(output_folder_path, exist_ok=True)
    encoder = EncoderGaussian(encoder_sizes)
    if decoder_distribution == "normal":
        decoder = Decoder(decoder_sizes)
    elif decoder_distribution == "gamma":
        decoder = Decoder(decoder_sizes, is_gaussian=False)
    else:
        raise ValueError("Unsupported decoder distribution")
    vae = VAE(encoder, decoder, encoder_sizes[-1])
    vae_trained, train_recloss, train_regloss, test_recloss, test_regloss = train(vae, train_data_path, test_data_path,
                                                                                  epochs,
                                                                                  batch_size, learning_rate, beta)
    torch.save(vae_trained, f"{output_folder_path}/model")
    losses = pd.DataFrame.from_dict({"Training reconstruction losses": train_recloss,
                                     "Training regularisation losses": train_regloss,
                                     "Test reconstruction losses": test_recloss,
                                     "Test regularisation losses": test_regloss,
                                     })
    losses.to_csv(f"{output_folder_path}/losses.csv")


def main():
    latent_sizes = {"normal": (3, 4, 6, 50), "gamma": (3, 6, 7, 50)}
    results_path = "results"
    for distribution in ("normal", "gamma"):
        for latent_size in latent_sizes[distribution]:
            run_analysis(train_data_path="data/SAD2022Z_Project1_GEX_train.h5ad",
                         test_data_path="data/SAD2022Z_Project1_GEX_test.h5ad",
                         output_folder_path=f"{results_path}/{distribution}_{latent_size}",
                         encoder_sizes=[5000, 500, 300, latent_size],
                         decoder_sizes=[latent_size, 300, 500, 5000],
                         decoder_distribution=distribution,
                         epochs=50,
                         batch_size=64,
                         learning_rate=0.003
                         )


if __name__ == "__main__":
    main()
