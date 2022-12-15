import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fast_histogram import histogram1d
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import *


def save_histogram(data, bins, clip, title, path, remove_zeros=False):
    data = data.todense().flatten()
    data = data[data != 0] if remove_zeros else data
    hist = histogram1d(data, bins=bins, range=clip)
    plt.ticklabel_format(style='plain')
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Values")
    plt.hist(np.linspace(start=clip[0], stop=clip[1], num=bins), weights=hist, bins=bins)
    plt.savefig(path, bbox_inches="tight")
    plt.clf()


def save_learning_curve(path, data, labels, title):
    for y, label in zip(data, labels):
        sns.scatterplot(x=np.arange(0, len(y)), y=y, label=label)
    plt.ticklabel_format(style='plain')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.clf()


def read_losses(path):
    data = pd.read_csv(path)
    train_rec_losses = data["Training reconstruction losses"]
    train_reg_losses = data["Training regularisation losses:"]
    test_rec_losses = data["Test reconstruction losses"]
    test_reg_losses = data["Test regularisation losses"]
    return train_rec_losses, train_reg_losses, test_rec_losses, test_reg_losses


def save_all_learning_curves(distributions, latent_sizes, results_path):
    for distr in distributions:
        for latent_size in latent_sizes[distr]:
            losses_file = f"results/{distr}_{latent_size}/losses.csv"
            train_rec, train_reg, test_rec, test_reg = read_losses(losses_file)
            loss_train = -train_rec + train_reg
            loss_test = -test_rec + test_reg
            train_rec, test_rec = -train_rec, -test_rec
            for datatype in ("training", "test"):
                elbo = loss_train if datatype == "training" else loss_test
                losses = [train_rec, train_reg] if datatype == "training" else [test_rec, test_reg]
                for figure_type in ("elbo", "losses"):
                    labels = [datatype] if figure_type == "elbo" else ["reconstruction loss", "regularisation loss"]
                    data = [elbo] if figure_type == "elbo" else losses
                    save_learning_curve(
                        path=f"{results_path}/learning_curve_{distr}_{latent_size}_{figure_type}_{datatype}.png",
                        data=data,
                        labels=labels,
                        title=f"ELBO for {datatype} data, latent size={latent_size}, decoder={distr}")


def get_latent_space(distribution, latent_size):
    xtrain = torch.tensor(anndata.read_h5ad("data/SAD2022Z_Project1_GEX_train.h5ad").X.todense())
    vae = torch.load(f"results/{distribution}_{latent_size}/model")
    latent_space = vae.encoder(xtrain)
    param1 = latent_space[0].detach().numpy()
    param2 = latent_space[1].detach().numpy()
    return param1, param2


def save_pca_plot(distribution, latent_size, results_path):
    labels = anndata.read_h5ad("data/SAD2022Z_Project1_GEX_train.h5ad").obs["cell_type"]
    params_names = ("alpha", "beta") if distribution == "gamma" else ("mu", "sigma")
    params = get_latent_space(distribution, latent_size)
    for param, name in zip(params, params_names):
        pca = PCA(n_components=2)
        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
        xt = pipeline.fit_transform(param)
        plt.title(f"PCA for latent {name} on training data colored by cell type, decoder={distribution}, latent space size={latent_size}")
        sns.scatterplot(x=xt[:, 0], y=xt[:, 1], hue=list(labels), legend=False)
        plt.savefig(f"{results_path}/PCA_plot_{name}_{latent_size}", bbox_inches="tight")
        plt.clf()


def find_component_nr(distribution, max_latent_size, expl_var_threshold):
    param1, param2 = get_latent_space(distribution, max_latent_size)
    max_comp_nr = float("-inf")
    for data in (param1, param2):
        pca = PCA(n_components=expl_var_threshold)
        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
        pipeline.fit_transform(data)
        comp_nr = pca.n_components_
        if comp_nr > max_comp_nr:
            max_comp_nr = comp_nr
    return max_comp_nr


def read_last_loss(distribution, latent_size):
    losses_file = f"results/{distribution}_{latent_size}/losses.csv"
    train_rec, train_reg, test_rec, test_reg = read_losses(losses_file)
    last_loss = -list(test_rec)[-1] + list(test_reg)[-1]
    return last_loss


def save_elbo_table(distribution, expl_var_thresholds, max_latent_size, results_path):
    component_numbers = [find_component_nr(distribution, max_latent_size, thr) for thr in expl_var_thresholds]
    losses = [read_last_loss(distribution, latent_size) for latent_size in component_numbers]
    df = pd.DataFrame({"explained variance": expl_var_thresholds,
                       "latent space size": component_numbers,
                       "-ELBO on last epoch": losses
                       })

    df.to_csv(f"{results_path}/{distribution}_loss_table.csv")


def main():
    test_data = anndata.read_h5ad("data/SAD2022Z_Project1_GEX_test.h5ad")
    train_data = anndata.read_h5ad("data/SAD2022Z_Project1_GEX_train.h5ad")
    results_path = "eval_results"
    os.makedirs(results_path, exist_ok=True)
    data_to_labels = {("preprocessed", "training"): train_data.X,
                      ("preprocessed", "test"): test_data.X,
                      ("raw", "training"): train_data.layers["counts"],
                      ("raw", "test"): test_data.layers["counts"]
                      }
    for is_processed, is_training in data_to_labels.keys():
        data = data_to_labels[is_processed, is_training]
        save_histogram(data,
                       bins=22,
                       clip=(0, 10),
                       title=f"Histogram of the {is_processed} {is_training} data",
                       path=f"{results_path}/histogram_{is_processed}_{is_training}",
                       )
        save_histogram(data,
                       bins=33,
                       clip=(0, 10),
                       title=f"Histogram of non-zero values of the {is_processed} {is_training} data",
                       path=f"{results_path}/histogram_{is_processed}_{is_training}_non_zero",
                       remove_zeros=True
                       )

    plotted_latent_sizes = {"normal": [50, 4], "gamma": [50, 7]}
    latent_sizes = {"normal": [3, 4, 6, 50], "gamma": [3, 6, 7, 50]}
    distributions = plotted_latent_sizes.keys()
    save_all_learning_curves(distributions, plotted_latent_sizes, results_path)
    explained_variance_thresholds = [0.99, 0.95, 0.8]
    for distr in distributions:
        save_elbo_table(distr, explained_variance_thresholds, 50, results_path)
        for latent_size in latent_sizes[distr]:
            save_pca_plot(distr, latent_size, results_path)


if __name__ == "__main__":
    main()
