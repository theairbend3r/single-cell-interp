import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How does a single-cell autoencoder represent the interferon response?
    """)


@app.cell
def _():
    import marimo as mo

    import itertools

    import gseapy
    import pertpy.data
    import scanpy as sc

    import numpy as np
    import sklearn as sk
    import scipy as sp
    import polars as pl

    import matplotlib.pyplot as plt
    import seaborn as sns

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from ignite.metrics import Accuracy, Loss
    from ignite.handlers import EarlyStopping
    from ignite.engine import (
        Engine,
        Events,
        create_supervised_trainer,
        create_supervised_evaluator,
    )

    sns.set_style("darkgrid")
    return (
        DataLoader,
        EarlyStopping,
        Engine,
        Events,
        Loss,
        TensorDataset,
        gseapy,
        itertools,
        mo,
        nn,
        np,
        pertpy,
        pl,
        plt,
        sc,
        sk,
        sns,
        sp,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Analysis
    """)


@app.cell
def _(pertpy):
    adata = pertpy.data.kang_2018()
    adata.var_names_make_unique()
    return (adata,)


@app.cell
def _(adata, pl):
    df_obs = pl.from_pandas(adata.obs.reset_index())
    df_obs.head()
    return (df_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the study design balanced?
    """)


@app.cell(hide_code=True)
def _(df_obs, plt, sns):
    def _():
        fig, axes = plt.subplot_mosaic(
            """
            abc
            ddd
            """,
            figsize=(15, 8),
            sharex=False,
            sharey=False,
            constrained_layout=True,
        )
        sns.countplot(data=df_obs.to_pandas(), x="label", ax=axes["a"])

        sns.countplot(
            data=df_obs.to_pandas(), y="cell_type", hue="label", ax=axes["b"]
        ).tick_params(axis="x", labelrotation=45)

        sns.countplot(data=df_obs.to_pandas(), y="replicate", hue="label", ax=axes["c"])

        sns.countplot(
            data=df_obs.to_pandas(),
            x="replicate",
            hue="cell_type",
            ax=axes["d"],
        )

        fig.suptitle("Is the Study Design Balanced?")

        return fig

    _()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quality control
    """)


@app.cell(hide_code=True)
def _(df_obs, plt, sns):
    def _():
        fig, axes = plt.subplot_mosaic(
            """
            ab
            cc
            """,
            figsize=(15, 8),
            sharex=False,
            sharey=False,
            constrained_layout=True,
        )

        sns.violinplot(
            data=df_obs.to_pandas(),
            y="nCount_RNA",
            x="cell_type",
            hue="label",
            log_scale=True,
            split=True,
            inner="quartile",
            ax=axes["a"],
        ).set(title="number of detected mRNA molecules across all genes in a cell")
        axes["a"].tick_params(axis="x", labelrotation=45)

        sns.violinplot(
            data=df_obs.to_pandas(),
            y="nFeature_RNA",
            x="cell_type",
            log_scale=True,
            hue="label",
            split=True,
            inner="quartile",
            ax=axes["b"],
        ).set(title="number of genes with at least one detected mRNA molecule in a cell")
        axes["b"].tick_params(axis="x", labelrotation=45)

        sns.scatterplot(
            data=df_obs.to_pandas(),
            x="nCount_RNA",
            y="nFeature_RNA",
            hue="cell_type",
            s=10,
            linewidth=0,
            alpha=0.5,
            ax=axes["c"],
        )
        axes["c"].set(xscale="log", yscale="log")

        return fig

    _()


@app.cell
def _(np, pl, sp):
    def get_gene_contrib(X, gene_names):
        total_molecules_per_cell = np.asarray(X.sum(axis=1)).ravel()
        gene_contrib = sp.sparse.diags(1 / total_molecules_per_cell) @ X
        avg_gene_contrib = np.asarray(gene_contrib.mean(axis=0)).ravel()

        df_avg_gene_contrib = (
            pl.DataFrame(
                {
                    "gene": gene_names,
                    "avg_pct_per_cell": avg_gene_contrib * 100,
                }
            )
            .sort("avg_pct_per_cell", descending=True)
            .with_columns(pl.col("avg_pct_per_cell").cum_sum().alias("cum_pct"))
        )

        return df_avg_gene_contrib

    return (get_gene_contrib,)


@app.cell
def _(adata, get_gene_contrib):
    df_gc = get_gene_contrib(X=adata.X, gene_names=adata.var_names)
    df_gc
    return (df_gc,)


@app.cell(hide_code=True)
def _(df_gc, plt, sns):
    def _():
        fig, axes = plt.subplot_mosaic(
            """
        ab
        """,
            figsize=(15, 6),
            constrained_layout=True,
        )

        top = df_gc.head(30).to_pandas()

        sns.barplot(data=top, y="gene", x="avg_pct_per_cell", ax=axes["a"]).set(
            title="Top 30 genes: avg % of a cell's molecules",
            xlabel="avg % per cell",
            ylabel="",
        )

        df_cum = df_gc.with_row_index("num_genes", offset=1).to_pandas()
        ax = axes["b"]
        sns.lineplot(data=df_cum, x="num_genes", y="cum_pct", ax=ax).set(
            title="Cumulative % of molecules by gene rank"
        )

        for thr in [50, 80, 90]:
            n = int((df_cum["cum_pct"] < thr).sum()) + 1
            ax.axhline(thr, ls="--", c="grey", lw=0.7)
            ax.plot(n, thr, "o", c="C3")
            ax.annotate(
                f"{n} genes ({thr}%)",
                (n, thr),
                xytext=(5, -12),
                textcoords="offset points",
            )

        return fig

    _()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interferon Alpha Response genes
    """)


@app.cell
def _(gseapy):
    hallmark = gseapy.get_library("MSigDB_Hallmark_2020", organism="Human")
    genes_ISG = hallmark["Interferon Alpha Response"]
    genes_ISG
    return (genes_ISG,)


@app.cell
def _(gseapy):
    msigdb = gseapy.Msigdb().get_gmt(category="c2.all", dbver="2023.2.Hs")
    genes_HKP = msigdb["HSIAO_HOUSEKEEPING_GENES"]
    genes_HKP
    return (genes_HKP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocess Data
    """)


@app.cell
def _(adata, genes_HKP, genes_ISG, sc):
    ad = adata.copy()
    ad.layers["counts"] = ad.X.copy()
    sc.pp.filter_genes(ad, min_cells=10)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, batch_key="replicate")

    gene_mask = ad.var["highly_variable"] | ad.var_names.isin(genes_ISG + genes_HKP[:10])
    ad = ad[:, gene_mask].copy()
    print(ad.shape)
    return (ad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deep Learning
    """)


@app.cell
def _(ad, itertools):
    def make_traintest_replicates(replicates, n_test=1):
        reps = sorted(replicates)
        return [
            {"test": list(test), "train": [r for r in reps if r not in test]}
            for test in itertools.combinations(reps, n_test)
        ]

    folds = make_traintest_replicates(replicates=ad.obs["replicate"].unique().tolist(), n_test=1)
    train_replicates = folds[0]["train"]
    test_replicates = folds[0]["test"]

    train_replicates, test_replicates
    return test_replicates, train_replicates


@app.cell
def _(
    DataLoader,
    TensorDataset,
    ad,
    np,
    sk,
    test_replicates,
    torch,
    train_replicates,
):
    replicates = ad.obs["replicate"].astype(str).to_numpy()
    trainval_idx = np.where(np.isin(replicates, train_replicates))[0]
    test_idx = np.where(np.isin(replicates, test_replicates))[0]

    strata = (ad.obs["replicate"].astype(str) + "_" + ad.obs["label"].astype(str)).to_numpy()
    train_idx, val_idx = sk.model_selection.train_test_split(
        trainval_idx,
        test_size=0.1,
        random_state=99,
        stratify=strata[trainval_idx],
    )

    x_all = torch.tensor(ad.X.toarray(), dtype=torch.float32)
    x_train = x_all[train_idx]
    x_val = x_all[val_idx]
    x_test = x_all[test_idx]

    loader_train = DataLoader(TensorDataset(x_train), batch_size=1024, shuffle=True)
    loader_val = DataLoader(TensorDataset(x_val), batch_size=1024, shuffle=False)
    loader_test = DataLoader(TensorDataset(x_test), batch_size=1024, shuffle=False)

    print(
        f"train: {len(train_idx)} cells | val: {len(val_idx)} cells | test: {len(test_idx)} cells"
    )

    print(f"train/val donors: {sorted(train_replicates)}\ntest donors: {sorted(test_replicates)}")
    return (
        loader_test,
        loader_train,
        loader_val,
        test_idx,
        trainval_idx,
        x_all,
        x_train,
    )


@app.cell
def _(nn):
    class AutoEncoder(nn.Module):
        def __init__(self, n_genes, n_hidden_1, n_hidden_2, n_latent):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(n_genes, n_hidden_1),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_hidden_2),
                nn.ReLU(),
                nn.Linear(n_hidden_2, n_latent),
            )
            self.dec = nn.Sequential(
                nn.Linear(n_latent, n_hidden_2),
                nn.ReLU(),
                nn.Linear(n_hidden_2, n_hidden_1),
                nn.ReLU(),
                nn.Linear(n_hidden_1, n_genes),
            )

        def forward(self, x):
            h = self.enc(x)
            xhat = self.dec(h)
            return h, xhat

    return (AutoEncoder,)


@app.cell
def _(AutoEncoder, nn, torch, x_train):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = AutoEncoder(
        n_genes=x_train.shape[1],
        n_hidden_1=256,
        n_hidden_2=100,
        n_latent=32,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    return criterion, device, model, optimiser


@app.cell
def _(
    EarlyStopping,
    Engine,
    Events,
    Loss,
    criterion,
    device,
    loader_train,
    loader_val,
    model,
    optimiser,
    torch,
):
    def train_step(engine, batch):
        model.train()
        x = batch[0].to(device)
        optimiser.zero_grad()
        _, x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimiser.step()
        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x = batch[0].to(device)
            _, x_hat = model(x)
            return x_hat, x

    trainer = Engine(train_step)
    train_evaluator = Engine(eval_step)
    val_evaluator = Engine(eval_step)

    Loss(criterion).attach(train_evaluator, "loss")
    Loss(criterion).attach(val_evaluator, "loss")

    history = []

    def score_function(engine):
        return -engine.state.metrics["loss"]

    early_stop = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)

    val_evaluator.add_event_handler(Events.COMPLETED, early_stop)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        train_evaluator.run(loader_train)
        val_evaluator.run(loader_val)
        tr = train_evaluator.state.metrics["loss"]
        va = val_evaluator.state.metrics["loss"]
        history.append({"epoch": engine.state.epoch, "train": tr, "val": va})

        if engine.state.epoch % 10 == 0:
            print(f"Epoch {engine.state.epoch:3d} | train MSE {tr:.4f} | val MSE {va:.4f}")

    return history, trainer, val_evaluator


@app.cell
def _(loader_train, trainer):
    trainer.run(loader_train, max_epochs=50)


@app.cell
def _(history, pl, plt):
    df_hist = pl.DataFrame(history)

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df_hist["epoch"], df_hist["train"], label="train")
    ax.plot(df_hist["epoch"], df_hist["val"], label="val")
    ax.set(xlabel="epoch", ylabel="MSE", title="Model training curve")
    ax.legend()


@app.cell
def _(loader_test, loader_train, val_evaluator):
    val_evaluator.run(loader_train)
    print(f"train MSE: {val_evaluator.state.metrics['loss']:.4f}")

    val_evaluator.run(loader_test)
    print(f"test MSE for held-out donor: {val_evaluator.state.metrics['loss']:.4f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis
    """)


@app.cell
def _(ad, device, model, np, test_idx, torch, trainval_idx, x_all):
    model.eval()
    with torch.no_grad():
        h_all = model(x_all.to(device))[0].cpu().numpy()

    labels = ad.obs["label"].astype(str).to_numpy()
    stim_idx = np.where(labels == "stim")[0]
    ctrl_idx = np.where(labels == "ctrl")[0]

    idx_stim_train = np.intersect1d(trainval_idx, stim_idx)
    idx_ctrl_train = np.intersect1d(trainval_idx, ctrl_idx)
    idx_stim_test = np.intersect1d(test_idx, stim_idx)
    idx_ctrl_test = np.intersect1d(test_idx, ctrl_idx)

    h_stim_train = h_all[idx_stim_train]
    h_ctrl_train = h_all[idx_ctrl_train]
    h_stim_test = h_all[idx_stim_test]
    h_ctrl_test = h_all[idx_ctrl_test]

    print(
        f"train: {len(h_stim_train)} stim, {len(h_ctrl_train)} ctrl | "
        f"test: {len(h_stim_test)} stim, {len(h_ctrl_test)} ctrl"
    )
    return (
        h_ctrl_test,
        h_ctrl_train,
        h_stim_test,
        h_stim_train,
        idx_ctrl_test,
        idx_stim_test,
    )


@app.cell
def _(h_ctrl_train, h_stim_train):
    # interferon direction from training donors only
    v = h_stim_train.mean(0) - h_ctrl_train.mean(0)
    return (v,)


@app.cell
def _(
    ad,
    device,
    genes_ISG,
    h_ctrl_test,
    idx_stim_test,
    model,
    np,
    torch,
    v,
    x_all,
):
    v_t = torch.tensor(v, device=device, dtype=torch.float32)
    with torch.no_grad():
        h_c = torch.tensor(h_ctrl_test, device=device)
        x_base = model.dec(h_c).cpu().numpy()  # decoded ctrl (before)
        x_steer = model.dec(h_c + v_t).cpu().numpy()  # decoded ctrl + v (after)

    x_real_stim = x_all[idx_stim_test].numpy()  # real stimulated cells
    ctypes = ad.obs["cell_type"].astype(str).to_numpy()
    gene_names = ad.var_names.to_numpy()
    isg_idx = np.where(np.isin(gene_names, genes_ISG))[0]
    return ctypes, isg_idx, x_base, x_real_stim, x_steer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Control cells after steering with interferon
    """)


@app.cell
def _(
    ad,
    h_ctrl_test,
    h_stim_test,
    idx_ctrl_test,
    idx_stim_test,
    np,
    plt,
    sk,
    sns,
    v,
):
    def _():
        ct = ad.obs["cell_type"].astype(str).to_numpy()
        pca = sk.decomposition.PCA(n_components=2).fit(np.vstack([h_ctrl_test, h_stim_test]))

        groups = [
            ("real ctrl", pca.transform(h_ctrl_test), ct[idx_ctrl_test]),
            ("real stim", pca.transform(h_stim_test), ct[idx_stim_test]),
            ("ctrl + v", pca.transform(h_ctrl_test + v), ct[idx_ctrl_test]),
        ]
        types = sorted(np.unique(ct))
        colors = dict(zip(types, sns.color_palette("tab10", len(types))))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

        # by condition
        for name, p, _ in groups:
            axes[0].scatter(p[:, 0], p[:, 1], s=3, alpha=0.4, label=name)
        axes[0].legend(markerscale=4)
        axes[0].set(title="By condition", xticks=[], yticks=[])

        # by cell type (real ctrl/stim as faint dots, steered as crosses)
        for name, p, c in groups:
            for t in types:
                m = c == t
                if name == "ctrl + v":
                    axes[1].scatter(p[m, 0], p[m, 1], s=12, marker="x", alpha=0.8, color=colors[t])
                else:
                    axes[1].scatter(
                        p[m, 0],
                        p[m, 1],
                        s=3,
                        alpha=0.15,
                        color=colors[t],
                        label=t if name == "real ctrl" else None,
                    )
        axes[1].legend(markerscale=4, fontsize=8)
        axes[1].set(title="By cell type (× = ctrl + v)", xticks=[], yticks=[])
        plt.suptitle("Held Out Donor")

        return fig

    _()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Average expression of the Interferon genes in each cell
    """)


@app.cell
def _(
    ctypes,
    idx_ctrl_test,
    idx_stim_test,
    isg_idx,
    pl,
    plt,
    sns,
    x_base,
    x_real_stim,
    x_steer,
):
    def _():
        score = lambda M: M[:, isg_idx].mean(1)
        df = pl.concat(
            [
                pl.DataFrame(
                    {
                        "cell_type": ctypes[idx_ctrl_test],
                        "state": "ctrl",
                        "ifn_score": score(x_base),
                    }
                ),
                pl.DataFrame(
                    {
                        "cell_type": ctypes[idx_ctrl_test],
                        "state": "ctrl + v",
                        "ifn_score": score(x_steer),
                    }
                ),
                pl.DataFrame(
                    {
                        "cell_type": ctypes[idx_stim_test],
                        "state": "real stim",
                        "ifn_score": score(x_real_stim),
                    }
                ),
            ]
        )
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.violinplot(
            df.to_pandas(),
            x="cell_type",
            y="ifn_score",
            hue="state",
            cut=0,
            inner="quartile",
            ax=ax,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.set(xlabel="", title="Interferon score per cell")
        return fig

    _()


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
