# MovieLens Recommender — Alternating Least Squares

A from-scratch implementation of Alternating Least Squares (ALS) matrix factorisation for collaborative filtering, applied to the MovieLens datasets (100K and 32M ratings). The project progresses from a simple bias-only model through full latent-factor ALS, and finally to a feature-regularised variant that incorporates movie metadata (genres, decade) into the item embeddings.

---

## Project Structure

```
project/
├── derivations.md              # Full mathematical derivations of all ALS update equations
├── requirements.txt            # Python dependencies
├── model/
│   └── all_data.npz            # Saved model parameters and training history
└── notebooks/
    ├── 01_100k_data.ipynb      # Prototyping and validation on MovieLens 100K
    ├── 02_32m_data-plots.ipynb # Exploratory Data Analysis on MovieLens 32M
    ├── 32m_training.ipynb      # Hyperparameter search and full model training on 32M
    ├── 03_32m_embeddings.ipynb # Embedding analysis and movie similarity
    └── utils.py                # ALS update functions (biases, embeddings, features)
```

---

## Notebooks

**`01_100k_data.ipynb`** — Development on MovieLens 100K. Covers data loading, train/test splitting, a global-mean baseline, bias-only ALS, and full ALS with latent factors. Includes convergence plots and RMSE evaluation.

**`02_32m_data-plots.ipynb`** — Exploratory analysis of the 32M dataset. Examines rating distributions, user activity, and movie popularity, with log-log scale visualisations of the long-tail structure.

**`32m_training.ipynb`** — Full training pipeline for the 32M dataset. Constructs the item feature matrix from genre and decade metadata, runs hyperparameter optimisation with Optuna, sweeps embedding dimensions, and saves the final trained model.

**`03_32m_embeddings.ipynb`** — Post-training analysis. Loads the saved model, visualises learned embeddings via PCA, explores genre and decade structure, and demonstrates nearest-neighbour movie similarity queries.

---

## Models

The rating prediction for user $m$ and item $n$ is:

$$\hat{r}_{mn} = \mathbf{u}_m^\top \mathbf{v}_n + b_m^{(u)} + b_n^{(i)}$$

Three model variants are implemented, each a strict extension of the previous:

**Bias-only ALS** — Learns per-user and per-item bias terms; no latent factors.

**ALS with latent factors** — Adds user embeddings $\mathbf{u}_m$ and item embeddings $\mathbf{v}_n$, solved via closed-form ALS updates.

**Feature-regularised ALS** — Regularises item embeddings toward a weighted average of learned feature embeddings (genres, decade), enabling generalisation to items with few ratings.

Full derivations of all closed-form update equations are in [`derivations.md`](derivations.md).

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `optuna`, `pandas`, `tqdm`.

**Data:** The notebooks download the MovieLens datasets automatically from GroupLens. Run `01_100k_data.ipynb` first to fetch the 100K data, and `02_32m_data-plots.ipynb` for the 32M data. The data directory is expected one level above the `notebooks/` folder.

---

## Results

The trained model parameters for the 32M dataset are saved in `model/all_data.npz` and loaded directly in `03_32m_embeddings.ipynb` for analysis without retraining. Hyperparameters ($\lambda$, $\tau$, $\gamma$, embedding dimension $k$) were tuned via Optuna.