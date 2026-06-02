# MovieLens Recommender — Alternating Least Squares

A from-scratch implementation of Alternating Least Squares (ALS) matrix factorisation for collaborative filtering, applied to the MovieLens datasets (100K and 32M ratings). The project progresses from a simple bias-only model through full latent-factor ALS, and finally to a feature-regularised variant that incorporates movie metadata (genres, decade) into the item embeddings.

---

## Project Structure

```
project/
├── derivations.ipynb           # Full mathematical derivations of all ALS update equations
├── requirements.txt            # Python dependencies
├── model/
│   └── all_data.npz            # Saved model parameters and training history
├── notebooks/
│   ├── 01_100k_data.ipynb      # Prototyping and validation on MovieLens 100K
│   ├── 02_32m_data-plots.ipynb # Exploratory Data Analysis on MovieLens 32M
│   ├── 32m_training.ipynb      # Hyperparameter search and full model training on 32M
│   ├── 03_32m_embeddings.ipynb # Embedding analysis and movie similarity
│   └── utils.py                # ALS update functions (biases, embeddings, features)
└── application/
    ├── requirements.txt        # Application-specific dependencies
    └── src/
        ├── app.py              # Streamlit web application
        ├── recommender.py      # Inference — embedding lookup and similarity search
        └── utils/
            └── tmdb.py         # TMDB API integration for movie posters
```

---

## Notebooks

**`01_100k_data.ipynb`** — Development on MovieLens 100K. Covers data loading, train/test splitting, a global-mean baseline, bias-only ALS, and full ALS with latent factors. Includes convergence plots and RMSE evaluation.

**`02_32m_data-plots.ipynb`** — Exploratory analysis of the 32M dataset. Examines rating distributions, user activity, and movie popularity, with log-log scale visualisations of the long-tail structure.

**`03_32m_embeddings.ipynb`** — Post-training analysis. Loads the saved model, visualises learned embeddings via PCA, explores genre and decade structure, and demonstrates nearest-neighbour movie similarity queries.

**`32m_training.ipynb`** — Full training pipeline for the 32M dataset. Constructs the item feature matrix from genre and decade metadata, runs hyperparameter optimisation with Optuna, sweeps embedding dimensions, and saves the final trained model.

---

## Models

The rating prediction for user $m$ and item $n$ is:

$$\hat{r}_{mn} = \mathbf{u}_m^\top \mathbf{v}_n + b_m^{(u)} + b_n^{(i)}$$

Three model variants are implemented, each a strict extension of the previous:

**Bias-only ALS** — Learns per-user and per-item bias terms; no latent factors.

**ALS with latent factors** — Adds user embeddings $\mathbf{u}_m$ and item embeddings $\mathbf{v}_n$, solved via closed-form ALS updates.

**Feature-regularised ALS** — Regularises item embeddings toward a weighted average of learned feature embeddings (genres, decade), enabling generalisation to items with few ratings.

Full derivations of all closed-form update equations are in [`derivations.ipynb`](derivations.ipynb).

---

## Application

A live demo of the recommender is deployed at **[ge-movielens-recommender.streamlit.app](https://ge-movielens-recommender.streamlit.app)**.

The application is a Streamlit web app called **CineMatch**. Given a movie title, it retrieves the ten most similar films by computing cosine similarities over the learned item embeddings from the trained ALS model. Movie posters and TMDB links are fetched at runtime via the TMDB API.

**Structure:**

- `app.py` — Streamlit front-end; handles page layout, search, and card rendering.
- `recommender.py` — Loads the saved model artefacts (`all_data1.npz`) and exposes the `recommend(title, k)` function, which returns the top- $k$ most similar items by embedding dot product.
- `utils/tmdb.py` — Fetches poster URLs from the TMDB API using the MovieLens–TMDB ID mapping in `data/links.csv`.
- `data/all_data1.npz` — A version of the trained model artefacts packaged for the application (a filtered subset of the full `model/all_data.npz` for version control).

**Running locally:**

```bash
cd application
pip install -r requirements.txt
streamlit run src/app.py
```

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
