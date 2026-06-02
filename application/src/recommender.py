import numpy as np

# Load model artefacts
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../data/all_data1.npz")

data = np.load(MODEL_PATH, allow_pickle=True)

item_embedding = data["item_embedding"]
item_index_to_title = data["item_index_to_title"].item()
title_to_item_index = data["title_to_item_index"].item()
item_index_to_movieid = data["item_index_to_movieid"].item()


def recommend(title, k=10):
    idx = title_to_item_index.get(title)

    if idx is None:
        return []

    query_vec = item_embedding[idx]
    sims = item_embedding @ query_vec

    top_idx = np.argsort(sims)[::-1]
    top_idx = top_idx[top_idx != idx][:k]

    return [
        (i, item_index_to_movieid[i], item_index_to_title[i], float(sims[i]))
        for i in top_idx
    ]