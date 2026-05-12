import re

# Extract year from movie title
def extract_year(title):
    if m := re.search(r'\((\d{4})\)', title):
        return int(m.group(1))
    return None

def normalize_title(title):
    """Remove year in parentheses and extra whitespace, convert to lowercase."""
    return re.sub(r"\s*\(\d{4}\)", "", title).strip().lower()




import numpy as np

# -------------------- User Updates --------------------
def update_user_bias(user, mu, data_by_user_train, lam, gamma, tau,
                     item_bias, user_bias, item_embedding, user_embedding):
    """Update user bias given training data."""
    start_ptr = data_by_user_train.indptr[user]
    end_ptr = data_by_user_train.indptr[user + 1]
    items = data_by_user_train.indices[start_ptr:end_ptr]
    ratings = data_by_user_train.data[start_ptr:end_ptr]

    if len(ratings) == 0:
        return 0.0

    users = np.full_like(items, user)
    uv = np.sum(user_embedding[users] * item_embedding[items], axis=1)
    bias = np.sum(ratings - uv - item_bias[items] - mu)
    return lam * bias / (lam * len(ratings) + gamma)


def update_user_embedding(user, mu, data_by_user_train, lam, gamma, tau,
                          item_bias, user_bias, item_embedding, user_embedding):
    """Update user embedding vector."""
    start_ptr = data_by_user_train.indptr[user]
    end_ptr = data_by_user_train.indptr[user + 1]
    items = data_by_user_train.indices[start_ptr:end_ptr]
    ratings = data_by_user_train.data[start_ptr:end_ptr]

    k = item_embedding.shape[1]
    if len(ratings) == 0:
        return np.zeros(k)

    users = np.full_like(items, user)
    sum1 = item_embedding[items].T @ item_embedding[items]
    sum2 = item_embedding[items].T @ (ratings - user_bias[users] - item_bias[items] - mu)
    p1 = lam * sum1 + tau * np.eye(k)
    p2 = lam * sum2
    return np.linalg.solve(p1, p2)



# -------------------- Item Updates --------------------
def update_item_bias(item, mu, data_by_movie_train, lam, gamma, tau,
                     item_bias, user_bias, item_embedding, user_embedding):
    """Update item bias given training data."""
    start_ptr = data_by_movie_train.indptr[item]
    end_ptr = data_by_movie_train.indptr[item + 1]
    users = data_by_movie_train.indices[start_ptr:end_ptr]
    ratings = data_by_movie_train.data[start_ptr:end_ptr]

    if len(ratings) == 0:
        return 0.0

    items = np.full_like(users, item)
    uv = np.sum(user_embedding[users] * item_embedding[items], axis=1)
    bias = np.sum(ratings - uv - user_bias[users] - mu)
    return lam * bias / (lam * len(ratings) + gamma)


def update_item_embedding(item, mu, data_by_movie_train, lam, gamma, tau,
                          item_bias, user_bias, item_embedding, user_embedding,
                          item_features, feature_embedding):
    """Update item embedding vector including feature contribution."""
    start_ptr = data_by_movie_train.indptr[item]
    end_ptr = data_by_movie_train.indptr[item + 1]
    users = data_by_movie_train.indices[start_ptr:end_ptr]
    ratings = data_by_movie_train.data[start_ptr:end_ptr]

    k = user_embedding.shape[1]
    if len(ratings) == 0:
        return np.zeros(k)

    items = np.full_like(users, item)
    sum1 = user_embedding[users].T @ user_embedding[users]
    sum2 = user_embedding[users].T @ (ratings - item_bias[items] - user_bias[users] - mu)
    p1 = lam * sum1 + tau * np.eye(k)
    p2 = lam * sum2

    # Feature contribution
    feats = item_features[item]
    F_n = np.sum(feats)
    s_n = np.zeros(k) if F_n == 0 else (feats @ feature_embedding) / np.sqrt(F_n)

    p2 = lam * sum2 + tau * s_n
    return np.linalg.solve(p1, p2)



# -------------------- Feature Updates --------------------
def update_feature_embedding(item_features, item_embedding, tau, lam):
    """Update feature embeddings based on item embeddings."""
    n_items, n_features = item_features.shape
    k = item_embedding.shape[1]

    # Compute scaling for items with multiple features
    F_n = item_features.sum(axis=1, keepdims=True)
    scaling = np.where(F_n > 0, 1.0 / np.sqrt(F_n), 0.0)

    # Weighted features
    weighted_features = item_features * scaling

    # Linear system: A F = B
    A = lam * (weighted_features.T @ weighted_features) + tau * np.eye(n_features)
    B = lam * (weighted_features.T @ item_embedding)

    # Solve for feature embedding
    feature_embedding = np.linalg.solve(A, B)
    return feature_embedding
