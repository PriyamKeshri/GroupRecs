"""
Matrix Factorization recommender (Funk SVD style) implemented from scratch
using Stochastic Gradient Descent.

Why build this instead of using a library?
- Demonstrates real understanding of how collaborative filtering works
- Full control to expose per-user, per-movie latent vectors (useful later
  for explainability / debugging)
- No heavyweight dependencies (just numpy)

Model:
    predicted_rating(u, i) = global_mean + user_bias[u] + item_bias[i]
                              + user_factors[u] . item_factors[i]

Trained by minimizing squared error on observed ratings with L2 regularization.
"""

import numpy as np


class MatrixFactorizationRecommender:
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02,
                 n_epochs=30, random_state=42):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Learned parameters (populated in fit())
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0

        # ID <-> matrix index mappings
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}

    def fit(self, ratings_df, verbose=True):
        """
        ratings_df: pandas DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        rng = np.random.RandomState(self.random_state)

        user_ids = ratings_df["user_id"].unique()
        item_ids = ratings_df["item_id"].unique()

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

        n_users = len(user_ids)
        n_items = len(item_ids)

        self.global_mean = ratings_df["rating"].mean()
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Pre-map to indices once for speed
        u_idx = ratings_df["user_id"].map(self.user_id_to_idx).to_numpy()
        i_idx = ratings_df["item_id"].map(self.item_id_to_idx).to_numpy()
        r_val = ratings_df["rating"].to_numpy(dtype=float)

        n_samples = len(r_val)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            sq_error_sum = 0.0

            for k in order:
                u, i, r = u_idx[k], i_idx[k], r_val[k]

                pred = (self.global_mean + self.user_bias[u] + self.item_bias[i]
                        + np.dot(self.user_factors[u], self.item_factors[i]))
                err = r - pred
                sq_error_sum += err ** 2

                # SGD updates
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                uf = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * uf)
                self.item_factors[i] += self.lr * (err * uf - self.reg * self.item_factors[i])

            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
                rmse = np.sqrt(sq_error_sum / n_samples)
                print(f"  epoch {epoch + 1:>3}/{self.n_epochs}  train RMSE: {rmse:.4f}")

        return self

    def predict(self, user_id, item_id):
        """Predict rating for a known user_id and item_id. Returns float, clipped to [0.5, 5]."""
        if user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
            return self.global_mean  # fallback for unseen pairs

        u = self.user_id_to_idx[user_id]
        i = self.item_id_to_idx[item_id]
        pred = (self.global_mean + self.user_bias[u] + self.item_bias[i]
                + np.dot(self.user_factors[u], self.item_factors[i]))
        return float(np.clip(pred, 0.5, 5.0))

    def predict_for_user(self, user_id, item_ids=None):
        """Return {item_id: predicted_rating} for a known user across given items
        (defaults to all items the model was trained on)."""
        if item_ids is None:
            item_ids = list(self.item_id_to_idx.keys())
        return {iid: self.predict(user_id, iid) for iid in item_ids}

    def is_known_user(self, user_id):
        return user_id in self.user_id_to_idx
