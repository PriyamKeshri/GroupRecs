"""
Cold-start handler for the group recommender.

Problem this solves: in a real "movie night" group, not everyone has a
rating history in your dataset (a friend who's never used the app, a new
account, etc). Rather than excluding them or asking them to rate 20 movies
before they can participate, we ask for something much lighter: pick 3-5
genres you enjoy. We build a lightweight genre-preference vector and score
candidate movies by similarity -- giving a usable predicted rating with
almost zero friction.

This score is rescaled to the same 0.5-5.0 range as the matrix factorization
model so the two can be mixed fairly inside the group aggregator.
"""

import numpy as np


class ColdStartProfiler:
    def __init__(self, movies_df, genre_columns):
        """
        movies_df: DataFrame with one row per movie, including an 'item_id'
                   column and one-hot genre columns.
        genre_columns: list of column names in movies_df that are one-hot
                       genre flags, e.g. ['Action', 'Comedy', 'Drama', ...]
        """
        self.movies_df = movies_df.set_index("item_id")
        self.genre_columns = genre_columns

    def build_profile(self, liked_genres):
        """liked_genres: list of genre names the guest selected. Returns a
        normalized preference vector aligned to self.genre_columns."""
        vec = np.array([1.0 if g in liked_genres else 0.0 for g in self.genre_columns])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def predict_for_guest(self, liked_genres, item_ids=None):
        """Returns {item_id: predicted_rating} for a guest user based purely
        on genre overlap, rescaled to a 0.5-5.0 scale.

        Vectorized over all candidate movies at once (matrix @ vector, one
        norm computation) rather than a per-movie Python loop with a
        pandas .loc[] lookup each iteration -- the latter is fine at
        ml-100k/1M scale (thousands of movies) but takes several seconds
        at ml-25m scale (62k+ movies), long enough to blow through the
        UI's request timeout on every single cold-start guest added."""
        profile = self.build_profile(liked_genres)

        if item_ids is None:
            genre_frame = self.movies_df[self.genre_columns]
        else:
            valid_ids = [iid for iid in item_ids if iid in self.movies_df.index]
            genre_frame = self.movies_df.loc[valid_ids, self.genre_columns]

        genre_matrix = genre_frame.to_numpy(dtype=float)
        norms = np.linalg.norm(genre_matrix, axis=1)
        dots = genre_matrix @ profile
        # cos_sim is in [0, 1] since both vectors are non-negative one-hot-ish;
        # movies with an all-zero genre vector (norm 0) score 0 like before.
        cos_sim = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        scores = 0.5 + cos_sim * 4.5  # rescale to [0.5, 5.0]

        return dict(zip(genre_frame.index.tolist(), scores.tolist()))
