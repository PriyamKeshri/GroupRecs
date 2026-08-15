"""
Demographic-based cold start.

Alternative (and complementary) signal to genre-based cold start
(see cold_start.py): instead of asking a brand-new guest what genres they
like, ask for a few demographic facts -- age, gender, occupation -- and
predict their rating for each movie as the average rating given by *similar
existing users*. This is classic demographic filtering: no latent factors,
no genre vectors, just group-by averages over the observed ratings ("people
like you enjoyed this").

Falls back through progressively coarser peer groups whenever the exact
group has too few ratings on a movie to trust (MIN_GROUP_SIZE), all the way
down to that movie's overall mean and finally the dataset's global mean.

Use standalone for a guest who only wants to share demographics, or blend
with genre-based cold start via `blend_predictions()` for a guest who
provides both -- more signal than either alone.
"""

import numpy as np

# MovieLens' own age-bucket convention (used on their site historically).
AGE_BUCKETS = [
    (0, 17, "Under 18"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 49, "45-49"),
    (50, 55, "50-55"),
    (56, 200, "56+"),
]


def bucket_age(age):
    """Map a raw age (int) to one of the buckets above."""
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return AGE_BUCKETS[-1][2]


class DemographicProfiler:
    """Precomputes group-average ratings per movie at four levels of
    granularity so a guest with zero rating history can get a prediction
    from people who share their age bucket / gender / occupation."""

    MIN_GROUP_SIZE = 3  # minimum #ratings before a group average is trusted

    def __init__(self, users_df, ratings_df):
        """
        users_df: DataFrame with columns [user_id, age, gender, occupation]
        ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        merged = ratings_df.merge(
            users_df[["user_id", "age", "gender", "occupation"]], on="user_id"
        )
        merged["age_bucket"] = merged["age"].apply(bucket_age)

        self.global_mean = float(merged["rating"].mean())
        # Every item that has at least one rating -- used as the default
        # candidate set. NOT used directly as a value source (see below):
        # a single 5-star vote shouldn't be able to plant an obscure movie
        # at the top of every guest's list.
        self._item_ids = merged["item_id"].unique().tolist()
        # Per-movie mean, but only trusted once enough people rated it --
        # same MIN_GROUP_SIZE bar as every other fallback level, so a movie
        # with 1-2 ratings falls through to the global mean instead.
        self._by_item = self._group_means(merged, ["item_id"])

        # Finest -> coarsest peer groups, each a {key_tuple: mean_rating} dict.
        self._by_bucket_gender_occ = self._group_means(
            merged, ["age_bucket", "gender", "occupation", "item_id"]
        )
        self._by_bucket_gender = self._group_means(merged, ["age_bucket", "gender", "item_id"])
        self._by_bucket = self._group_means(merged, ["age_bucket", "item_id"])

    @classmethod
    def _group_means(cls, df, keys):
        grouped = df.groupby(keys)["rating"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= cls.MIN_GROUP_SIZE]
        return grouped["mean"].to_dict()

    def predict_for_guest(self, age, gender, occupation, item_ids=None):
        """Returns {item_id: predicted_rating}, falling back through
        progressively coarser peer groups (then the movie's overall mean,
        then the global mean) whenever the finer group has too few ratings
        to trust. Ratings are never exactly 0.0, so a plain `or` chain is
        safe here."""
        bucket = bucket_age(age)
        if item_ids is None:
            item_ids = self._item_ids

        scores = {}
        for iid in item_ids:
            value = (
                self._by_bucket_gender_occ.get((bucket, gender, occupation, iid))
                or self._by_bucket_gender.get((bucket, gender, iid))
                or self._by_bucket.get((bucket, iid))
                or self._by_item.get(iid)
                or self.global_mean
            )
            scores[iid] = float(np.clip(value, 0.5, 5.0))
        return scores


def blend_predictions(pred_dicts_with_weights):
    """Weighted-average multiple {item_id: rating} dicts into one.

    pred_dicts_with_weights: iterable of (predictions_dict, weight) pairs,
    e.g. [(genre_preds, 1.0), (demographic_preds, 1.0)] for an equal-weight
    blend. An item missing from some sources is still handled correctly --
    its blended score is the weighted average of only the sources that
    have it, not diluted by the ones that don't.
    """
    totals: dict = {}
    weight_totals: dict = {}
    for preds, weight in pred_dicts_with_weights:
        for item_id, value in preds.items():
            totals[item_id] = totals.get(item_id, 0.0) + value * weight
            weight_totals[item_id] = weight_totals.get(item_id, 0.0) + weight
    return {item_id: totals[item_id] / weight_totals[item_id] for item_id in totals}
