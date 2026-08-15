"""
Trains-once, caches-to-disk wrapper around dataset loading + model fitting.

Both demo.py and src/api.py need the same thing: "load the best available
MovieLens dataset and a matrix-factorization model trained on it." Training
takes anywhere from ~10s (100k) to ~2min (25m) with the from-scratch
pure-Python SGD trainer in recommender.py -- a real cost to pay on every
process restart, especially `uvicorn --reload`, which restarts on every
file save. This module fingerprints the (dataset, model hyperparameters)
combination and caches the fitted model + its dataframes to disk, so a
restart with nothing relevant changed is a fast pickle load instead of a
full retrain.

Cache invalidates automatically when the dataset file's mtime/size changes
(e.g. you re-download the dataset) or the model hyperparameters change. It
does NOT automatically invalidate if you change a loader's *sampling*
logic (e.g. load_movielens_25m's n_users default) without also changing
these params -- bump CACHE_VERSION below if you do that, or just delete
the .cache/ directory.
"""

import hashlib
import json
import os
import pickle

import pandas as pd

from .data_loader import (
    generate_synthetic_data,
    load_bollywood_catalog,
    load_movielens_100k,
    load_movielens_1m,
    load_movielens_25m,
)
from .recommender import MatrixFactorizationRecommender

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
CACHE_VERSION = "v1"

BOLLYWOOD_DIR = os.path.join(DATA_ROOT, "bollywood")
BOLLYWOOD_MARKER = os.path.join(BOLLYWOOD_DIR, "movies.csv")

DEFAULT_MODEL_PARAMS = {"n_factors": 20, "n_epochs": 20, "learning_rate": 0.01}

# (dataset_name, dir_name, marker filename inside that dir, loader_fn) in
# priority order -- first one whose marker file exists on disk wins.
DATASET_SOURCES = [
    ("movielens-25m", "ml-25m", "ratings.csv", load_movielens_25m),
    ("movielens-1m", "ml-1m", "ratings.dat", load_movielens_1m),
    ("movielens-100k", "ml-100k", "u.data", load_movielens_100k),
]

DATASETS_WITH_REAL_DEMOGRAPHICS = {"movielens-1m", "movielens-100k"}


def _pick_dataset():
    """Returns (dataset_name, loader_fn, data_dir, marker_path) for the
    biggest/newest dataset actually present on disk, or
    ("synthetic", None, None, None) if none have been downloaded."""
    for dataset_name, dir_name, marker_file, loader_fn in DATASET_SOURCES:
        data_dir = os.path.join(DATA_ROOT, dir_name)
        marker_path = os.path.join(data_dir, marker_file)
        if os.path.exists(marker_path):
            return dataset_name, loader_fn, data_dir, marker_path
    return "synthetic", None, None, None


def _cache_path(dataset_name, marker_path, model_params):
    """A cache file name derived from everything that should invalidate it:
    dataset identity, the source file's mtime/size (catches re-downloads),
    the model hyperparameters, and a manual version bump for logic changes
    that don't show up in any of the above."""
    fingerprint = {"version": CACHE_VERSION, "dataset": dataset_name, "model_params": model_params}
    if marker_path:
        stat = os.stat(marker_path)
        fingerprint["source_mtime"] = stat.st_mtime
        fingerprint["source_size"] = stat.st_size
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{dataset_name}-{key}.pkl")


def _merge_bollywood_catalog(movies_df, genre_columns, verbose=True):
    """If data/bollywood/movies.csv is present, extends movies_df/
    genre_columns with it -- see data_loader.load_bollywood_catalog for
    what that does and doesn't give collaborative-filtering users. Applied
    fresh on every call (cache or no cache) rather than baked into the
    cached pickle -- it's a ~625KB CSV, re-parsing it is cheap, and this
    way updating the Bollywood catalog doesn't require reasoning about
    cache invalidation at all."""
    if not os.path.exists(BOLLYWOOD_MARKER):
        return movies_df, genre_columns

    bollywood_movies, bollywood_genres = load_bollywood_catalog(BOLLYWOOD_DIR)
    merged_genre_columns = sorted(set(genre_columns) | set(bollywood_genres))

    # Both frames need every column in the merged vocabulary before
    # concatenating, else a genre only one side has shows up as NaN
    # instead of 0 for the other side's rows.
    movies_df = movies_df.copy()
    for genre in merged_genre_columns:
        if genre not in movies_df.columns:
            movies_df[genre] = 0
        if genre not in bollywood_movies.columns:
            bollywood_movies[genre] = 0

    merged = pd.concat([movies_df, bollywood_movies], ignore_index=True, sort=False)
    if verbose:
        print(f"  Merged in {len(bollywood_movies)} Bollywood titles from data/bollywood/ "
              f"({len(merged)} movies total)")
    return merged, merged_genre_columns


def load_or_train(model_params=None, verbose=True):
    """
    Picks the best available dataset, then either loads a cached trained
    model for it or trains fresh (and writes the cache for next time).
    Also merges in the Bollywood catalog if data/bollywood/ is present.

    Returns a dict: ratings_df, movies_df, genre_columns, users_df,
    dataset_name, model, demographics_source, from_cache.
    """
    model_params = model_params or DEFAULT_MODEL_PARAMS
    dataset_name, loader_fn, data_dir, marker_path = _pick_dataset()
    cache_path = _cache_path(dataset_name, marker_path, model_params)
    demographics_source = "real" if dataset_name in DATASETS_WITH_REAL_DEMOGRAPHICS else "synthetic"

    if os.path.exists(cache_path):
        if verbose:
            print(f"  Using cached {dataset_name} model ({os.path.basename(cache_path)}) -- skipping retrain")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        cached["dataset_name"] = dataset_name
        cached["demographics_source"] = demographics_source
        cached["from_cache"] = True
        cached["movies_df"], cached["genre_columns"] = _merge_bollywood_catalog(
            cached["movies_df"], cached["genre_columns"], verbose=verbose
        )
        return cached

    if verbose:
        note = " (this trains fresh -- may take a while, see README > Dataset)" if dataset_name != "synthetic" else ""
        print(f"  No cache for {dataset_name} yet{note}")

    if loader_fn:
        ratings_df, movies_df, genre_columns, users_df = loader_fn(data_dir)
    else:
        ratings_df, movies_df, genre_columns, users_df = generate_synthetic_data()

    model = MatrixFactorizationRecommender(**model_params)
    model.fit(ratings_df, verbose=verbose)

    result = {
        "ratings_df": ratings_df,
        "movies_df": movies_df,
        "genre_columns": genre_columns,
        "users_df": users_df,
        "model": model,
        "dataset_name": dataset_name,
        "demographics_source": demographics_source,
        "from_cache": False,
    }

    if dataset_name != "synthetic":
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            to_cache = {k: v for k, v in result.items() if k not in ("dataset_name", "demographics_source", "from_cache")}
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(to_cache, f)
            os.replace(tmp_path, cache_path)  # atomic -- no half-written cache file if interrupted mid-write
            if verbose:
                print(f"  Cached trained model to {os.path.relpath(cache_path, PROJECT_ROOT)}")
        except OSError as exc:
            # Read-only / ephemeral filesystems (e.g. serverless platforms like
            # Vercel, whose filesystem is read-only outside /tmp) can't persist
            # a cache between invocations -- that just means every cold start
            # retrains, not a reason to crash the request.
            if verbose:
                print(f"  Could not write model cache ({exc}) -- continuing without it")

    result["movies_df"], result["genre_columns"] = _merge_bollywood_catalog(
        result["movies_df"], result["genre_columns"], verbose=verbose
    )
    return result
