"""
Data loading utilities.

Four sources are supported, in increasing order of size/recency:
1. generate_synthetic_data() -- lets us build and test the whole pipeline
   right now without needing any real dataset downloaded yet.
2. load_movielens_100k(path) -- loader for MovieLens 100k (u.data + u.item +
   u.user), ~1,700 movies / 943 users / 100k ratings, released 2000-ish.
   Download from https://files.grouplens.org/datasets/movielens/ml-100k.zip
3. load_movielens_1m(path) -- loader for MovieLens 1M (ratings.dat +
   movies.dat + users.dat), ~3,900 movies / 6,040 users / 1M ratings.
   Different on-disk format (`::`-separated, no header) but normalized to
   the exact same shape. Download from
   https://files.grouplens.org/datasets/movielens/ml-1m.zip
4. load_movielens_25m(path) -- loader for MovieLens 25M (movies.csv +
   ratings.csv), ~62,000 movies with titles up to 2019 / 25M ratings.
   GroupLens stopped collecting demographics after the 1M release, so this
   dataset has no age/gender/occupation -- synthesized instead (see the
   function docstring). Ratings are subsampled for trainability (see
   max_ratings). Download from
   https://files.grouplens.org/datasets/movielens/ml-25m.zip

All four return a 4-tuple: (ratings_df, movies_df, genre_columns,
users_df). users_df (user_id, age, gender, occupation) feeds the
demographic cold-start in src/demographics.py -- see that module for why
it's a useful alternative to genre-based cold start.

A fifth loader, load_bollywood_catalog(), is different in kind: it's a
supplementary movie catalog (title + genres only, no ratings/users) merged
on top of whichever base dataset above is active -- see
src/model_cache.py:load_or_train() for how the merge works and why
collaborative-filtering users only get an unpersonalized prediction for
these titles (there's no per-user rating data for them anywhere).
"""

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
    "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller",
]

# Same occupation vocabulary MovieLens 100k uses (u.occupation), reused for
# the synthetic generator too so both data sources share one vocabulary and
# the API/UI don't need to special-case which is active.
OCCUPATIONS = [
    "administrator", "artist", "doctor", "educator", "engineer",
    "entertainment", "executive", "healthcare", "homemaker", "lawyer",
    "librarian", "marketing", "none", "other", "programmer", "retired",
    "salesman", "scientist", "student", "technician", "writer",
]

# MovieLens 1M encodes occupation as an integer 0-20 (see its README) --
# different vocabulary than the 100k dataset's free-text occupations, but
# nothing downstream hardcodes either list (the API reads whatever's
# actually present in users_df), so this is safe to differ.
ML_1M_OCCUPATIONS = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}


def generate_synthetic_data(n_users=60, n_movies=40, seed=7):
    """
    Builds a small synthetic MovieLens-like dataset:
    - movies_df: item_id, title, one-hot genre columns
    - ratings_df: user_id, item_id, rating (1-5)
    - users_df: user_id, age, gender, occupation

    Users are generated with a hidden genre preference so the data has real
    structure for the model to learn (not pure noise) -- this makes the demo
    predictions meaningful instead of arbitrary. Demographics are independent
    random draws (not correlated with the hidden genre preference), which is
    fine -- they only need to exist so the demographic cold-start path has
    something to precompute group averages over.
    """
    rng = np.random.RandomState(seed)

    # --- Movies: each gets 1-2 genres ---
    movie_rows = []
    for movie_id in range(1, n_movies + 1):
        n_genres = rng.choice([1, 2], p=[0.6, 0.4])
        genres = rng.choice(GENRES, size=n_genres, replace=False)
        row = {"item_id": movie_id, "title": f"Movie {movie_id:02d} ({'/'.join(genres)})"}
        for g in GENRES:
            row[g] = 1 if g in genres else 0
        movie_rows.append(row)
    movies_df = pd.DataFrame(movie_rows)

    # --- Users: each has a hidden preference for 2-3 genres, plus demographics ---
    user_rows = []
    ratings_rows = []
    for user_id in range(1, n_users + 1):
        n_pref = rng.choice([2, 3])
        preferred_genres = set(rng.choice(GENRES, size=n_pref, replace=False))

        user_rows.append({
            "user_id": user_id,
            "age": int(rng.randint(15, 70)),
            "gender": rng.choice(["M", "F"]),
            "occupation": rng.choice(OCCUPATIONS),
        })

        # Each user rates a random subset of movies (simulate sparsity)
        n_rated = rng.randint(8, 20)
        rated_movies = rng.choice(movies_df["item_id"], size=n_rated, replace=False)

        for movie_id in rated_movies:
            movie_genres = set(
                g for g in GENRES if movies_df.loc[movies_df.item_id == movie_id, g].values[0] == 1
            )
            overlap = len(preferred_genres & movie_genres)
            # Base rating driven by genre match + noise
            base = 2.0 + overlap * 1.3
            rating = np.clip(base + rng.normal(0, 0.6), 1, 5)
            ratings_rows.append({"user_id": user_id, "item_id": movie_id, "rating": round(rating * 2) / 2})

    ratings_df = pd.DataFrame(ratings_rows)
    users_df = pd.DataFrame(user_rows)
    return ratings_df, movies_df, GENRES, users_df


def load_movielens_100k(data_dir):
    """
    Loads the real MovieLens 100k dataset.
    Expects `u.data` (ratings), `u.item` (movie metadata) and `u.user`
    (demographics) inside data_dir.
    Download from: https://files.grouplens.org/datasets/movielens/ml-100k.zip

    Returns: (ratings_df, movies_df, genre_columns, users_df) in the same
    shape as generate_synthetic_data(), so the rest of the pipeline doesn't
    change.
    """
    ratings_df = pd.read_csv(
        f"{data_dir}/u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )[["user_id", "item_id", "rating"]]

    item_cols = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    movies_df = pd.read_csv(
        f"{data_dir}/u.item", sep="|", encoding="latin-1", names=item_cols
    )

    genre_columns = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    users_df = pd.read_csv(
        f"{data_dir}/u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )[["user_id", "age", "gender", "occupation"]]

    return ratings_df, movies_df, genre_columns, users_df


def load_movielens_1m(data_dir):
    """
    Loads the real MovieLens 1M dataset.
    Expects `ratings.dat`, `movies.dat` and `users.dat` inside data_dir
    (`::`-separated, no header -- a different format than the 100k files).
    Download from: https://files.grouplens.org/datasets/movielens/ml-1m.zip

    Returns: (ratings_df, movies_df, genre_columns, users_df) in the same
    shape as generate_synthetic_data() / load_movielens_100k().
    """
    ratings_df = pd.read_csv(
        f"{data_dir}/ratings.dat", sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )[["user_id", "item_id", "rating"]]

    raw_movies = pd.read_csv(
        f"{data_dir}/movies.dat", sep="::", engine="python",
        names=["item_id", "title", "genres"], encoding="latin-1",
    )

    # genres arrive as "Animation|Children's|Comedy" -- explode into the
    # same one-hot column shape the rest of the pipeline expects.
    genre_columns = sorted({g for genre_list in raw_movies["genres"] for g in genre_list.split("|")})
    movies_df = raw_movies[["item_id", "title"]].copy()
    for genre in genre_columns:
        movies_df[genre] = raw_movies["genres"].apply(lambda gs, genre=genre: int(genre in gs.split("|")))

    raw_users = pd.read_csv(
        f"{data_dir}/users.dat", sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    # age is already a MovieLens age-bucket code (1, 18, 25, 35, 45, 50, 56)
    # that lines up exactly with src/demographics.py's own bucket edges, and
    # occupation is an integer code -- both normalized to the same shape as
    # the other two loaders (raw age int, free-text occupation string).
    users_df = pd.DataFrame({
        "user_id": raw_users["user_id"],
        "age": raw_users["age"],
        "gender": raw_users["gender"],
        "occupation": raw_users["occupation"].map(ML_1M_OCCUPATIONS),
    })

    return ratings_df, movies_df, genre_columns, users_df


def load_movielens_25m(data_dir, n_users=8000, seed=42):
    """
    Loads the real MovieLens 25M dataset -- the newest, largest catalog
    (movie titles up to 2019).
    Expects `movies.csv` and `ratings.csv` inside data_dir (standard CSV,
    header row). Download from:
    https://files.grouplens.org/datasets/movielens/ml-25m.zip

    Two differences from the other loaders, both because of what this
    dataset actually contains:
    - `ratings.csv` has 25M rows from 162,541 users; training the
      from-scratch pure-Python SGD recommender on all of them would take
      the better part of an hour, and a 162k-entry user list isn't
      browsable in the UI anyway. So `n_users` users are sampled and ALL of
      their ratings kept (not a flat row sample) -- this keeps per-user
      history dense (~150 ratings/user, close to real usage) rather than
      thinning every user down to a couple of ratings each. Collaborative
      filtering coverage (how many distinct movies end up with enough
      ratings to predict against) grows with n_users but with diminishing
      returns -- popular movies get covered almost immediately, the long
      tail needs a lot more users to reach. Only 59,047 of the ~62,423
      catalog titles have *any* rating in the full dataset, so that's the
      hard ceiling regardless of n_users:
          n_users=2,500  -> ~390k ratings,   ~18k movies covered (~30%), ~40s train
          n_users=8,000  -> ~1.2M ratings,   ~25k movies covered (~43%), ~110s train  (default)
          n_users=20,000 -> ~3.0M ratings,   ~32k movies covered (~54%), ~280s train
      The full movie catalog (all ~62k titles + genres) is kept regardless
      of n_users -- genre-based cold start and the movie listing aren't
      affected by the ratings subsample, only collaborative filtering
      coverage is.
    - GroupLens stopped collecting age/gender/occupation after the 1M
      release, so this dataset has no real demographics. users_df is
      synthesized (seeded, so reproducible) for every sampled user_id --
      demographic cold-start still works mechanically, it's just averaging
      over synthetic peers rather than real ones. Callers should treat this
      dataset's demographics as "structurally present, not a real signal."

    Returns: (ratings_df, movies_df, genre_columns, users_df) in the same
    shape as the other loaders.
    """
    raw_movies = pd.read_csv(f"{data_dir}/movies.csv")  # movieId, title, genres

    NO_GENRES = "(no genres listed)"
    genre_lists = raw_movies["genres"].apply(lambda gs: [] if gs == NO_GENRES else gs.split("|"))
    genre_columns = sorted({g for genres in genre_lists for g in genres})

    movies_df = raw_movies.rename(columns={"movieId": "item_id"})[["item_id", "title"]].copy()
    for genre in genre_columns:
        movies_df[genre] = genre_lists.apply(lambda genres, genre=genre: int(genre in genres))

    all_ratings = pd.read_csv(
        f"{data_dir}/ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    ).rename(columns={"userId": "user_id", "movieId": "item_id"})

    rng = np.random.RandomState(seed)
    all_user_ids = all_ratings["user_id"].unique()
    sampled_user_ids = rng.choice(all_user_ids, size=min(n_users, len(all_user_ids)), replace=False)
    ratings_df = all_ratings[all_ratings["user_id"].isin(sampled_user_ids)].reset_index(drop=True)

    users_df = pd.DataFrame({
        "user_id": sampled_user_ids,
        "age": rng.randint(15, 70, size=len(sampled_user_ids)),
        "gender": rng.choice(["M", "F"], size=len(sampled_user_ids)),
        "occupation": rng.choice(OCCUPATIONS, size=len(sampled_user_ids)),
    })

    return ratings_df, movies_df, genre_columns, users_df


# Item IDs for the Bollywood catalog are offset well above any MovieLens
# item_id (which top out in the low hundreds of thousands even for ml-25m),
# so merging the two catalogs can never collide on item_id.
BOLLYWOOD_ITEM_ID_OFFSET = 900_000_000


def load_bollywood_catalog(data_dir):
    """
    Loads a supplementary Bollywood movie catalog -- title + genres only,
    no ratings or users, because no per-user rating data exists for these
    movies anywhere (every Bollywood dataset we could find is IMDb/Wikipedia
    metadata, not a MovieLens-style ratings log). Meant to be merged on top
    of whichever base dataset is active (see model_cache.py:load_or_train),
    extending the genre-based cold-start catalog with real Hindi-cinema
    titles. Collaborative-filtering users get an unpersonalized (global
    mean) prediction for these titles instead of a tailored one, same as
    for any other movie they have no rating signal on.

    Expects a CSV at `{data_dir}/movies.csv` with columns movie_id,
    movie_name, year, genre (comma-separated) -- other columns (overview,
    director, cast) are ignored. Download from:
    https://github.com/devensinghbhagtani/Bollywood-Movie-Dataset

    Returns: (movies_df, genre_columns) -- note the 2-tuple, not the
    4-tuple the other loaders return, since there's no ratings_df/users_df.
    """
    raw = pd.read_csv(f"{data_dir}/movies.csv")

    genre_lists = raw["genre"].fillna("").apply(
        lambda gs: [g.strip() for g in gs.split(",") if g.strip()]
    )
    genre_columns = sorted({g for genres in genre_lists for g in genres})

    names = raw["movie_name"].astype(str).str.strip()
    years = pd.to_numeric(raw["year"], errors="coerce")
    titled = names + " (" + years.astype("Int64").astype(str) + ")"
    titles = titled.where(years.notna(), names)

    movies_df = pd.DataFrame({
        "item_id": BOLLYWOOD_ITEM_ID_OFFSET + np.arange(len(raw)),
        "title": titles,
    })
    for genre in genre_columns:
        movies_df[genre] = genre_lists.apply(lambda genres, genre=genre: int(genre in genres))

    return movies_df, genre_columns
