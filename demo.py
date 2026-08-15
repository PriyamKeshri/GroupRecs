"""
End-to-end demo: trains (or loads a cached trained model for) the
biggest/newest real MovieLens dataset available on disk (25M, then 1M,
then 100k, falling back to synthetic data if none has been downloaded),
simulates a group of 5 friends planning movie night covering every
prediction path this project supports -- 2 with rating history
(collaborative filtering), 1 brand-new guest via genre cold-start, 1 via
demographic cold-start, and 1 blending both -- and shows how the
recommended movie CHANGES depending on the aggregation strategy.

Run: python3 demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.cold_start import ColdStartProfiler
from src.demographics import DemographicProfiler, blend_predictions
from src.group_aggregator import compare_all_strategies, recommend_for_group
from src.model_cache import load_or_train


def main():
    print("=" * 70)
    print("STEP 1: Loading data + model (cached if a matching one exists)")
    print("=" * 70)
    result = load_or_train()
    ratings_df = result["ratings_df"]
    movies_df = result["movies_df"]
    genre_cols = result["genre_columns"]
    users_df = result["users_df"]
    model = result["model"]
    print(f"  dataset: {result['dataset_name']}"
          + ("  (from cache -- see .cache/)" if result["from_cache"] else "  (trained fresh)"))
    print(f"  {len(ratings_df)} ratings across {ratings_df.user_id.nunique()} users "
          f"and {ratings_df.item_id.nunique()} movies")
    if result["demographics_source"] == "synthetic":
        print("  Note: this dataset has no real demographics -- age/gender/occupation are synthesized.")
    print()

    print("=" * 70)
    print("STEP 2: Simulating a movie-night group")
    print("=" * 70)
    all_item_ids = list(model.item_id_to_idx.keys())

    # Two existing users with rating history -- pure collaborative filtering.
    # Picked dynamically (not hardcoded to user_id 1/2) because the 25M
    # loader samples a random subset of users -- ids 1 and 2 from the full
    # 162k-user pool usually aren't even in that subset.
    known_user_ids = list(model.user_id_to_idx.keys())
    alice_preds = model.predict_for_user(user_id=known_user_ids[0], item_ids=all_item_ids)
    bob_preds = model.predict_for_user(user_id=known_user_ids[1], item_ids=all_item_ids)

    # A brand-new guest with NO rating history -- cold start via genre picks
    genre_profiler = ColdStartProfiler(movies_df, genre_cols)
    chloe_preds = genre_profiler.predict_for_guest(
        liked_genres=["Comedy", "Romance"], item_ids=all_item_ids
    )

    # Another brand-new guest -- cold start via demographics instead: no
    # genre picks needed, just "people like you" (same age bucket / gender /
    # occupation) averaged from the training ratings.
    demo_profiler = DemographicProfiler(users_df, ratings_df)
    dev_preds = demo_profiler.predict_for_guest(
        age=25, gender="M", occupation="programmer", item_ids=all_item_ids
    )

    # A third guest gives BOTH signals -- genre picks and demographics --
    # blended into one prediction. More signal than either alone.
    eve_genre_preds = genre_profiler.predict_for_guest(
        liked_genres=["Horror", "Thriller"], item_ids=all_item_ids
    )
    # "writer" (unlike e.g. "doctor") is spelled the same in both the 100k
    # and 1M occupation vocabularies, so this line works regardless of
    # which dataset load_data() picked.
    eve_demo_preds = demo_profiler.predict_for_guest(
        age=45, gender="F", occupation="writer", item_ids=all_item_ids
    )
    eve_preds = blend_predictions([(eve_genre_preds, 1.0), (eve_demo_preds, 1.0)])

    group = {
        "Alice (history)": alice_preds,
        "Bob (history)": bob_preds,
        "Chloe (cold-start: genres = Comedy/Romance)": chloe_preds,
        "Dev (cold-start: demographics = 25/M/programmer)": dev_preds,
        "Eve (cold-start: genres + demographics blended)": eve_preds,
    }

    title_lookup = movies_df.set_index("item_id")["title"].to_dict()

    print("  Group: Alice & Bob (existing users, collaborative filtering)")
    print("       + Chloe (cold-start via genres)")
    print("       + Dev (cold-start via demographics)")
    print("       + Eve (cold-start via genres + demographics, blended)\n")

    print("=" * 70)
    print("STEP 3: Comparing aggregation strategies")
    print("=" * 70)
    results = compare_all_strategies(group, top_n=3)

    for strategy_name, ranked_movies in results.items():
        print(f"\n--- Strategy: {strategy_name} ---")
        for item_id, score in ranked_movies:
            title = title_lookup.get(item_id, f"Movie {item_id}")
            # show each member's individual predicted rating for transparency
            per_member = ", ".join(
                f"{name.split()[0]}={preds[item_id]:.1f}" for name, preds in group.items()
            )
            print(f"  {title:45s}  group_score={score:.2f}   ({per_member})")

    print("\n" + "=" * 70)
    print("Notice how 'least_misery' avoids movies that any one person scores")
    print("low, while 'average' can pick something divisive if it has a high")
    print("mean. This contrast is the core talking point of the project.")
    print("=" * 70)


if __name__ == "__main__":
    main()
