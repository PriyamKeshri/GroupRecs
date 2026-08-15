"""
Group aggregation strategies.

Given predicted ratings from each group member for each candidate movie,
this module combines them into a single ranked list for the group using
different social-choice strategies. This is the genuinely distinctive part
of the project -- most tutorials only ever recommend for a single user.

Input shape expected by every strategy function:
    member_predictions: dict[user_label] -> dict[item_id] -> predicted_rating
    e.g. {
        "Alice": {1: 4.2, 2: 3.1, ...},
        "Bob":   {1: 2.0, 2: 4.8, ...},
    }

Output: dict[item_id] -> aggregated_score, ready to sort descending.
"""

import numpy as np


def _stack_ratings(member_predictions, item_ids):
    """Build a (n_members, n_items) matrix of predicted ratings, aligned to item_ids."""
    members = list(member_predictions.keys())
    matrix = np.zeros((len(members), len(item_ids)))
    for m_idx, member in enumerate(members):
        preds = member_predictions[member]
        for i_idx, iid in enumerate(item_ids):
            matrix[m_idx, i_idx] = preds.get(iid, 0.0)
    return members, matrix


def average_aggregation(member_predictions, item_ids):
    """Group score = mean predicted rating across members.
    Optimizes for overall group satisfaction; can let one person's strong
    dislike get outvoted by the others."""
    _, matrix = _stack_ratings(member_predictions, item_ids)
    scores = matrix.mean(axis=0)
    return dict(zip(item_ids, scores))


def least_misery_aggregation(member_predictions, item_ids):
    """Group score = minimum predicted rating across members.
    Optimizes so that no single person is stuck with a movie they hate --
    classic 'least misery' strategy from group recommender literature."""
    _, matrix = _stack_ratings(member_predictions, item_ids)
    scores = matrix.min(axis=0)
    return dict(zip(item_ids, scores))


def most_pleasure_aggregation(member_predictions, item_ids):
    """Group score = maximum predicted rating across members.
    Rarely used alone (ignores everyone else), but useful as a contrast
    case to show what 'only the most enthusiastic person matters' looks like."""
    _, matrix = _stack_ratings(member_predictions, item_ids)
    scores = matrix.max(axis=0)
    return dict(zip(item_ids, scores))


def fairness_aware_aggregation(member_predictions, item_ids, disagreement_penalty=0.5):
    """Group score = average rating, penalized by how much the group disagrees
    (standard deviation across members). A movie that's a 5 for one person and
    a 1 for another gets penalized even though its average might look fine --
    avoiding divisive picks in favor of ones everyone is reasonably happy with."""
    _, matrix = _stack_ratings(member_predictions, item_ids)
    avg = matrix.mean(axis=0)
    disagreement = matrix.std(axis=0)
    scores = avg - disagreement_penalty * disagreement
    return dict(zip(item_ids, scores))


STRATEGIES = {
    "average": average_aggregation,
    "least_misery": least_misery_aggregation,
    "most_pleasure": most_pleasure_aggregation,
    "fairness_aware": fairness_aware_aggregation,
}


def recommend_for_group(member_predictions, top_n=10, strategy="average", **kwargs):
    """
    member_predictions: dict[user_label] -> dict[item_id] -> predicted_rating
    strategy: one of STRATEGIES keys
    Returns: list of (item_id, score) sorted descending, length top_n
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES.keys())}")

    # Use the intersection of items every member has a prediction for
    item_id_sets = [set(preds.keys()) for preds in member_predictions.values()]
    common_items = list(set.intersection(*item_id_sets)) if item_id_sets else []

    scores = STRATEGIES[strategy](member_predictions, common_items, **kwargs)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def compare_all_strategies(member_predictions, top_n=5):
    """Convenience function: run every strategy and return results side by side.
    Great for the demo/portfolio piece -- show how the recommended list SHIFTS
    depending on which notion of 'fairness' you optimize for."""
    return {
        name: recommend_for_group(member_predictions, top_n=top_n, strategy=name)
        for name in STRATEGIES
    }
