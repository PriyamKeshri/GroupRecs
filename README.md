# 🎬 GroupRecs — Fair Group Movie Recommendation System

> **Recommending a movie for one person is easy. Recommending one that a whole group can agree on is the real challenge.**

GroupRecs is a **group movie recommendation system** designed for movie nights with multiple people.

Unlike traditional recommendation systems that optimize recommendations for a single user, GroupRecs predicts what **each member of a group** is likely to enjoy and then combines those predictions using different **group decision-making strategies**.

The system supports users with existing rating histories as well as **brand-new users with no historical data**, using a lightweight genre-based cold-start profile.

---

## ✨ Features

- 🎯 Personalized movie rating prediction
- 🧠 Matrix Factorization implemented from scratch
- ⚡ Stochastic Gradient Descent (SGD) optimization
- 🆕 Cold-start recommendation for new users
- 🎭 Genre-based preference profiling
- 👥 Group-aware movie recommendation
- ⚖️ Multiple social-choice aggregation strategies
- 📊 Comparison of different group recommendation strategies
- 🧪 Synthetic MovieLens-like data generation
- 🎬 Support for the MovieLens 100K dataset
- 🧩 Modular recommendation pipeline using NumPy and Pandas

---

# 🧠 Problem Statement

Traditional recommendation systems usually answer:

> **"What movie should I recommend to this user?"**

However, movie nights typically involve multiple people.

For example:

```text
Alice → Loves Comedy & Romance
Bob   → Loves Action & Thriller
Chloe → Loves Drama
Dev   → Loves Horror & Thriller
```

A movie that Alice loves might be terrible for Dev.

Therefore, simply recommending the highest-rated movie for one person is not enough.

GroupRecs approaches this as a **group decision-making problem**.

The system follows this pipeline:

```text
Individual Preferences
          │
          ▼
┌──────────────────────┐
│ Individual Prediction│
│       Models         │
└──────────┬───────────┘
           │
           ▼
    Predicted Ratings
           │
           ▼
┌──────────────────────┐
│ Group Aggregation    │
│                      │
│ • Average            │
│ • Least Misery       │
│ • Most Pleasure      │
│ • Fairness-Aware     │
└──────────┬───────────┘
           │
           ▼
      Group Ranking
           │
           ▼
      🎬 Movie Night
```

---

# 🚀 How It Works

GroupRecs consists of three major stages:

1. **Individual Recommendation**
2. **Cold-Start Recommendation**
3. **Group Preference Aggregation**

## 1. 🎯 Individual Recommendation

For users with historical ratings, GroupRecs uses **Matrix Factorization** to learn latent representations of users and movies.

The predicted rating is modeled as:

```text
rating(u, i) =
    global_mean
    + user_bias
    + item_bias
    + user_factors · item_factors
```

The model is trained using **Stochastic Gradient Descent (SGD)** with L2 regularization.

The matrix factorization implementation is built **from scratch using NumPy**, rather than relying on a dedicated recommendation-system library.

## 2. 🆕 Cold-Start Recommendation

Traditional collaborative filtering struggles when a user has no rating history.

This is known as the **cold-start problem**.

GroupRecs handles new users using a lightweight **genre-based preference profiler**.

Instead of requiring a new user to rate many movies, the user can simply select genres they enjoy.

```text
User Preferences

✓ Comedy
✓ Romance
```

The system creates a genre preference vector and compares it with movie genre vectors using **cosine similarity**.

The similarity score is then converted into a predicted rating.

This allows completely new users to participate in group recommendations immediately.


# 📊 Group Strategy Comparison

| Strategy | Objective | Behavior |
|----------|-----------|-----------|
| `average` | Overall satisfaction | Balances everyone's predicted ratings |
| `least_misery` | Avoid strong dislike | Protects the least satisfied member |
| `most_pleasure` | Maximum enthusiasm | Favors the strongest individual preference |
| `fairness_aware` | Satisfaction + fairness | Penalizes large disagreements |

> **There is no universally "best" group recommendation strategy.**

Different strategies optimize different objectives.

---


# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/group-movie-recommender.git
cd group-movie-recommender
```

Create a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the demo:

```bash
python demo.py
```

---

# 👨‍💻 Author

## Priyam Keshri

# ⭐ Support

If you found this project interesting or useful, consider giving the repository a ⭐.
---

<p align="center">
  <b>🎬 One movie. Multiple preferences. One group decision.</b>
</p>
