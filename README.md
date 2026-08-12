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

## 3. 👥 Group Recommendation

After generating individual predictions, GroupRecs combines them into a group-level ranking.

### 📊 Average

```text
Group Score = Mean(Member Ratings)
```

Optimizes for overall group satisfaction.

### 😈 Least Misery

```text
Group Score = Minimum(Member Ratings)
```

Focuses on the member who likes the movie the least.

### 😍 Most Pleasure

```text
Group Score = Maximum(Member Ratings)
```

Prioritizes the strongest individual preference.

### ⚖️ Fairness-Aware

```text
Group Score =
    Average Rating
    - λ × Rating Disagreement
```

Balances overall satisfaction against disagreement between group members.

For example:

```text
Alice → ⭐⭐⭐⭐⭐
Bob   → ⭐
Chloe → ⭐⭐⭐⭐⭐
Dev   → ⭐
```

may have a reasonable average rating, but significant disagreement.

A fairness-aware strategy can instead favor:

```text
Alice → ⭐⭐⭐⭐
Bob   → ⭐⭐⭐⭐
Chloe → ⭐⭐⭐⭐
Dev   → ⭐⭐⭐⭐
```

---

# 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   Movie Dataset  │
                    └────────┬─────────┘
                             │
                 ┌───────────▼───────────┐
                 │     Data Loader       │
                 │                       │
                 │ • Synthetic Dataset  │
                 │ • MovieLens 100K      │
                 └───────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌──────────────────┐
     │ Matrix          │          │ Cold-Start       │
     │ Factorization   │          │ Profiler         │
     │                 │          │                  │
     │ SGD + Latent    │          │ Genre Preferences│
     │ Factors         │          │ + Cosine Sim.    │
     └────────┬────────┘          └────────┬─────────┘
              │                            │
              └──────────────┬─────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Individual          │
                  │ Predictions         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Group Aggregator    │
                  │                     │
                  │ Average             │
                  │ Least Misery        │
                  │ Most Pleasure       │
                  │ Fairness-Aware      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Ranked Group        │
                  │ Recommendations     │
                  └─────────────────────┘
```

---

# 📁 Project Structure

```text
group_movie_recommender/
│
├── demo.py
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── recommender.py
    ├── cold_start.py
    └── group_aggregator.py
```

---

# 📂 Core Modules

## `src/data_loader.py`

Responsible for:

- Generating reproducible synthetic MovieLens-like data
- Loading MovieLens 100K data
- Preparing movie metadata
- Preparing user ratings
- Processing movie genres

## `src/recommender.py`

Contains the collaborative filtering implementation:

- User latent factors
- Movie latent factors
- User bias
- Item bias
- Global rating mean
- SGD optimization
- L2 regularization
- Rating prediction
- Per-user recommendations

## `src/cold_start.py`

Handles users without historical rating data.

The module:

1. Accepts preferred genres
2. Builds a normalized preference vector
3. Represents movies using genre vectors
4. Calculates cosine similarity
5. Converts similarity into predicted ratings

## `src/group_aggregator.py`

Contains the group recommendation logic.

Supported strategies:

```python
average
least_misery
most_pleasure
fairness_aware
```

## `demo.py`

Runs the complete recommendation pipeline:

```text
Generate Data
      ↓
Train Model
      ↓
Create Group
      ↓
Generate Individual Predictions
      ↓
Handle Cold-Start Users
      ↓
Aggregate Group Preferences
      ↓
Compare Strategies
      ↓
Display Recommendations
```

---

# 🧪 Example Group

The demo simulates a group containing both existing and new users.

```text
Alice → Existing user with rating history

Bob   → Existing user with rating history

Chloe → New user
        Preferences:
        Comedy + Romance

Dev   → New user
        Preferences:
        Horror + Thriller
```

This demonstrates how the system can combine collaborative-filtering and cold-start predictions within the same group.

---

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

# 🧮 Mathematical Intuition

## Matrix Factorization

A user's preference for a movie can be represented using latent vectors:

```text
User Vector        Movie Vector

[u₁ u₂ u₃ ...] · [v₁ v₂ v₃ ...]
         │
         ▼
   Predicted Rating
```

The predicted rating can be represented as:

```text
r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ
```

Where:

```text
μ   = Global average rating
bᵤ  = User bias
bᵢ  = Movie bias
pᵤ  = User latent vector
qᵢ  = Movie latent vector
```

## Optimization

The model minimizes a regularized squared-error objective:

```text
Loss =
    Σ(rᵤᵢ - r̂ᵤᵢ)²
    + λ(||pᵤ||² + ||qᵢ||²)
```

The parameters are updated using **Stochastic Gradient Descent**.

## Cold-Start Similarity

For new users, movie and preference vectors are compared using cosine similarity:

```text
                  A · B
cosine(A, B) = ─────────────
               ||A|| × ||B||
```

A higher similarity indicates stronger alignment between the user's preferred genres and the movie's genres.

---

# ⚖️ Fairness-Aware Recommendation

The fairness-aware strategy introduces a disagreement penalty:

```text
Fairness Score
=
Average Satisfaction
-
λ × Disagreement
```

Where:

```text
λ = Fairness penalty weight
```

This creates a trade-off between:

```text
Maximum Satisfaction
        ↕
Maximum Agreement
```

The goal is not simply to find the movie with the highest average score, but to find a movie that provides a **better-balanced experience for the group**.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Core implementation |
| 🔢 NumPy | Matrix operations and ML algorithms |
| 🐼 Pandas | Data loading and preprocessing |
| 🤖 Matrix Factorization | Collaborative filtering |
| ⚡ SGD | Model optimization |
| 📐 Cosine Similarity | Cold-start recommendation |
| ⚖️ Social Choice | Group preference aggregation |
| 🎬 MovieLens 100K | Movie rating dataset |

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

The demo executes:

```text
1. Generate / load movie data
2. Train matrix factorization model
3. Create a simulated group
4. Generate individual predictions
5. Generate cold-start predictions
6. Apply group aggregation strategies
7. Rank recommended movies
8. Compare recommendation strategies
```

---

# 🎬 MovieLens 100K Dataset

GroupRecs supports the **MovieLens 100K dataset**.

Download:

https://files.grouplens.org/datasets/movielens/ml-100k.zip

After downloading, extract it into:

```text
data/
└── ml-100k/
    ├── u.data
    ├── u.item
    └── ...
```

Example usage:

```python
from src.data_loader import load_movielens_100k

ratings_df, movies_df, genre_columns = load_movielens_100k(
    "data/ml-100k"
)
```

---

# 💡 Why This Project?

Most introductory recommendation-system projects focus on:

> **"Recommend movies that this user will like."**

GroupRecs explores a more challenging question:

> **"How can we recommend a movie that a group of people can collectively enjoy?"**

This introduces additional challenges such as:

- Conflicting preferences
- Group satisfaction
- Fairness
- Disagreement
- Cold-start users
- Different definitions of a "good" recommendation

The project combines concepts from:

- Machine Learning
- Recommendation Systems
- Collaborative Filtering
- Content-Based Recommendation
- Group Decision Making
- Social Choice
- Fairness-Aware Ranking

---

# 📚 Concepts Demonstrated

## Machine Learning

- Matrix Factorization
- Collaborative Filtering
- Latent Factor Models
- Stochastic Gradient Descent
- Regularization
- Model Training
- Rating Prediction

## Recommendation Systems

- Personalized Recommendation
- Content-Based Recommendation
- Cold-Start Problem
- Genre-Based Profiling
- Similarity-Based Ranking
- Top-K Recommendation

## Group Recommendation

- Group Preference Aggregation
- Average Aggregation
- Least Misery
- Most Pleasure
- Fairness-Aware Ranking
- Preference Disagreement

## Data Science

- NumPy
- Pandas
- Data Preprocessing
- Feature Representation
- Vector Similarity
- Dataset Handling

---

# 🚧 Roadmap

- [ ] Integrate MovieLens 100K as the default dataset
- [ ] Add train / validation / test splits
- [ ] Add RMSE evaluation
- [ ] Add MAE evaluation
- [ ] Add Precision@K
- [ ] Add Recall@K
- [ ] Experiment with different latent-factor sizes
- [ ] Tune SGD learning rate and regularization
- [ ] Add additional group aggregation strategies
- [ ] Add user weighting
- [ ] Add movie diversity constraints
- [ ] Add popularity-aware recommendations
- [ ] Build a Streamlit interface
- [ ] Build a FastAPI backend
- [ ] Add real-time group movie rooms
- [ ] Add user authentication
- [ ] Deploy the application

---

# 🌐 Future Architecture

```text
                 ┌──────────────────┐
                 │    Streamlit     │
                 │    Frontend      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     FastAPI      │
                 │     Backend      │
                 └────────┬─────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │Recommender │  │   Group    │  │   User     │
   │   Engine   │  │   Engine   │  │  Profiles  │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                ┌─────────────────┐
                │ Movie Database  │
                └─────────────────┘
```

---

# 🔬 Future Research Directions

### Dynamic Group Preferences

Allow users to change their preferences during a movie session.

### User Importance Weighting

Different users could have different levels of influence:

```text
Alice → 40%
Bob   → 30%
Chloe → 20%
Dev   → 10%
```

### Diversity-Aware Recommendation

Prevent the system from recommending multiple movies that are nearly identical.

### Context-Aware Recommendation

Consider contextual information such as:

```text
Time of day
Movie duration
Genre
Age suitability
Previous group choices
Mood
```

### Multi-Objective Optimization

Future versions could optimize:

```text
Satisfaction
     +
Fairness
     +
Diversity
     +
Novelty
     +
Popularity
```

---

# 📈 Project Highlights

```text
┌─────────────────────────────────────────────┐
│                  GROUPRECS                  │
├─────────────────────────────────────────────┤
│                                             │
│  🤖 Collaborative Filtering                 │
│  🧠 Matrix Factorization                    │
│  ⚡ SGD Optimization                        │
│  🆕 Cold-Start Handling                     │
│  🎭 Genre Profiling                         │
│  👥 Group Recommendation                    │
│  ⚖️ Fairness-Aware Ranking                  │
│  📊 Multiple Aggregation Strategies         │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 🎯 Learning Outcomes

Through this project, I explored how to:

- Build a recommendation system from scratch
- Implement matrix factorization without a specialized recommender library
- Optimize latent-factor models using SGD
- Handle the cold-start problem
- Represent user and movie preferences as vectors
- Use cosine similarity for content-based recommendation
- Design group recommendation algorithms
- Compare different social-choice strategies
- Incorporate fairness into recommendation ranking
- Build modular machine-learning components
- Work with real-world movie-rating datasets

---

# 👨‍💻 Author

## Priyam Keshri

**AI/ML Engineer • GenAI Developer • Software Engineer**

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Generative AI
- Recommendation Systems
- Software Engineering
- Backend Development
- Data Science

---

# ⭐ Support

If you found this project interesting or useful, consider giving the repository a ⭐.

---

<p align="center">
  <b>🎬 One movie. Multiple preferences. One group decision.</b>
</p>
