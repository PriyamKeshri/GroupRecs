"""
FastAPI backend for the group movie recommender.

Wraps the existing pure-Python pieces (recommender.py, cold_start.py,
demographics.py, group_aggregator.py) behind a small REST API so a real
frontend (Streamlit, a mobile app, curl, whatever) can drive a "movie
night" session:

    1. POST /groups                          -- start a session
    2. POST /groups/{id}/members/existing     -- add a member with rating history
    3. POST /groups/{id}/members/guest        -- add a cold-start guest, via
                                                  genres, demographics, or both
    4. GET  /groups/{id}/recommend            -- ranked list under one strategy
    5. GET  /groups/{id}/compare              -- all strategies side by side

A guest can supply `liked_genres`, or `age`+`gender`+`occupation`, or both --
when both are given their predictions are blended equally (see
demographics.blend_predictions).

State is kept in memory (a dict of groups + one trained model), which is
fine for a demo/portfolio API. Swap `AppState` for a real DB-backed store
before this ever sees real traffic.

Mutating endpoints (create a group, add/remove a member) require HTTP
Basic Auth so a public deployment isn't editable by anyone with the link
-- read endpoints (browsing the catalog, viewing a group's recommendations)
stay open. Credentials come from the AUTH_USERNAME / AUTH_PASSWORD
environment variables -- never hardcode real credentials here, this file
is public. See _require_auth() below.

FastAPI runs sync endpoint functions (like these) across a worker thread
pool, so concurrent requests are genuinely concurrent, not just
interleaved async tasks -- two requests can run their Python bodies at the
same instant on different threads. Every endpoint that reads or writes
`state.groups` (or a group's `members` dict) holds `state.lock` for that
reason; drop it and a member add/remove racing a recommend/compare read
reliably throws `RuntimeError: dictionary changed size during iteration`
(this happened in practice -- Streamlit's rerun model fires overlapping
requests routinely, not just under artificial load).
"""

import os
import secrets
import threading
import uuid
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, model_validator

from .cold_start import ColdStartProfiler
from .demographics import DemographicProfiler, blend_predictions
from .group_aggregator import STRATEGIES, compare_all_strategies, recommend_for_group
from .model_cache import load_or_train
from .recommender import MatrixFactorizationRecommender

VALID_GENDERS = {"M", "F"}

# Fallback credentials so local dev (./run.sh, demo/testing) works with zero
# setup. A deployed instance MUST override these via real environment
# variables -- _require_auth() below prints a loud startup warning if it's
# still running on the placeholder password, so a forgotten override is
# obvious in the logs rather than a silent security hole.
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "changeme")

_basic_auth = HTTPBasic()


def _require_auth(credentials: HTTPBasicCredentials = Depends(_basic_auth)):
    """Dependency for mutating endpoints. compare_digest avoids leaking
    match-length via response-timing side channels."""
    user_ok = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    pass_ok = secrets.compare_digest(credentials.password, AUTH_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


# --------------------------------------------------------------------------
# In-memory app state
# --------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.movies_df = None
        self.genre_columns: List[str] = []
        self.occupations: List[str] = []
        self.recommender: Optional[MatrixFactorizationRecommender] = None
        self.cold_start: Optional[ColdStartProfiler] = None
        self.demographics: Optional[DemographicProfiler] = None
        self.dataset: str = "unknown"
        self.demographics_source: str = "none"  # "real" | "synthetic" | "none"
        self.groups: Dict[str, dict] = {}  # group_id -> {"name": str, "members": {label: member}}
        self.lock = threading.Lock()  # guards state.groups -- see module docstring


state = AppState()


def _train_model():
    """Load the best available dataset (cached model if one matches) and
    populate app state. See src/model_cache.py for the caching contract."""
    result = load_or_train()

    state.movies_df = result["movies_df"]
    state.genre_columns = result["genre_columns"]
    state.occupations = sorted(result["users_df"]["occupation"].unique().tolist())
    state.recommender = result["model"]
    state.cold_start = ColdStartProfiler(result["movies_df"], result["genre_columns"])
    state.demographics = DemographicProfiler(result["users_df"], result["ratings_df"])
    state.dataset = result["dataset_name"]
    state.demographics_source = result["demographics_source"]


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

class GroupCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, examples=["Friday movie night"])


class ExistingMemberRequest(BaseModel):
    label: str = Field(..., min_length=1, examples=["Alice"])
    user_id: int


class GuestMemberRequest(BaseModel):
    label: str = Field(..., min_length=1, examples=["Chloe"])
    liked_genres: Optional[List[str]] = Field(default=None, examples=[["Comedy", "Sci-Fi"]])
    age: Optional[int] = Field(default=None, ge=1, le=120, examples=[25])
    gender: Optional[str] = Field(default=None, examples=["F"])
    occupation: Optional[str] = Field(default=None, examples=["student"])

    @model_validator(mode="after")
    def _require_at_least_one_signal(self):
        has_genres = bool(self.liked_genres)
        demo_fields = [self.age, self.gender, self.occupation]
        has_demo = any(f is not None for f in demo_fields)
        if has_demo and not all(f is not None for f in demo_fields):
            raise ValueError("age, gender, and occupation must all be given together (or all omitted)")
        if not has_genres and not has_demo:
            raise ValueError(
                "provide at least one signal: 'liked_genres', or 'age'+'gender'+'occupation'"
            )
        return self


class MemberInfo(BaseModel):
    label: str
    kind: str  # "existing" | "guest"
    detail: str  # user_id, or comma-joined genres


class GroupResponse(BaseModel):
    group_id: str
    name: str
    members: List[MemberInfo]


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    score: float
    member_scores: Dict[str, float]


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="Group Movie Recommender API",
    description="Predicts per-member ratings and aggregates them into a "
                 "single group recommendation using multiple social-choice "
                 "strategies (average / least_misery / most_pleasure / fairness_aware).",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup():
    _train_model()
    if AUTH_PASSWORD == "changeme":
        print(
            "  WARNING: AUTH_PASSWORD is not set -- running on the default "
            "admin/changeme credentials. Fine for local dev; set AUTH_USERNAME "
            "and AUTH_PASSWORD env vars before this is reachable publicly."
        )


# ---- helpers --------------------------------------------------------------

def _get_group(group_id: str) -> dict:
    group = state.groups.get(group_id)
    if group is None:
        raise HTTPException(404, f"group '{group_id}' not found")
    return group


def _member_info(label: str, member: dict) -> MemberInfo:
    if member["kind"] == "existing":
        return MemberInfo(label=label, kind=member["kind"], detail=f"user_id={member['user_id']}")

    parts = []
    if member.get("liked_genres"):
        parts.append(", ".join(member["liked_genres"]))
    if member.get("age") is not None:
        parts.append(f"{member['age']}yo {member['gender']}, {member['occupation']}")
    return MemberInfo(label=label, kind=member["kind"], detail="  +  ".join(parts))


def _group_response(group_id: str, group: dict) -> GroupResponse:
    return GroupResponse(
        group_id=group_id,
        name=group["name"],
        members=[_member_info(label, m) for label, m in group["members"].items()],
    )


def _member_predictions(group: dict) -> Dict[str, Dict[int, float]]:
    if not group["members"]:
        raise HTTPException(400, "group has no members yet -- add at least one")

    preds = {}
    for label, member in group["members"].items():
        if member["kind"] == "existing":
            preds[label] = state.recommender.predict_for_user(member["user_id"])
        else:
            signals = []
            if member.get("liked_genres"):
                signals.append((state.cold_start.predict_for_guest(member["liked_genres"]), 1.0))
            if member.get("age") is not None:
                demo_preds = state.demographics.predict_for_guest(
                    member["age"], member["gender"], member["occupation"]
                )
                signals.append((demo_preds, 1.0))
            preds[label] = blend_predictions(signals)
    return preds


def _title_for(item_id: int) -> str:
    row = state.movies_df.loc[state.movies_df["item_id"] == item_id]
    return row["title"].iloc[0] if len(row) else f"Movie {item_id}"


def _format_ranked(ranked, preds) -> List[RecommendationItem]:
    out = []
    for item_id, score in ranked:
        member_scores = {label: round(float(p.get(item_id, 0.0)), 3) for label, p in preds.items()}
        out.append(RecommendationItem(
            item_id=int(item_id),
            title=_title_for(item_id),
            score=round(float(score), 3),
            member_scores=member_scores,
        ))
    return out


# ---- health / catalog ------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": state.dataset,
        "n_known_users": len(state.recommender.user_id_to_idx),
        "demographics_source": state.demographics_source,
    }


@app.get("/genres")
def list_genres():
    return state.genre_columns


@app.get("/occupations")
def list_occupations():
    """Occupations present in the loaded dataset -- usable with the
    demographic side of the 'guest member' endpoint."""
    return state.occupations


@app.get("/users")
def list_users():
    """user_ids the trained model knows -- usable with the 'existing member' endpoint."""
    return sorted(int(uid) for uid in state.recommender.user_id_to_idx.keys())


@app.get("/movies")
def list_movies(genre: Optional[str] = None):
    df = state.movies_df
    if genre is not None:
        if genre not in state.genre_columns:
            raise HTTPException(400, f"unknown genre '{genre}'. choose from {state.genre_columns}")
        df = df[df[genre] == 1]
    return df[["item_id", "title"]].to_dict(orient="records")


# ---- groups -----------------------------------------------------------------

@app.post("/groups", response_model=GroupResponse, status_code=201)
def create_group(req: GroupCreateRequest, _auth: None = Depends(_require_auth)):
    with state.lock:
        group_id = uuid.uuid4().hex[:8]
        state.groups[group_id] = {"name": req.name, "members": {}}
        return _group_response(group_id, state.groups[group_id])


@app.get("/groups/{group_id}", response_model=GroupResponse)
def get_group(group_id: str):
    with state.lock:
        group = _get_group(group_id)
        return _group_response(group_id, group)


@app.post("/groups/{group_id}/members/existing", response_model=GroupResponse, status_code=201)
def add_existing_member(group_id: str, req: ExistingMemberRequest, _auth: None = Depends(_require_auth)):
    with state.lock:
        group = _get_group(group_id)
        if req.label in group["members"]:
            raise HTTPException(400, f"label '{req.label}' is already used in this group")
        if not state.recommender.is_known_user(req.user_id):
            raise HTTPException(400, f"user_id {req.user_id} is not known to the model (see GET /users)")

        group["members"][req.label] = {"kind": "existing", "user_id": req.user_id}
        return _group_response(group_id, group)


@app.post("/groups/{group_id}/members/guest", response_model=GroupResponse, status_code=201)
def add_guest_member(group_id: str, req: GuestMemberRequest, _auth: None = Depends(_require_auth)):
    with state.lock:
        group = _get_group(group_id)
        if req.label in group["members"]:
            raise HTTPException(400, f"label '{req.label}' is already used in this group")

        member = {"kind": "guest"}

        if req.liked_genres:
            unknown = [g for g in req.liked_genres if g not in state.genre_columns]
            if unknown:
                raise HTTPException(400, f"unknown genres {unknown}. choose from {state.genre_columns}")
            member["liked_genres"] = req.liked_genres

        if req.age is not None:
            gender = req.gender.upper()
            if gender not in VALID_GENDERS:
                raise HTTPException(400, f"unknown gender '{req.gender}'. choose from {sorted(VALID_GENDERS)}")
            if req.occupation not in state.occupations:
                raise HTTPException(400, f"unknown occupation '{req.occupation}'. choose from {state.occupations}")
            member["age"] = req.age
            member["gender"] = gender
            member["occupation"] = req.occupation

        group["members"][req.label] = member
        return _group_response(group_id, group)


@app.delete("/groups/{group_id}/members/{label}", response_model=GroupResponse)
def remove_member(group_id: str, label: str, _auth: None = Depends(_require_auth)):
    with state.lock:
        group = _get_group(group_id)
        if label not in group["members"]:
            raise HTTPException(404, f"member '{label}' not found in group '{group_id}'")
        del group["members"][label]
        return _group_response(group_id, group)


# ---- recommendations ---------------------------------------------------------

@app.get("/groups/{group_id}/recommend", response_model=List[RecommendationItem])
def recommend(group_id: str, strategy: str = "average", top_n: int = 10):
    if strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy '{strategy}'. choose from {list(STRATEGIES.keys())}")

    with state.lock:
        group = _get_group(group_id)
        preds = _member_predictions(group)
    ranked = recommend_for_group(preds, top_n=top_n, strategy=strategy)
    return _format_ranked(ranked, preds)


@app.get("/groups/{group_id}/compare")
def compare(group_id: str, top_n: int = 5):
    with state.lock:
        group = _get_group(group_id)
        preds = _member_predictions(group)
    all_results = compare_all_strategies(preds, top_n=top_n)
    return {strategy: _format_ranked(ranked, preds) for strategy, ranked in all_results.items()}
