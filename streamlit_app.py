"""
Streamlit "group room" UI for the group movie recommender.

Talks to the FastAPI backend (src/api.py) over HTTP -- this file has zero
knowledge of matrix factorization / cold-start / aggregation internals, it
just renders whatever the API returns. Run the API first:

    python3 -m uvicorn src.api:app --reload --port 8000

then:

    streamlit run streamlit_app.py

Visual theme lives in .streamlit/config.toml (base palette) plus the CSS
block below (cards, chips, score bars) -- everything else is plain
Streamlit widgets.
"""

import html
import os

import requests
import streamlit as st


def _default_api_url():
    """Where the API lives, in priority order: Streamlit Cloud's "Secrets"
    panel (st.secrets), then a plain environment variable (Render/other
    hosts), then localhost for local dev. Lets the same code default
    correctly whether it's running locally or deployed, without a code
    change -- just set API_BASE_URL wherever it's hosted."""
    try:
        if "API_BASE_URL" in st.secrets:
            return st.secrets["API_BASE_URL"]
    except Exception:
        pass
    return os.environ.get("API_BASE_URL", "http://localhost:8000")


DEFAULT_API_URL = _default_api_url()

STRATEGIES = [
    ("average", "🔵 Average", "Best overall satisfaction", "#4ea8de"),
    ("least_misery", "🟢 Least misery", "No one hates the pick", "#2a9d8f"),
    ("most_pleasure", "🔴 Most pleasure", "The biggest fan's favorite", "#e63946"),
    ("fairness_aware", "🟣 Fairness-aware", "Penalizes divisive picks", "#9b5de5"),
]
STRATEGY_COLORS = {key: color for key, _, _, color in STRATEGIES}
STRATEGY_TAB_LABELS = [label for _, label, _, _ in STRATEGIES]

MEMBER_STYLE = {
    "existing": ("📼", "#4ea8de"),
    "guest": ("🆕", "#f3a712"),
}

st.set_page_config(page_title="Group Movie Night", page_icon="🎬", layout="centered")

st.markdown("""
<style>
/* ---- layout tightening ---- */
.block-container { padding-top: 2.5rem; max-width: 760px; }

/* ---- hero header ---- */
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.15rem;
    background: linear-gradient(90deg, #e94560, #f3a712);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: rgba(234,234,234,0.55); font-size: 1rem; margin-bottom: 1.6rem; }

/* ---- section headings ---- */
.section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin: 1.4rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ---- session badge ---- */
.session-badge {
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.75rem;
    color: rgba(234,234,234,0.55);
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    margin-top: 0.1rem;
}

/* ---- member chip ---- */
.member-chip {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid var(--accent, #e94560);
    border-radius: 10px;
    padding: 0.55rem 1rem;
    margin-bottom: 0.5rem;
}
.member-icon { font-size: 1.1rem; }
.member-name { font-weight: 600; }
.member-detail {
    color: rgba(234,234,234,0.55);
    font-size: 0.82rem;
    margin-left: auto;
    text-align: right;
    padding-left: 0.75rem;
}

/* ---- recommendation card ---- */
.rec-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
}
.rec-rank {
    flex-shrink: 0;
    width: 2.1rem;
    height: 2.1rem;
    border-radius: 50%;
    background: var(--accent, #e94560);
    color: #0f0f1a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.95rem;
}
.rec-body { flex: 1; min-width: 0; }
.rec-title {
    font-weight: 650;
    font-size: 1rem;
    margin-bottom: 0.4rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.rec-score-bar {
    width: 100%;
    height: 5px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 0.55rem;
}
.rec-score-fill { height: 100%; background: var(--accent, #e94560); border-radius: 3px; }
.rec-members { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.score-chip {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.06);
}
.rec-score-badge {
    flex-shrink: 0;
    font-weight: 800;
    font-size: 1.15rem;
    color: var(--accent, #e94560);
    min-width: 2.6rem;
    text-align: right;
}

/* ---- misc ---- */
.dim-caption { color: rgba(234,234,234,0.45); font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# API client helpers -- every network call funnels through here so
# connection errors are handled in exactly one place.
# --------------------------------------------------------------------------

def _request(method, path, **kwargs):
    url = f"{st.session_state.api_url}{path}"
    try:
        return requests.request(method, url, timeout=5, **kwargs)
    except requests.exceptions.RequestException as exc:
        st.error(f"Can't reach the API at `{st.session_state.api_url}`.\n\n{exc}")
        st.stop()


def api_get(path, **params):
    return _request("GET", path, params=params)


def api_post(path, body):
    return _request("POST", path, json=body)


def api_delete(path):
    return _request("DELETE", path)


def error_detail(resp, fallback="Something went wrong."):
    try:
        detail = resp.json().get("detail", fallback)
        return detail if isinstance(detail, str) else fallback
    except ValueError:
        return fallback


def esc(text):
    """Escape user-supplied text before it goes into raw HTML (member names
    are free text, so this is real, not decorative)."""
    return html.escape(str(text))


def score_color(score):
    if score >= 4.0:
        return "#2a9d8f"
    if score >= 2.5:
        return "#f3a712"
    return "#e63946"


@st.cache_data(ttl=60)
def get_users(api_url):
    return requests.get(f"{api_url}/users", timeout=5).json()


@st.cache_data(ttl=60)
def get_genres(api_url):
    return requests.get(f"{api_url}/genres", timeout=5).json()


@st.cache_data(ttl=60)
def get_occupations(api_url):
    return requests.get(f"{api_url}/occupations", timeout=5).json()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

st.session_state.setdefault("api_url", DEFAULT_API_URL)
st.session_state.setdefault("group_id", None)
st.session_state.setdefault("group_name", None)

st.sidebar.markdown("### ⚙️ Settings")
st.session_state.api_url = st.sidebar.text_input("API base URL", st.session_state.api_url)

health = api_get("/health")
if health.status_code == 200:
    info = health.json()
    st.sidebar.success(f"Connected — {info['n_known_users']} known users")
    st.sidebar.markdown(f'<span class="session-badge">dataset: {esc(info.get("dataset", "unknown"))}</span>',
                         unsafe_allow_html=True)
    if info.get("demographics_source") == "synthetic":
        st.sidebar.caption(
            "⚠️ This dataset has no real age/gender/occupation data — "
            "demographic cold-start uses synthesized peers, not real ones."
        )
else:
    st.sidebar.error(f"API returned {health.status_code}")
    st.stop()

st.markdown('<div class="hero-title">🎬 Group Movie Night</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Predict what your group will actually enjoy watching — together.</div>',
            unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Step 1: create / resume a group
# --------------------------------------------------------------------------

if st.session_state.group_id is None:
    st.markdown('<div class="section-heading">🚀 Start a session</div>', unsafe_allow_html=True)
    with st.form("create_group"):
        name = st.text_input("Group name", placeholder="Friday night")
        submitted = st.form_submit_button("Create group", type="primary")
    if submitted:
        if not name:
            st.error("Give the session a name.")
        else:
            resp = api_post("/groups", {"name": name})
            if resp.status_code == 201:
                data = resp.json()
                st.session_state.group_id = data["group_id"]
                st.session_state.group_name = data["name"]
                st.rerun()
            else:
                st.error(error_detail(resp, "Failed to create group."))
    st.stop()

gid = st.session_state.group_id

group_resp = api_get(f"/groups/{gid}")
if group_resp.status_code != 200:
    st.warning("This session no longer exists on the server (did it restart?). Starting over.")
    st.session_state.group_id = None
    st.session_state.group_name = None
    st.rerun()
group = group_resp.json()

header_col, reset_col = st.columns([5, 2])
with header_col:
    st.markdown(f'<div class="section-heading">👥 {esc(group["name"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="session-badge">session {esc(gid)}</span>', unsafe_allow_html=True)
with reset_col:
    st.write("")
    if st.button("Start a new session"):
        st.session_state.group_id = None
        st.session_state.group_name = None
        st.rerun()


# --------------------------------------------------------------------------
# Step 2: members
# --------------------------------------------------------------------------

st.markdown('<div class="section-heading">Members</div>', unsafe_allow_html=True)

if not group["members"]:
    st.info("No members yet — add someone below.")
else:
    for m in group["members"]:
        icon, color = MEMBER_STYLE.get(m["kind"], ("👤", "#e94560"))
        row_col, btn_col = st.columns([9, 1])
        with row_col:
            st.markdown(
                f"""<div class="member-chip" style="--accent:{color}">
                        <span class="member-icon">{icon}</span>
                        <span class="member-name">{esc(m['label'])}</span>
                        <span class="member-detail">{esc(m['detail'])}</span>
                    </div>""",
                unsafe_allow_html=True,
            )
        with btn_col:
            if st.button("✕", key=f"remove_{m['label']}", help=f"Remove {m['label']}"):
                api_delete(f"/groups/{gid}/members/{m['label']}")
                st.rerun()

st.markdown('<div class="section-heading">➕ Add a member</div>', unsafe_allow_html=True)
tab_existing, tab_guest = st.tabs(["🎞️ Existing user", "🆕 New guest (cold start)"])

with tab_existing:
    users = get_users(st.session_state.api_url)
    st.markdown('<span class="dim-caption">Someone with rating history already in the system.</span>',
                unsafe_allow_html=True)
    with st.form("add_existing"):
        label = st.text_input("Name", key="existing_label")
        user_id = st.selectbox("User ID", users)
        submitted = st.form_submit_button("Add", type="primary")
    if submitted:
        if not label:
            st.error("Give them a name.")
        else:
            resp = api_post(f"/groups/{gid}/members/existing", {"label": label, "user_id": user_id})
            if resp.status_code == 201:
                st.rerun()
            else:
                st.error(error_detail(resp, "Failed to add member."))

with tab_guest:
    genres = get_genres(st.session_state.api_url)
    occupations = get_occupations(st.session_state.api_url)
    st.markdown(
        '<span class="dim-caption">A friend with no rating history. Give either signal below '
        '(or both — they get blended into one prediction).</span>',
        unsafe_allow_html=True,
    )
    with st.form("add_guest"):
        label = st.text_input("Name", key="guest_label")

        st.markdown("**🎭 By taste**")
        liked = st.multiselect("Genres they like", genres)

        st.markdown("**👥 By demographics** — predicts from what similar people rated")
        demo_cols = st.columns(3)
        age = demo_cols[0].number_input("Age", min_value=0, max_value=120, value=0, help="0 = skip")
        gender = demo_cols[1].selectbox("Gender", ["(skip)", "M", "F"])
        occupation = demo_cols[2].selectbox("Occupation", ["(skip)"] + occupations)

        submitted = st.form_submit_button("Add", type="primary")
    if submitted:
        has_demo_field = age > 0 or gender != "(skip)" or occupation != "(skip)"
        has_full_demo = age > 0 and gender != "(skip)" and occupation != "(skip)"

        if not label:
            st.error("Give them a name.")
        elif not liked and not has_demo_field:
            st.error("Pick at least one genre, or fill in age + gender + occupation.")
        elif has_demo_field and not has_full_demo:
            st.error("Age, gender, and occupation must all be filled in together (or all left blank).")
        else:
            body = {"label": label}
            if liked:
                body["liked_genres"] = liked
            if has_full_demo:
                body["age"] = age
                body["gender"] = gender
                body["occupation"] = occupation
            resp = api_post(f"/groups/{gid}/members/guest", body)
            if resp.status_code == 201:
                st.rerun()
            else:
                st.error(error_detail(resp, "Failed to add member."))


# --------------------------------------------------------------------------
# Step 3: recommendations
# --------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-heading">🎯 Recommendations</div>', unsafe_allow_html=True)

if not group["members"]:
    st.info("Add at least one member to get recommendations.")
else:
    top_n = st.slider("How many movies per strategy", 1, 10, 5)

    compare_resp = api_get(f"/groups/{gid}/compare", top_n=top_n)
    if compare_resp.status_code != 200:
        st.error(error_detail(compare_resp, "Failed to get recommendations."))
    else:
        results = compare_resp.json()
        strategy_tabs = st.tabs(STRATEGY_TAB_LABELS)
        for tab, (strategy_key, _, strategy_desc, accent) in zip(strategy_tabs, STRATEGIES):
            with tab:
                st.markdown(f'<span class="dim-caption">{esc(strategy_desc)}</span>', unsafe_allow_html=True)
                st.write("")
                items = results[strategy_key]
                if not items:
                    st.caption("No candidates in common across members yet.")
                for rank, item in enumerate(items, start=1):
                    score_pct = max(0.0, min(1.0, item["score"] / 5.0)) * 100
                    member_chips = "".join(
                        f'<span class="score-chip" style="color:{score_color(score)}">'
                        f'{esc(name)}: {score}</span>'
                        for name, score in item["member_scores"].items()
                    )
                    st.markdown(
                        f"""<div class="rec-card" style="--accent:{accent}">
                                <div class="rec-rank">{rank}</div>
                                <div class="rec-body">
                                    <div class="rec-title">{esc(item['title'])}</div>
                                    <div class="rec-score-bar">
                                        <div class="rec-score-fill" style="width:{score_pct}%"></div>
                                    </div>
                                    <div class="rec-members">{member_chips}</div>
                                </div>
                                <div class="rec-score-badge">{item['score']}</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<span class="dim-caption">Notice how the top pick can shift between tabs — that\'s the whole '
            "point. 'Average' can surface something divisive if the mean is high; 'Least misery' avoids "
            "anything one person would hate; 'Most pleasure' chases the single biggest fan; "
            "'Fairness-aware' softens divisive picks even with a strong average. Per-member scores above "
            "are color-coded green/amber/red so a divisive pick is visible at a glance.</span>",
            unsafe_allow_html=True,
        )
