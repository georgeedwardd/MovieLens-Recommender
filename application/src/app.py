import streamlit as st
from recommender import recommend, title_to_item_index
from utils.tmdb import get_poster_url
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --border:    rgba(255,255,255,0.07);
    --gold:      #c9a84c;
    --gold-dim:  rgba(201,168,76,0.15);
    --text:      #e8e4dc;
    --muted:     rgba(232,228,220,0.45);
    --accent:    #7c6eae;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton { display: none !important; }

[data-testid="stMainBlockContainer"],
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

.hero {
    position: relative;
    width: 100%;
    padding: 72px 64px 56px;
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%,
            rgba(201,168,76,0.10) 0%, transparent 70%),
        linear-gradient(180deg, #0d0d14 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: repeating-linear-gradient(
        90deg,
        rgba(255,255,255,0.018) 0px,
        rgba(255,255,255,0.018) 1px,
        transparent 1px,
        transparent 80px
    );
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(52px, 7vw, 92px);
    font-weight: 300;
    line-height: 0.95;
    color: var(--text);
    margin: 0 0 8px;
    letter-spacing: -0.01em;
}
.hero-title em { font-style: italic; color: var(--gold); }
.hero-sub {
    font-size: 15px;
    font-weight: 300;
    color: var(--muted);
    max-width: 480px;
    line-height: 1.6;
    margin-top: 16px;
}

.search-wrapper { padding: 48px 64px 0; }
.search-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(201,168,76,0.4) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px var(--gold-dim) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }

.section-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 52px 64px 28px;
}
.section-header h2 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 400;
    color: var(--text);
    margin: 0;
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}
.section-count {
    font-size: 11px;
    letter-spacing: 0.14em;
    color: var(--muted);
}

/* ── Card columns: use Streamlit's native column layout ── */
.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    cursor: default;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
.movie-card:hover {
    transform: translateY(-4px);
    border-color: rgba(201,168,76,0.35);
    box-shadow: 0 16px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(201,168,76,0.12);
}
.movie-card img {
    width: 100%;
    display: block;
    aspect-ratio: 2/3;
    object-fit: cover;
    filter: saturate(0.9) brightness(0.92);
    transition: filter 0.3s ease;
}
.movie-card:hover img { filter: saturate(1.05) brightness(1); }
.movie-card .card-no-poster {
    aspect-ratio: 2/3;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #1a1a26, #0f0f1a);
    font-size: 40px;
    color: var(--muted);
}
.movie-card .card-body { padding: 14px 14px 16px; }
.movie-card .card-rank {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.16em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 5px;
}
.movie-card .card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 15px;
    font-weight: 400;
    color: var(--text);
    line-height: 1.35;
    margin-bottom: 8px;
}
.movie-card .card-score-bar-bg {
    height: 2px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 6px;
}
.movie-card .card-score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--gold));
    border-radius: 2px;
}
.movie-card .card-score {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.04em;
}

/* Strip padding Streamlit adds inside columns */
[data-testid="stColumn"] > div:first-child { padding: 0 !important; }
[data-testid="stHorizontalBlock"] {
    gap: 16px !important;
    padding: 0 64px 24px !important;
}

.empty-state {
    text-align: center;
    padding: 80px 64px;
    color: var(--muted);
}
.empty-state .big-icon { font-size: 56px; margin-bottom: 16px; }
.empty-state p {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 300;
    font-style: italic;
}

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a4e; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
LINKS_PATH = os.path.join(BASE_DIR, "../data/links.csv")

links = pd.read_csv(LINKS_PATH)
movieid_to_tmdbid = dict(zip(links["movieId"], links["tmdbId"]))

def movieid_to_tmdb(movie_id):
    return movieid_to_tmdbid.get(movie_id)

def card_html(rank, title, score, bar_pct, poster_url, tmdb_id):
    img = (f'<img src="{poster_url}" alt="{title}" loading="lazy">'
           if poster_url else '<div class="card-no-poster">🎬</div>')
    link = f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else f"https://www.google.com/search?q={title.replace(' ', '+')}"
    return (
        f'<a href="{link}" target="_blank" style="text-decoration:none;display:block;">'
        f'<div class="movie-card">{img}'
        f'<div class="card-body">'
        f'<div class="card-rank">#{rank:02d}</div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-score-bar-bg">'
        f'<div class="card-score-bar-fill" style="width:{bar_pct}%"></div>'
        f'</div>'
        f'<div class="card-score">Similarity: {score:.3f}</div>'
        f'</div></div>'
        f'</a>'
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">Movie Recommender System</div>'
    '<div class="hero-title">Cine<em>Match</em></div>'
    '<div class="hero-sub">A machine learning powered system for personalised movie discovery.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.search-label {
    margin-bottom: 0.5rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown(
        '<div class="search-label">Select a title</div>',
        unsafe_allow_html=True
    )

    selected_movie = st.selectbox(
        label="",
        options=list(title_to_item_index.keys()),
        index=None,
        placeholder="Search for a movie…",
        label_visibility="collapsed",
    )

# ── Results ───────────────────────────────────────────────────────────────────
@st.fragment
def show_results(movie):
    results = recommend(movie, k=10)
    scores = [s for _, _, _, s in results]
    max_score = max(scores) if scores else 1.0

    st.markdown(
        f'<div class="section-header">'
        f'<h2>Because you chose <em style="font-style:italic;color:var(--gold)">{movie}</em></h2>'
        f'<div class="section-line"></div>'
        f'<div class="section-count">{len(results)} titles</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Render in rows of 5 using Streamlit columns — each card streams in as
    # its poster resolves, and columns are native so layout is reliable.
    rows = [results[i:i+5] for i in range(0, len(results), 5)]
    rank = 1
    for row in rows:
        cols = st.columns(len(row))
        for col, (movie_idx, movie_id, title, score) in zip(cols, row):
            tmdb_id = movieid_to_tmdb(movie_id)
            poster_url = get_poster_url(tmdb_id)
            bar_pct = round((score / max_score) * 100, 1)
            with col:
                st.markdown(card_html(rank, title, score, bar_pct, poster_url, tmdb_id), unsafe_allow_html=True)
            rank += 1

if selected_movie and selected_movie != "Select a movie...":
    show_results(selected_movie)
else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="big-icon">🎞️</div>'
        '<p>Choose a film above to begin your journey.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
