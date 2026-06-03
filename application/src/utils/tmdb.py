import requests
import streamlit as st

API_KEY = st.secrets["TMDB_API_KEY"]

_tmdb_cache = {}


def get_poster_url(tmdb_id):
    if tmdb_id is None:
        return None

    if tmdb_id in _tmdb_cache:
        return _tmdb_cache[tmdb_id]

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": API_KEY}

    r = requests.get(url, params=params).json()

    poster_path = r.get("poster_path")

    if not poster_path:
        return None

    full_url = f"https://image.tmdb.org/t/p/w500{poster_path}"

    _tmdb_cache[tmdb_id] = full_url
    return full_url