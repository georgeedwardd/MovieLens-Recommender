import requests

API_KEY = "bd71de79cf848b2c0a35fc9ae921ec13"

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