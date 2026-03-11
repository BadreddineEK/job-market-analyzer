"""
Job search module — multiple backends for French job market.

Backends (tried in order):
  1. Indeed France via python-jobspy   — works without any config
  2. France Travail (Pôle Emploi) API  — best French source, needs free env setup
     Set FRANCETRAVAIL_CLIENT_ID and FRANCETRAVAIL_CLIENT_SECRET in .env
  3. WTTJ via dynamic Algolia key      — extracts API key live from WTTJ page

Each result: {title, company, url, location, source, description}
When description is non-empty, app.py skips re-scraping the URL.
"""

from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# DDG HTML headers (less bot-detector friendly than Chrome spoof)
_DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://duckduckgo.com/",
}

SUPPORTED_SITES = ["Indeed", "WTTJ", "Hellowork", "France Travail"]

# ─── 1. Indeed via jobspy ─────────────────────────────────────────────────────

def _search_indeed(query: str, location: str, n: int) -> list[dict]:
    """Search Indeed France via python-jobspy (no API key needed)."""
    try:
        from jobspy import scrape_jobs
        df = scrape_jobs(
            site_name=["indeed"],
            search_term=query,
            location=location or "France",
            country_indeed="France",
            results_wanted=n,
            verbose=0,
        )
        if df is None or df.empty:
            return []
        results = []
        seen: set[str] = set()
        for _, row in df.iterrows():
            url = str(row.get("job_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            desc = str(row.get("description") or "").strip()
            results.append({
                "title":       str(row.get("title")    or "").strip(),
                "company":     str(row.get("company")   or "").strip(),
                "url":         url,
                "location":    str(row.get("location")  or "").strip(),
                "source":      "Indeed",
                "description": desc[:6000],
            })
        return results
    except Exception:
        return []


# ─── 2. France Travail (Pôle Emploi) API ─────────────────────────────────────

_FT_TOKEN_URL = "https://francetravail.io/connexion/oauth2/access_token?realm=%2Fpartenaire"
_FT_SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
_ft_token_cache: dict = {}


def _ft_token() -> str | None:
    """Fetch (or reuse cached) France Travail OAuth2 access token."""
    client_id     = os.getenv("FRANCETRAVAIL_CLIENT_ID", "")
    client_secret = os.getenv("FRANCETRAVAIL_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None

    now = time.time()
    if _ft_token_cache.get("expires_at", 0) > now + 30:
        return _ft_token_cache["token"]

    try:
        r = requests.post(
            _FT_TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     client_id,
                "client_secret": client_secret,
                "scope":         "api_offresdemploiv2 o2dsoffre",
            },
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        _ft_token_cache["token"]      = payload["access_token"]
        _ft_token_cache["expires_at"] = now + payload.get("expires_in", 1200)
        return _ft_token_cache["token"]
    except Exception:
        return None


def _search_france_travail(query: str, location: str, n: int) -> list[dict]:
    """Search France Travail API (requires FRANCETRAVAIL_CLIENT_ID/SECRET in .env)."""
    token = _ft_token()
    if not token:
        return []

    params: dict = {
        "motsCles":        query,
        "range":           f"0-{min(n, 150) - 1}",
        "sort":            "1",   # pertinence
    }
    if location:
        params["commune"] = location  # INSEE code or free text (best-effort)

    try:
        r = requests.get(
            _FT_SEARCH_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    results = []
    for offer in data.get("resultats", []):
        offer_id = offer.get("id", "")
        url = f"https://www.francetravail.fr/offres/recherche/detail/{offer_id}"
        desc = offer.get("description", "")
        entreprise = offer.get("entreprise", {}) or {}
        lieu = (offer.get("lieuTravail") or {}).get("libelle", "")
        results.append({
            "title":       offer.get("intitule", ""),
            "company":     entreprise.get("nom", ""),
            "url":         url,
            "location":    lieu,
            "source":      "France Travail",
            "description": desc[:6000],
        })
    return results


# ─── 3. WTTJ + Hellowork via DuckDuckGo site-search ─────────────────────────
#
# DuckDuckGo HTML (html.duckduckgo.com) is scrapable without authentication
# and without JavaScript. We use it to find job URLs from WTTJ and Hellowork,
# which themselves need JS rendering for their search pages.
# Results have no description → processed via normal URL scraping pipeline.

_JOB_URL_PATTERNS = {
    "WTTJ": re.compile(
        r"welcometothejungle\.com/fr/companies/[^/]+/jobs/[^?&#\s]+"
    ),
    "Hellowork": re.compile(
        r"hellowork\.com/fr-fr/emploi/[^?&#\s]+\.html"
    ),
}


def _extract_ddg_url(href: str) -> str | None:
    """Extract the real destination URL from a DuckDuckGo redirect href."""
    if not href:
        return None
    # DDG HTML uses /l/?uddg=<encoded-url> redirects
    if "uddg=" in href:
        try:
            full = "https://duckduckgo.com" + href if href.startswith("/") else href
            params = parse_qs(urlparse(full).query)
            raw = params.get("uddg", [None])[0]
            return unquote(raw) if raw else None
        except Exception:
            return None
    if href.startswith("http"):
        return href
    return None


def _ddg_job_search(
    query: str,
    location: str,
    n: int,
    site_filter: str,
    source_key: str,
) -> list[dict]:
    """
    Search DuckDuckGo HTML for job listings on a specific site.
    `site_filter` e.g. "welcometothejungle.com"
    `source_key`  e.g. "WTTJ" — used to pick the URL regex validator.
    """
    q = f"site:{site_filter} {query} emploi"
    if location:
        q += f" {location}"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"

    try:
        r = requests.get(url, headers=_DDG_HEADERS, timeout=18)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.content, "html.parser")
    pattern = _JOB_URL_PATTERNS.get(source_key)
    results: list[dict] = []
    seen: set[str] = set()

    for a in soup.select("a.result__a"):
        real = _extract_ddg_url(a.get("href", ""))
        if not real or real in seen:
            continue
        # Validate it's actually a job page (not a company home/blog)
        if pattern and not pattern.search(real):
            continue
        seen.add(real)
        title = a.get_text(strip=True)
        results.append({
            "title":       title,
            "company":     "",
            "url":         real,
            "location":    location,
            "source":      source_key,
            "description": "",   # will be scraped during analysis
        })
        if len(results) >= n:
            break

    return results


def _search_wttj(query: str, location: str, n: int) -> list[dict]:
    """Search WTTJ: Algolia API first, DDG HTML fallback."""
    # 1. Try Algolia (fast path with description)
    algolia = _search_wttj_algolia(query, location, n)
    if algolia:
        return algolia
    # 2. Fallback: DuckDuckGo site search (no description, but works reliably)
    return _ddg_job_search(query, location, n, "welcometothejungle.com", "WTTJ")


def _search_hellowork(query: str, location: str, n: int) -> list[dict]:
    """Search Hellowork via DuckDuckGo site search."""
    return _ddg_job_search(query, location, n, "hellowork.com", "Hellowork")


# ── Algolia (fast path, key extracted dynamically) ────────────────────────────

_wttj_algolia_key_cache: dict = {}


def _get_wttj_algolia_key() -> tuple[str, str] | None:
    """Extract current Algolia App ID + Search API key from WTTJ's live page."""
    now = time.time()
    if _wttj_algolia_key_cache.get("expires_at", 0) > now:
        return _wttj_algolia_key_cache.get("key")

    try:
        resp = requests.get(
            "https://www.welcometothejungle.com/fr/jobs?query=python",
            headers=_HEADERS,
            timeout=18,
        )
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(resp.content, "html.parser")

        # Strategy A: search entire HTML for the 32-char hex Algolia key
        # The App ID CSEKHVMS53 is stable; the key is what rotates.
        _ALGOLIA_APP_ID = "CSEKHVMS53"
        key_candidates = re.findall(r'"([a-f0-9]{32})"', html)
        # Try each candidate against a lightweight Algolia test call
        for candidate in key_candidates:
            if _test_algolia_key(_ALGOLIA_APP_ID, candidate):
                pair = (_ALGOLIA_APP_ID, candidate)
                _wttj_algolia_key_cache["key"]        = pair
                _wttj_algolia_key_cache["expires_at"] = now + 3600
                return pair

        # Strategy B: __NEXT_DATA__ deep search
        nd = soup.find("script", id="__NEXT_DATA__")
        if nd and nd.string:
            d = json.loads(nd.string)
            cfg = _deep_find(d, "algolia") or _deep_find(d, "ALGOLIA")
            if isinstance(cfg, dict):
                app_id  = cfg.get("appId")  or cfg.get("applicationId")
                api_key = cfg.get("apiKey") or cfg.get("searchApiKey")
                if app_id and api_key and _test_algolia_key(app_id, api_key):
                    pair = (app_id, api_key)
                    _wttj_algolia_key_cache["key"]        = pair
                    _wttj_algolia_key_cache["expires_at"] = now + 3600
                    return pair
    except Exception:
        pass

    # Cache the failure for 10 minutes so we don't hammer the site
    _wttj_algolia_key_cache["key"]        = None
    _wttj_algolia_key_cache["expires_at"] = now + 600
    return None


def _test_algolia_key(app_id: str, api_key: str) -> bool:
    """Quick check: does this key work against WTTJ's Algolia index?"""
    try:
        r = requests.post(
            f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/*/queries",
            headers={
                "X-Algolia-Application-Id": app_id,
                "X-Algolia-API-Key":         api_key,
                "Content-Type":              "application/json",
            },
            json={"requests": [{"indexName": "wttj_fr_productions", "query": "a", "params": "hitsPerPage=1"}]},
            timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


def _search_wttj_algolia(query: str, location: str, n: int) -> list[dict]:
    """Call WTTJ Algolia API."""
    creds = _get_wttj_algolia_key()
    if not creds:
        return []
    app_id, api_key = creds
    params_str = f"hitsPerPage={n}&filters=offices.country_code%3AFR"
    if location:
        params_str += f"&aroundQuery={quote_plus(location)}&aroundRadius=50000"
    try:
        r = requests.post(
            f"https://{app_id.lower()}-dsn.algolia.net/1/indexes/*/queries",
            headers={
                "X-Algolia-Application-Id": app_id,
                "X-Algolia-API-Key":         api_key,
                "Content-Type":              "application/json",
            },
            json={"requests": [{"indexName": "wttj_fr_productions", "query": query, "params": params_str}]},
            timeout=15,
        )
        r.raise_for_status()
        hits = r.json()["results"][0].get("hits", [])
    except Exception:
        return []
    results = []
    for hit in hits:
        org      = hit.get("organization") or {}
        org_slug = org.get("slug", "")
        job_slug = hit.get("slug", "")
        if not (org_slug and job_slug):
            continue
        url = f"https://www.welcometothejungle.com/fr/companies/{org_slug}/jobs/{job_slug}"
        offices = hit.get("offices") or []
        loc = ", ".join(o.get("city", "") for o in offices if o.get("city"))
        results.append({
            "title":       hit.get("name") or hit.get("title", ""),
            "company":     org.get("name", ""),
            "url":         url,
            "location":    loc,
            "source":      "WTTJ",
            "description": str(hit.get("description") or "")[:6000],
        })
    return results


def _deep_find(obj, key: str):
    """Recursively search a nested dict/list for a given key."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _deep_find(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find(item, key)
            if found is not None:
                return found
    return None


# ─── Orchestrator ─────────────────────────────────────────────────────────────

_BACKEND_MAP = {
    "Indeed":         _search_indeed,
    "WTTJ":           _search_wttj,
    "Hellowork":      _search_hellowork,
    "France Travail": _search_france_travail,
}


def search_all(
    query: str,
    location: str = "",
    n_per_site: int = 10,
    sites: list[str] | None = None,
) -> list[dict]:
    """
    Search across multiple backends. Returns deduped list of job dicts.
    Each item: {title, company, url, location, source, description}
    """
    if sites is None:
        sites = ["Indeed", "WTTJ"]

    all_results: list[dict] = []
    seen: set[str] = set()

    for site in sites:
        fn = _BACKEND_MAP.get(site)
        if not fn:
            continue
        try:
            for r in fn(query, location, n_per_site):
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    all_results.append(r)
        except Exception:
            continue

    return all_results


def ft_configured() -> bool:
    """Return True if France Travail API credentials are set."""
    return bool(
        os.getenv("FRANCETRAVAIL_CLIENT_ID")
        and os.getenv("FRANCETRAVAIL_CLIENT_SECRET")
    )
