import os
import hashlib
import streamlit as st

# ── Streamlit Cloud secrets bridge ────────────────────────────────────────────
# Pre-load any secrets configured in the Streamlit Cloud dashboard.
# These are then overridable via the in-app "Clés API" panel.
_SECRET_ENV_MAP = {
    "GROQ_API_KEY":                 "GROQ_API_KEY",
    "FRANCETRAVAIL_CLIENT_ID":      "FRANCETRAVAIL_CLIENT_ID",
    "FRANCETRAVAIL_CLIENT_SECRET":  "FRANCETRAVAIL_CLIENT_SECRET",
}
for _sk, _ek in _SECRET_ENV_MAP.items():
    try:
        if _sk in st.secrets and not os.getenv(_ek):
            os.environ[_ek] = st.secrets[_sk]
    except Exception:
        pass

import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
from datetime import datetime
from src.agent import (
    process_offer, run_gap_analysis, get_top_skills,
    load_profile, save_profile, compute_match_score,
    get_skill_match_detail, get_blocking_skills, get_salary_stats,
)
from src.chains import cover_letter_chain, interview_prep_chain, profile_adaptation_chain
from src.utils import offers_to_dataframe
from src.job_search import search_all, SUPPORTED_SITES, ft_configured

st.set_page_config(
    page_title="Job Market Analyzer",
    page_icon="🤖",
    layout="wide",
)

PROFILE_PATH = Path(__file__).parent / "config" / "profile.yaml"
MAX_URLS = 30

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "offers": [],
    "gap_report": None,
    "generated_profile": None,
    "found_jobs": [],       # results from auto-search
    "cfg_groq_key": "",     # in-app Groq API key
    "cfg_ft_id": "",        # in-app France Travail client ID
    "cfg_ft_secret": "",    # in-app France Travail client secret
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Apply in-app keys to os.environ (overrides .env, lower priority than secrets) ──
_KEY_MAP = {
    "cfg_groq_key":  "GROQ_API_KEY",
    "cfg_ft_id":     "FRANCETRAVAIL_CLIENT_ID",
    "cfg_ft_secret": "FRANCETRAVAIL_CLIENT_SECRET",
}
for _ss_key, _env_key in _KEY_MAP.items():
    _val = st.session_state.get(_ss_key, "").strip()
    if _val:
        os.environ[_env_key] = _val

# ── API key check ──────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "**GROQ_API_KEY non configurée.** "
        "Renseigne-la dans le panneau **🔑 Clés API** (barre latérale), "
        "dans ton fichier `.env` (local) ou dans les Secrets Streamlit Cloud.",
        icon="⚠️",
    )

if "profile_yaml" not in st.session_state:
    st.session_state.profile_yaml = PROFILE_PATH.read_text(encoding="utf-8")
if "profile_editor" not in st.session_state:
    st.session_state.profile_editor = st.session_state.profile_yaml


def _offer_key(offer: dict, suffix: str) -> str:
    """Stable session-state key for per-offer generated content."""
    uid = f"{offer.get('title','')}{offer.get('company','')}{offer.get('url','')}"
    return f"{suffix}_{hashlib.md5(uid.encode()).hexdigest()[:8]}"


def _save_profile(content: str) -> None:
    """Update session + try disk (silent fail on Cloud)."""
    st.session_state.profile_yaml = content
    # Can't write widget key after instantiation — stage for next rerun instead
    st.session_state._pending_profile_editor = content
    st.session_state.gap_report = None
    try:
        save_profile(content)
    except Exception:
        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Job Market Analyzer")
    st.caption("Powered by Groq · Llama 3.3 70B")
    st.divider()

    # Profile summary — always visible
    try:
        _p = yaml.safe_load(st.session_state.profile_yaml)
        _n_skills = sum(len(v) for v in _p.get("skills", {}).values() if v)
        st.markdown(f"**{_p.get('name', '—')}**")
        st.caption(
            f"{_p.get('current_role', '—')} → {_p.get('target_role', '—')}  \n"
            f"{_n_skills} compétences · {_p.get('experience_years', '?')} ans XP"
        )
    except Exception:
        st.caption("Profil non chargé")

    st.divider()

    # Clear session
    if st.session_state.offers:
        if st.button("🗑 Vider la session", use_container_width=True):
            st.session_state.offers = []
            st.session_state.gap_report = None
            st.rerun()

    # Profile editor — single expander, two tabs
    with st.expander("⚙️ Mon Profil", expanded=False):
        tab_edit, tab_import = st.tabs(["✏️ Éditer", "📤 Importer"])

        with tab_edit:
            # Apply any externally-staged profile content before the widget renders
            _pending = st.session_state.pop("_pending_profile_editor", None)
            if _pending is not None:
                st.session_state.profile_editor = _pending
            edited_yaml = st.text_area(
                "YAML",
                height=380,
                label_visibility="collapsed",
                key="profile_editor",
            )
            col_s, col_d = st.columns(2)
            with col_s:
                if st.button("💾 Sauvegarder", use_container_width=True, key="btn_save_profile"):
                    try:
                        yaml.safe_load(edited_yaml)
                        _save_profile(edited_yaml)
                        st.success("Sauvegardé !")
                    except yaml.YAMLError as e:
                        st.error(f"YAML invalide : {e}")
            with col_d:
                st.download_button(
                    "⬇️ Exporter",
                    data=st.session_state.profile_yaml,
                    file_name="profile.yaml",
                    mime="text/yaml",
                    use_container_width=True,
                )

        with tab_import:
            st.markdown("**Depuis un fichier YAML**")
            yaml_file = st.file_uploader(
                "Fichier .yaml", type=["yaml", "yml"], key="yaml_upload",
                label_visibility="collapsed",
            )
            if yaml_file is not None:
                try:
                    content = yaml_file.read().decode("utf-8")
                    yaml.safe_load(content)
                    _save_profile(content)
                    st.success("Profil importé !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Fichier invalide : {e}")

            st.divider()
            st.markdown("**Depuis ton CV (PDF natif)**")
            cv_file = st.file_uploader(
                "CV PDF", type=["pdf"], key="cv_upload",
                label_visibility="collapsed",
            )
            if cv_file is not None:
                if st.button("🤖 Extraire le profil", use_container_width=True, key="btn_extract_cv"):
                    with st.spinner("Analyse du CV…"):
                        try:
                            import pypdf
                            reader = pypdf.PdfReader(cv_file)
                            cv_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                            if not cv_text.strip():
                                st.error("Impossible d'extraire le texte (PDF scanné ?).")
                            else:
                                st.session_state._last_cv_text = cv_text
                                if len(cv_text) < 300:
                                    st.warning(f"Texte court ({len(cv_text)} cars) — PDF scanné ou protégé ?")
                                from src.chains import cv_to_profile_chain
                                st.session_state.generated_profile = cv_to_profile_chain(cv_text)
                                st.success("✅ Profil extrait ! Ferme ce panneau et consulte le bandeau de révision dans la page principale.")
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                if "_last_cv_text" in st.session_state:
                    with st.expander(f"📄 Texte brut extrait du PDF ({len(st.session_state._last_cv_text)} cars)", expanded=False):
                        st.text(st.session_state._last_cv_text[:2000])

    # ── API Keys panel ────────────────────────────────────────────────────────
    _groq_set = bool(os.getenv("GROQ_API_KEY"))
    _ft_set   = ft_configured()
    _key_icon = "🔑" if (_groq_set and _ft_set) else "🔑⚠️"
    with st.expander(f"{_key_icon} Clés API", expanded=not _groq_set):
        st.caption(
            "Les clés sont stockées uniquement dans ta session (non persistées). "
            "Elles prennent la priorité sur le fichier `.env`."
        )

        # Groq
        _groq_status = "✅ configurée" if _groq_set else "❌ manquante"
        st.markdown(f"**Groq API Key** — {_groq_status}")
        _new_groq = st.text_input(
            "Groq API Key",
            value=st.session_state.cfg_groq_key,
            type="password",
            placeholder="gsk_...",
            key="input_groq_key",
            label_visibility="collapsed",
        )

        st.divider()

        # France Travail
        _ft_status = "✅ configurée" if _ft_set else "❌ non configurée"
        st.markdown(f"**France Travail** — {_ft_status}")
        st.caption(
            "Inscription gratuite : [francetravail.io/data/api](https://francetravail.io/data/api)"
        )
        _new_ft_id = st.text_input(
            "Client ID",
            value=st.session_state.cfg_ft_id,
            type="password",
            placeholder="Client ID",
            key="input_ft_id",
        )
        _new_ft_secret = st.text_input(
            "Client Secret",
            value=st.session_state.cfg_ft_secret,
            type="password",
            placeholder="Client Secret",
            key="input_ft_secret",
        )

        if st.button("💾 Appliquer les clés", use_container_width=True, type="primary", key="btn_apply_keys"):
            _changed = False
            if _new_groq.strip() != st.session_state.cfg_groq_key:
                st.session_state.cfg_groq_key = _new_groq.strip()
                _changed = True
            if _new_ft_id.strip() != st.session_state.cfg_ft_id:
                st.session_state.cfg_ft_id = _new_ft_id.strip()
                _changed = True
            if _new_ft_secret.strip() != st.session_state.cfg_ft_secret:
                st.session_state.cfg_ft_secret = _new_ft_secret.strip()
                _changed = True
            if _changed:
                # Apply immediately to os.environ for the rest of this run
                for _ss_key, _env_key in _KEY_MAP.items():
                    _v = st.session_state.get(_ss_key, "").strip()
                    if _v:
                        os.environ[_env_key] = _v
                st.success("Clés appliquées !")
                st.rerun()
            else:
                st.info("Aucun changement détecté.")

# ── CV review banner — main area, above everything ────────────────────────────
if st.session_state.generated_profile:
    with st.container(border=True):
        st.subheader("🆕 Profil extrait depuis ton CV")
        st.caption(
            "L'IA a analysé ton CV. Relis, corrige si besoin, puis applique."
        )
        reviewed = st.text_area(
            "Profil extrait",
            value=st.session_state.generated_profile,
            height=280,
            key="cv_review",
            label_visibility="collapsed",
        )
        c_apply, c_cancel = st.columns(2)
        with c_apply:
            if st.button("✅ Appliquer ce profil", type="primary", use_container_width=True):
                try:
                    yaml.safe_load(reviewed)
                    _save_profile(reviewed)
                    st.session_state.generated_profile = None
                    st.success("Profil appliqué !")
                    st.rerun()
                except yaml.YAMLError as e:
                    st.error(f"YAML invalide : {e}")
        with c_cancel:
            if st.button("✕ Ignorer", use_container_width=True):
                st.session_state.generated_profile = None
                st.rerun()
    st.divider()

# ── Main title ────────────────────────────────────────────────────────────────
st.title("🔍 Analyseur d'Offres d'Emploi")
st.markdown(
    "Analyse des offres en lot, détecte les lacunes par rapport à ton profil, "
    "identifie les tendances du marché."
)
st.divider()

# ── Onboarding hint — visible only before first analysis ──────────────────────
if not st.session_state.offers:
    st.info(
        "**Pour commencer :**  \n"
        "**1.** Configure **ton** profil → ouvre ⚙️ *Mon Profil* dans le panneau latéral  \n"
        "**2.** Recherche des offres automatiquement **ou** colle des URLs manuellement  \n"
        "**3.** Clique **Analyser** — les résultats avec score de compatibilité apparaîtront ici",
        icon="👋",
    )

# ── Auto-search ────────────────────────────────────────────────────────────────
try:
    _sp = yaml.safe_load(st.session_state.profile_yaml)
    _default_query    = _sp.get("target_role", "")
    _default_location = (_sp.get("job_preferences") or {}).get("location", "")
except Exception:
    _default_query    = ""
    _default_location = ""

with st.expander("🔍 Recherche automatique d'offres", expanded=not bool(st.session_state.found_jobs)):
    st.caption("Cherche des offres en direct sur Indeed et WTTJ — et sur France Travail si configuré.")

    _ft_ok = ft_configured()
    _default_sites = ["Indeed", "WTTJ", "Hellowork"] + (["France Travail"] if _ft_ok else [])

    if not _ft_ok:
        st.info(
            "💡 **Bonus :** active **France Travail** (Pôle Emploi) pour des milliers d'offres supplémentaires.  \n"
            "Inscription gratuite → [francetravail.io/data/api](https://francetravail.io/data/api)  \n"
            "Puis renseigne ton *Client ID* et *Client Secret* dans le panneau **🔑 Clés API** de la barre latérale.",
            icon="🔑",
        )

    col_q, col_l = st.columns([2, 1])
    with col_q:
        search_query = st.text_input(
            "Mots-clés (poste, technologie…)",
            value=_default_query,
            placeholder="ex. Data Engineer, React, Développeur Python…",
            key="search_query",
        )
    with col_l:
        search_location = st.text_input(
            "Lieu (optionnel)",
            value=_default_location,
            placeholder="Paris, Lyon, Remote…",
            key="search_location",
        )
    col_n, col_sites = st.columns([1, 2])
    with col_n:
        n_results = st.select_slider(
            "Nb d'offres par source",
            options=[5, 10, 15, 20],
            value=10,
            key="n_results",
        )
    with col_sites:
        _site_labels = [
            s if s != "France Travail" or _ft_ok else "France Travail ⚙️"
            for s in SUPPORTED_SITES
        ]
        chosen_sites_raw = st.multiselect(
            "Sources",
            _site_labels,
            default=_default_sites,
            key="search_sites",
        )
        # Strip the ⚙️ label before passing to backend
        chosen_sites = [s.replace(" ⚙️", "") for s in chosen_sites_raw]

    if st.button("🔍 Chercher des offres", type="primary", use_container_width=True, key="btn_search"):
        if not search_query.strip():
            st.warning("Renseigne au moins un mot-clé pour lancer la recherche.")
        elif not chosen_sites:
            st.warning("Sélectionne au moins une source.")
        else:
            with st.spinner(f"Recherche sur {', '.join(chosen_sites)}…"):
                _found = search_all(
                    search_query.strip(),
                    search_location.strip(),
                    n_per_site=n_results,
                    sites=chosen_sites,
                )
            st.session_state.found_jobs = _found
            if _found:
                n_with_desc = sum(1 for j in _found if j.get("description"))
                st.success(
                    f"✅ **{len(_found)} offre(s) trouvée(s)** "
                    f"({n_with_desc} avec description — analyse directe sans scraping). "
                    f"Clique **Analyser** pour les traiter."
                )
            else:
                st.warning(
                    "Aucune offre trouvée. "
                    "Essaie d'autres mots-clés, change le lieu, ou colle des URLs manuellement ci-dessous."
                )

# ── Found-jobs preview ─────────────────────────────────────────────────────────
if st.session_state.found_jobs:
    _fj = st.session_state.found_jobs
    st.caption(f"🔎 **{len(_fj)} offre(s) prête(s) à analyser** (trouvées via la recherche automatique)")
    _df_found = pd.DataFrame(_fj)[["title", "company", "location", "source", "url"]]
    _df_found.columns = ["Titre", "Entreprise", "Lieu", "Source", "Lien"]
    st.dataframe(
        _df_found,
        use_container_width=True,
        hide_index=True,
        column_config={"Lien": st.column_config.LinkColumn("Lien", display_text="🔗")},
    )
    if st.button("✕ Effacer ces résultats", key="btn_clear_search"):
        st.session_state.found_jobs = []
        st.rerun()
    st.divider()

# ── Manual URL input ───────────────────────────────────────────────────────────
urls_input = st.text_area(
    "URLs supplémentaires (une par ligne · max 30)",
    height=110,
    placeholder=(
        "https://www.welcometothejungle.com/fr/companies/.../jobs/...\n"
        "https://www.hellowork.com/fr-fr/emploi/...\n"
        "https://www.apec.fr/candidat/recherche-emploi.html/emploi/..."
    ),
)

with st.expander("➕ Coller du texte d'offre manuellement (LinkedIn, Indeed…)"):
    st.caption("Ouvre l'offre dans ton navigateur, Ctrl+A → Ctrl+C, colle ici.")
    manual_text = st.text_area(
        "Texte de l'offre",
        height=140,
        label_visibility="collapsed",
        placeholder="Colle le contenu complet de l'offre ici…",
    )

run_btn = st.button("🚀 Analyser", type="primary", use_container_width=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_btn:
    # Warn if profile looks unconfigured (still has default placeholder name)
    try:
        _check_p = yaml.safe_load(st.session_state.profile_yaml)
        if _check_p.get("name", "").strip() in ("Full Name", "Badreddine EL KHAMLICHI", ""):
            st.warning(
                "Le profil semble ne pas être configuré (nom par défaut détecté). "
                "Les scores de compatibilité seront calculés contre ce profil. "
                "Configure ton profil via ⚙️ *Mon Profil* dans le panneau latéral.",
                icon="⚠️",
            )
    except Exception:
        pass

    # Build ordered deduped list of (url, prefetched_text_or_None)
    # Found-jobs first (may carry description from jobspy → no re-scraping)
    _items: list[tuple[str, str | None]] = []
    _seen_u: set[str] = set()
    for _j in st.session_state.found_jobs:
        _u = _j.get("url", "").strip()
        if _u and _u not in _seen_u:
            _seen_u.add(_u)
            _items.append((_u, _j.get("description") or None))
    # Manual URLs (no prefetched text)
    _manual_urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    for _u in _manual_urls:
        if _u not in _seen_u:
            _seen_u.add(_u)
            _items.append((_u, None))

    # Validate URL format only for items that need scraping (no prefetched text)
    invalid = [u for u, t in _items if t is None and not u.startswith(("http://", "https://"))]

    if invalid:
        st.error(f"URLs invalides (doivent commencer par http:// ou https://) : {', '.join(invalid)}")
    elif not _items and not manual_text.strip():
        st.warning("Lance une recherche automatique ou colle au moins une URL / du texte d'offre.")
    else:
        if len(_items) > MAX_URLS:
            st.warning(f"Seules les {MAX_URLS} premières offres sont traitées.")
            _items = _items[:MAX_URLS]

        results: list[dict] = []

        if manual_text.strip():
            from src.chains import analyze_offer_chain
            with st.spinner("Analyse du texte collé…"):
                r = analyze_offer_chain(manual_text.strip())
            if r:
                r.update({"url": "Texte manuel", "status": "success"})
                results.append(r)
            else:
                results.append({"error": "Echec du parsing LLM", "url": "Texte manuel"})

        if _items:
            _n_direct = sum(1 for _, t in _items if t)
            _n_scrape = len(_items) - _n_direct
            _label = f"0 / {len(_items)} offre(s) analysée(s)"
            if _n_direct:
                _label += f" ({_n_direct} sans scraping · {_n_scrape} à scraper)"
            progress = st.progress(0, text=_label)
            scraped: list[dict | None] = [None] * len(_items)
            with ThreadPoolExecutor(max_workers=4) as executor:
                fut_idx = {
                    executor.submit(process_offer, url, text): i
                    for i, (url, text) in enumerate(_items)
                }
                done = 0
                for fut in as_completed(fut_idx):
                    try:
                        scraped[fut_idx[fut]] = fut.result()
                    except Exception as exc:
                        idx = fut_idx[fut]
                        scraped[idx] = {"error": str(exc), "url": _items[idx][0]}
                    done += 1
                    progress.progress(done / len(_items), text=f"{done} / {len(_items)} offre(s) analysée(s)…")
            progress.empty()
            results.extend(scraped)

        # Compute per-offer match scores
        try:
            _profile = yaml.safe_load(st.session_state.profile_yaml)
        except Exception:
            _profile = None
        if _profile:
            for o in results:
                if o and o.get("status") == "success":
                    try:
                        o["match_score"] = compute_match_score(o, _profile)
                    except Exception:
                        pass

        st.session_state.offers = results
        st.session_state.gap_report = None
        n_ok = sum(1 for o in results if o and o.get("status") == "success")
        n_err = sum(1 for o in results if o and o.get("error"))
        msg = f"**{n_ok} offre(s) analysée(s) avec succès.**"
        if n_err:
            msg += f"  {n_err} erreur(s) — voir l'onglet Offres."
        st.success(msg)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.offers:
    offers = st.session_state.offers
    successful = [o for o in offers if o and o.get("status") == "success"]
    errors = [o for o in offers if o and o.get("error")]

    n_ok = len(successful)
    _tab_offers  = f"📋 Offres ({n_ok})" if n_ok else "📋 Offres"
    _tab_report  = "🎯 Mon Rapport ✓" if st.session_state.gap_report else "🎯 Mon Rapport"
    _tab_export  = "💾 Export"

    st.caption("① Consulte les offres et les tendances · ② Génère ton rapport de compatibilité · ③ Exporte")

    tab1, tab2, tab3 = st.tabs([_tab_offers, _tab_report, _tab_export])

    # ── Tab 1 : Offres ────────────────────────────────────────────────────────
    with tab1:
        if errors:
            with st.expander(f"⚠️ {len(errors)} erreur(s) de scraping"):
                for e in errors:
                    st.error(f"**{e['url']}**  \n{e['error']}")

        if not successful:
            st.info("Aucune offre analysée avec succès. Vérifie les erreurs ci-dessus.")
        else:
            # ── Tiers de candidature ──
            has_scores = any("match_score" in o for o in successful)
            if has_scores:
                easy   = [o for o in successful if (o.get("match_score") or 0) >= 70]
                medium = [o for o in successful if 40 <= (o.get("match_score") or 0) < 70]
                hard   = [o for o in successful if (o.get("match_score") or 0) < 40]
                t1, t2, t3 = st.columns(3)
                t1.metric("🟢 Candidature directe", len(easy),   help="Match ≥ 70% — postule maintenant")
                t2.metric("🟡 Avec préparation",    len(medium), help="Match 40–70% — quelques compétences à travailler")
                t3.metric("🔴 Objectif long terme", len(hard),   help="Match < 40% — investment formation nécessaire")

            # ── Salary benchmarking ──
            salary_stats = get_salary_stats(successful)
            if salary_stats:
                s_min, s_max, s_avg = salary_stats["min"]//1000, salary_stats["max"]//1000, salary_stats["avg"]//1000
                bench_msg = f"💰 **Salaires dans ces offres :** {s_min}k – {s_max}k€ · moyenne **{s_avg}k€**"
                try:
                    _sp = yaml.safe_load(st.session_state.profile_yaml)
                    _prefs = _sp.get("job_preferences", {})
                    _pmin = (_prefs.get("min_salary") or 0) // 1000
                    _pmax = (_prefs.get("max_salary") or 0) // 1000
                    if _pmin:
                        bench_msg += f"  ·  Ton attente : {_pmin}k – {_pmax}k€"
                        if salary_stats["avg"] < _prefs.get("min_salary", 0):
                            bench_msg += "  ⚠️ *marché en dessous de ton attente min*"
                        elif salary_stats["avg"] > (_prefs.get("max_salary") or 0):
                            bench_msg += "  ✅ *marché au-dessus de ton attente max*"
                except Exception:
                    pass
                st.caption(bench_msg)

            st.divider()

            # ── Filters ──
            all_contracts = sorted({o.get("contract_type") or "N/A" for o in successful})
            all_locations = sorted({o.get("location")      or "N/A" for o in successful})

            fc, fl, fs = st.columns([2, 2, 1])
            with fc:
                sel_contracts = st.multiselect("Contrat", all_contracts, default=all_contracts)
            with fl:
                sel_locations = st.multiselect("Lieu", all_locations, default=all_locations)
            with fs:
                min_score = st.slider("Match min %", 0, 100, 0, step=5) if has_scores else 0

            filtered = [
                o for o in successful
                if (o.get("contract_type") or "N/A") in sel_contracts
                and (o.get("location")      or "N/A") in sel_locations
                and (o.get("match_score")   or 0)     >= min_score
            ]

            st.caption(f"{len(filtered)} / {len(successful)} offre(s) affichée(s)")

            # ── Table ── compact summary (detail is in the cards below)
            df_display = pd.DataFrame([{
                "Titre": o.get("title", "N/A"),
                "Entreprise": o.get("company", "N/A"),
                "Lieu": o.get("location", "N/A"),
                "Contrat": o.get("contract_type", "N/A"),
                "Niveau": o.get("level", "N/A"),
                "Salaire": o.get("salary_range", "N/A"),
                "Lien": o.get("url", ""),
            } for o in filtered])
            if has_scores:
                df_display.insert(0, "Match %", [o.get("match_score", 0) for o in filtered])
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Lien": st.column_config.LinkColumn("Lien", display_text="🔗 Voir"),
                },
            )

            st.divider()

            # ── Detail cards, sorted by score ──
            try:
                _card_profile = yaml.safe_load(st.session_state.profile_yaml)
            except Exception:
                _card_profile = None

            for offer in sorted(filtered, key=lambda o: o.get("match_score", 0), reverse=True):
                score = offer.get("match_score")
                label = f"**{offer.get('title', 'N/A')}** — {offer.get('company', 'N/A')}"
                if score is not None:
                    label += f" · {score}%"
                with st.expander(label):
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"**📍 Lieu :** {offer.get('location', 'N/A')}")
                        st.markdown(f"**📄 Contrat :** {offer.get('contract_type', 'N/A')}")
                        st.markdown(f"**⏳ Expérience :** {offer.get('experience_required', 'N/A')}")
                        st.markdown(f"**💶 Salaire :** {offer.get('salary_range', 'N/A')}")
                    with cb:
                        st.markdown(f"**🎓 Niveau :** {offer.get('level', 'N/A')}")
                        stack = ", ".join(s for s in (offer.get("tech_stack") or []) if s)
                        st.markdown(f"**🛠 Stack :** {stack or 'N/A'}")
                        if score is not None:
                            st.progress(score / 100, text=f"Match profil : {score}%")
                    # ── Matched / missing skills ──
                    if _card_profile and score is not None:
                        matched, missing = get_skill_match_detail(offer, _card_profile)
                        sm, smi = st.columns(2)
                        with sm:
                            if matched:
                                st.markdown(
                                    "**✅ Compétences matchées**\n" +
                                    "".join(f"- {s}\n" for s in matched)
                                )
                        with smi:
                            if missing:
                                st.markdown(
                                    "**❌ À acquérir**\n" +
                                    "".join(f"- {s}\n" for s in missing)
                                )
                    # ── URL ──
                    offer_url = offer.get("url", "")
                    if offer_url and offer_url.startswith("http"):
                        st.markdown(f"[🔗 Voir l'offre originale]({offer_url})")
                    st.markdown("**📝 Résumé**")
                    st.info(offer.get("summary", "N/A"))
                    missions = offer.get("missions") or []
                    if missions:
                        st.markdown("**✅ Missions**")
                        for m in missions:
                            st.markdown(f"- {m}")

                    # ── AI actions ──
                    if _card_profile:
                        st.divider()
                        _cl_key = _offer_key(offer, "cl")
                        _ip_key = _offer_key(offer, "ip")
                        _pa_key = _offer_key(offer, "pa")
                        ab1, ab2, ab3 = st.columns(3)
                        with ab1:
                            if st.button("📝 Lettre de motivation", key=f"btn_{_cl_key}", use_container_width=True):
                                with st.spinner("Rédaction…"):
                                    st.session_state[_cl_key] = cover_letter_chain(offer, _card_profile)
                        with ab2:
                            if st.button("❓ Prép. entretien", key=f"btn_{_ip_key}", use_container_width=True):
                                with st.spinner("Génération des questions…"):
                                    st.session_state[_ip_key] = interview_prep_chain(offer, _card_profile)
                        with ab3:
                            if st.button("🎯 Adapter mon profil", key=f"btn_{_pa_key}", use_container_width=True):
                                with st.spinner("Analyse…"):
                                    st.session_state[_pa_key] = profile_adaptation_chain(offer, _card_profile)
                        if _cl_key in st.session_state:
                            with st.expander("📝 Lettre de motivation", expanded=True):
                                st.text_area(
                                    "",
                                    value=st.session_state[_cl_key],
                                    height=280,
                                    key=f"ta_{_cl_key}",
                                    label_visibility="collapsed",
                                )
                                if st.button("🔄 Regénérer", key=f"regen_{_cl_key}"):
                                    del st.session_state[_cl_key]
                                    st.rerun()
                        if _ip_key in st.session_state:
                            with st.expander("❓ Questions d'entretien", expanded=True):
                                st.markdown(st.session_state[_ip_key])
                        if _pa_key in st.session_state:
                            with st.expander("🎯 Conseils d'adaptation du profil", expanded=True):
                                st.markdown(st.session_state[_pa_key])

        # ── Tendances marché — dans le même onglet, après les cards ──
        if successful:
            st.divider()
            with st.expander("📊 Tendances marché — compétences les plus demandées", expanded=False):
                top_skills = get_top_skills(successful, top_n=15)
                if top_skills:
                    skills_df = (
                        pd.DataFrame(top_skills, columns=["Compétence", "Occurrences"])
                        .sort_values("Occurrences", ascending=False)
                    )
                    st.bar_chart(skills_df.set_index("Compétence"))
                    st.dataframe(skills_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Pas assez de données pour afficher les tendances.")

                # ── Compétences bloquantes ──
                if _card_profile:
                    st.divider()
                    st.subheader("🚨 Compétences bloquantes")
                    st.caption("Compétences absentes de ton profil, classées par fréquence dans les offres analysées.")
                    blocking = get_blocking_skills(successful, _card_profile)
                    missing = [b for b in blocking if not b["in_profile"]]
                    present = [b for b in blocking if b["in_profile"]]
                    if missing:
                        for b in missing[:10]:
                            col_s, col_b = st.columns([2, 3])
                            with col_s:
                                st.markdown(f"**{b['skill']}**")
                            with col_b:
                                st.progress(b["pct"] / 100, text=f"{b['pct']}% des offres ({b['count']}/{b['total']})")
                    else:
                        st.success("Tu couvres toutes les compétences fréquentes !")
                    if present:
                        with st.expander(f"✅ {len(present)} compétences fréquentes déjà maîtrisées"):
                            st.write(", ".join(b["skill"] for b in present))

    # ── Tab 2 : Mon Rapport ────────────────────────────────────────────────────
    with tab2:
        if not successful:
            st.info("Lance une analyse d'abord pour générer le rapport.")
        else:
            # Profile summary
            try:
                _p = yaml.safe_load(st.session_state.profile_yaml)
                _all_s = [s for cat in _p.get("skills", {}).values() if cat for s in cat if s]
                def _t(s, n=22): return s[:n] + "…" if len(s) > n else s
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Nom", _t(_p.get("name", "—").split()[0], 18))
                m2.metric("Actuel", _t(_p.get("current_role", "—")))
                m3.metric("Cible", _t(_p.get("target_role", "—")))
                m4.metric("Compétences", len(_all_s))
            except Exception:
                pass

            st.divider()

            # Generate button — only shown when no report yet
            if not st.session_state.gap_report:
                st.subheader("🎯 Analyse des lacunes vs le marché")
                if st.button("🧠 Générer mon rapport", type="primary", use_container_width=True):
                    with st.spinner("L'IA analyse ton profil vs le marché…"):
                        try:
                            _profile = yaml.safe_load(st.session_state.profile_yaml)
                        except Exception:
                            _profile = load_profile()
                        st.session_state.gap_report = run_gap_analysis(successful, profile=_profile)
                    # No st.rerun() — second if block below renders the report immediately

            # Report display — checked independently so it renders in the same pass
            if st.session_state.gap_report:
                c_title, c_reset = st.columns([5, 1])
                with c_title:
                    st.subheader("🎯 Rapport de compatibilité")
                with c_reset:
                    if st.button("🔄", help="Régénérer le rapport"):
                        st.session_state.gap_report = None
                        st.rerun()
                st.markdown(st.session_state.gap_report)

    # ── Tab 3 : Export ────────────────────────────────────────────────────────
    with tab3:
        # ── Export data ──
        if successful:
            st.subheader("📥 Exporter les données")
            c1, c2 = st.columns(2)
            with c1:
                df_exp = offers_to_dataframe(successful)
                if any("match_score" in o for o in successful):
                    df_exp.insert(0, "Match %", [o.get("match_score", 0) for o in successful])
                st.download_button(
                    "⬇️ CSV",
                    data=df_exp.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="job_offers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "⬇️ JSON",
                    data=json.dumps(successful, ensure_ascii=False, indent=2),
                    file_name="job_offers.json",
                    mime="application/json",
                    use_container_width=True,
                )
            st.divider()

        # ── Session save / restore ──
        st.subheader("🔄 Session")
        st.caption(
            "Sauvegarde ta session dans un fichier JSON pour éviter de tout re-analyser "
            "à la prochaine visite. Recharge-la via le chargeur ci-dessous."
        )
        cs, cr = st.columns(2)
        with cs:
            if successful:
                session_data = {
                    "version": 1,
                    "exported_at": datetime.now().isoformat(),
                    "offers": st.session_state.offers,
                }
                st.download_button(
                    "💾 Sauvegarder la session",
                    data=json.dumps(session_data, ensure_ascii=False, indent=2),
                    file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Lance une analyse pour pouvoir sauvegarder une session.", icon="ℹ️")
        with cr:
            restore_file = st.file_uploader(
                "📂 Charger une session (.json)",
                type=["json"],
                key="session_restore",
                label_visibility="visible",
            )
            if restore_file is not None:
                try:
                    data = json.loads(restore_file.read().decode("utf-8"))
                    if "offers" not in data:
                        st.error("Fichier invalide (clé 'offers' manquante).")
                    else:
                        st.session_state.offers = data["offers"]
                        st.session_state.gap_report = None
                        # Recompute match scores against current profile
                        try:
                            _rp = yaml.safe_load(st.session_state.profile_yaml)
                            for _o in st.session_state.offers:
                                if _o and _o.get("status") == "success":
                                    try:
                                        _o["match_score"] = compute_match_score(_o, _rp)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        n = sum(1 for o in data["offers"] if o and o.get("status") == "success")
                        st.success(f"Session restaurée — {n} offre(s) chargée(s).")
                        st.rerun()
                except Exception as e:
                    st.error(f"Erreur de lecture : {e}")
