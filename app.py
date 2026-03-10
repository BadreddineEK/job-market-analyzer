import os
import streamlit as st

# ── Streamlit Cloud secrets bridge ────────────────────────────────────────────
try:
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
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
)
from src.utils import offers_to_dataframe

st.set_page_config(
    page_title="Job Market Analyzer",
    page_icon="🤖",
    layout="wide",
)

PROFILE_PATH = Path(__file__).parent / "config" / "profile.yaml"
MAX_URLS = 20

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "offers": [],
    "gap_report": None,
    "generated_profile": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if "profile_yaml" not in st.session_state:
    st.session_state.profile_yaml = PROFILE_PATH.read_text(encoding="utf-8")


def _save_profile(content: str) -> None:
    """Update session + try disk (silent fail on Cloud)."""
    st.session_state.profile_yaml = content
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
        _n_skills = sum(len(v) for v in _p.get("skills", {}).values())
        st.markdown(f"**{_p.get('name', '—')}**")
        st.caption(
            f"{_p.get('current_role', '—')} → {_p.get('target_role', '—')}  \n"
            f"{_n_skills} compétences · {_p.get('experience_years', '?')} ans XP"
        )
    except Exception:
        st.caption("Profil non chargé")

    st.divider()

    # Profile editor — single expander, two tabs
    with st.expander("⚙️ Mon Profil", expanded=bool(st.session_state.generated_profile is None and False)):
        tab_edit, tab_import = st.tabs(["✏️ Éditer", "📤 Importer"])

        with tab_edit:
            edited_yaml = st.text_area(
                "YAML",
                value=st.session_state.profile_yaml,
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
                                from src.chains import cv_to_profile_chain
                                st.session_state.generated_profile = cv_to_profile_chain(cv_text)
                                st.success("Profil extrait — consulte la zone de révision ci-dessous.")
                        except Exception as e:
                            st.error(f"Erreur : {e}")

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

# ── Input ─────────────────────────────────────────────────────────────────────
urls_input = st.text_area(
    "URLs des offres d'emploi (une par ligne · max 20)",
    height=140,
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
    raw_urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    invalid = [u for u in raw_urls if not u.startswith(("http://", "https://"))]

    if invalid:
        st.error(f"URLs invalides (doivent commencer par http:// ou https://) : {', '.join(invalid)}")
    elif not raw_urls and not manual_text.strip():
        st.warning("Colle au moins une URL ou du texte d'offre.")
    else:
        urls = raw_urls[:MAX_URLS]
        if len(raw_urls) > MAX_URLS:
            st.warning(f"Seules les {MAX_URLS} premières URLs sont traitées.")

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

        if urls:
            progress = st.progress(0, text=f"0 / {len(urls)} offre(s) analysée(s)…")
            scraped: list[dict | None] = [None] * len(urls)
            with ThreadPoolExecutor(max_workers=4) as executor:
                fut_idx = {executor.submit(process_offer, u): i for i, u in enumerate(urls)}
                done = 0
                for fut in as_completed(fut_idx):
                    scraped[fut_idx[fut]] = fut.result()
                    done += 1
                    progress.progress(done / len(urls), text=f"{done} / {len(urls)} offre(s) analysée(s)…")
            progress.empty()
            results.extend(scraped)

        # Compute per-offer match scores
        try:
            _profile = yaml.safe_load(st.session_state.profile_yaml)
            for o in results:
                if o and o.get("status") == "success":
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
    _tab_offers   = f"📋 Offres ({n_ok})" if n_ok else "📋 Offres"
    _tab_trends   = "📊 Tendances"
    _tab_gap      = "🎯 Gap Analysis ✓" if st.session_state.gap_report else "🎯 Gap Analysis"
    _tab_export   = "💾 Export"

    tab1, tab2, tab3, tab4 = st.tabs([_tab_offers, _tab_trends, _tab_gap, _tab_export])

    # ── Tab 1 : Offres ────────────────────────────────────────────────────────
    with tab1:
        if errors:
            with st.expander(f"⚠️ {len(errors)} erreur(s) de scraping"):
                for e in errors:
                    st.error(f"**{e['url']}**  \n{e['error']}")

        if not successful:
            st.info("Aucune offre analysée avec succès. Vérifie les erreurs ci-dessus.")
        else:
            # ── Filters ──
            all_contracts = sorted({o.get("contract_type", "N/A") for o in successful})
            all_locations = sorted({o.get("location", "N/A") for o in successful})
            has_scores = any("match_score" in o for o in successful)

            fc, fl, fs = st.columns([2, 2, 1])
            with fc:
                sel_contracts = st.multiselect("Contrat", all_contracts, default=all_contracts)
            with fl:
                sel_locations = st.multiselect("Lieu", all_locations, default=all_locations)
            with fs:
                min_score = st.slider("Match min %", 0, 100, 0, step=5) if has_scores else 0

            filtered = [
                o for o in successful
                if o.get("contract_type", "N/A") in sel_contracts
                and o.get("location", "N/A") in sel_locations
                and o.get("match_score", 0) >= min_score
            ]

            st.caption(f"{len(filtered)} / {len(successful)} offre(s) affichée(s)")

            # ── Table ──
            df = offers_to_dataframe(filtered)
            if has_scores:
                df.insert(0, "Match %", [o.get("match_score", 0) for o in filtered])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()

            # ── Detail cards, sorted by score ──
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
                        st.markdown(f"**🛠 Stack :** {', '.join(offer.get('tech_stack', []))}")
                        if score is not None:
                            st.progress(score / 100, text=f"Match profil : {score}%")
                    st.markdown("**📝 Résumé**")
                    st.info(offer.get("summary", "N/A"))
                    missions = offer.get("missions", [])
                    if missions:
                        st.markdown("**✅ Missions**")
                        for m in missions:
                            st.markdown(f"- {m}")

    # ── Tab 2 : Tendances ─────────────────────────────────────────────────────
    with tab2:
        if not successful:
            st.info("Lance une analyse pour voir les tendances.")
        else:
            st.subheader("🔥 Compétences les plus demandées")
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

    # ── Tab 3 : Gap Analysis ──────────────────────────────────────────────────
    with tab3:
        if not successful:
            st.info("Lance une analyse d'abord pour générer le rapport.")
        else:
            # Profile summary
            try:
                _p = yaml.safe_load(st.session_state.profile_yaml)
                _all_s = [s for cat in _p.get("skills", {}).values() for s in cat]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Nom", _p.get("name", "—").split()[0])
                m2.metric("Rôle actuel", _p.get("current_role", "—"))
                m3.metric("Cible", _p.get("target_role", "—"))
                m4.metric("Compétences", len(_all_s))
            except Exception:
                pass

            st.divider()

            if not st.session_state.gap_report:
                st.subheader("🎯 Analyse des lacunes vs le marché")
                if st.button("🧠 Générer le rapport", type="primary", use_container_width=True):
                    with st.spinner("L'IA analyse ton profil vs le marché…"):
                        try:
                            _profile = yaml.safe_load(st.session_state.profile_yaml)
                        except Exception:
                            _profile = load_profile()
                        st.session_state.gap_report = run_gap_analysis(successful, profile=_profile)
                    st.rerun()
            else:
                c_title, c_reset = st.columns([5, 1])
                with c_title:
                    st.subheader("🎯 Rapport de compétences")
                with c_reset:
                    if st.button("🔄", help="Régénérer le rapport"):
                        st.session_state.gap_report = None
                        st.rerun()
                st.markdown(st.session_state.gap_report)

    # ── Tab 4 : Export ────────────────────────────────────────────────────────
    with tab4:
        if successful:
            st.subheader("📥 Exporter les résultats")
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
        st.subheader("🔄 Session")
        st.caption("Sauvegarde et recharge une session pour éviter de tout re-analyser.")

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
                    n = sum(1 for o in data["offers"] if o and o.get("status") == "success")
                    st.success(f"Session restaurée — {n} offre(s) chargée(s).")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")
