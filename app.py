import os
import streamlit as st

# ── Streamlit Cloud secrets bridge ────────────────────────────────────────────
# Must run before any src import that calls load_dotenv / os.getenv
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
from src.agent import process_offer, run_gap_analysis, get_top_skills, load_profile, save_profile, compute_match_score
from src.utils import offers_to_dataframe

st.set_page_config(
    page_title="Job Market Analyzer",
    page_icon="🤖",
    layout="wide",
)

PROFILE_PATH = Path(__file__).parent / "config" / "profile.yaml"
MAX_URLS = 20

# ── Session state init ─────────────────────────────────────────────────────────
if "offers" not in st.session_state:
    st.session_state.offers = []
if "gap_report" not in st.session_state:
    st.session_state.gap_report = None
if "profile_yaml" not in st.session_state:
    st.session_state.profile_yaml = PROFILE_PATH.read_text(encoding="utf-8")
if "generated_profile" not in st.session_state:
    st.session_state.generated_profile = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Job Market Analyzer")
    st.caption("Powered by LangChain + Groq (Llama 3.3 70B)")
    st.divider()

    st.subheader("📋 Comment utiliser")
    st.markdown("""
1. Édite ton profil ci-dessous
2. Colle les URLs d'offres (une par ligne, max 20)
3. Clique sur **Analyser**
4. Consulte résultats, tendances et rapport de compétences
5. Exporte en CSV ou JSON

> **LinkedIn / Indeed** : colle le texte de l'offre directement.
    """)

    st.divider()

    # ── Profile editor ────────────────────────────────────────────────────────
    with st.expander("🧑 Mon Profil", expanded=False):
        edited_yaml = st.text_area(
            "Profil YAML",
            value=st.session_state.profile_yaml,
            height=420,
            label_visibility="collapsed",
            key="profile_editor",
        )

        col_s, col_d = st.columns(2)
        with col_s:
            if st.button("💾 Sauvegarder", use_container_width=True):
                try:
                    yaml.safe_load(edited_yaml)  # validate before accepting
                    st.session_state.profile_yaml = edited_yaml
                    st.session_state.gap_report = None  # invalidate stale report
                    try:  # disk write works locally, silently fails on Cloud
                        save_profile(edited_yaml)
                    except Exception:
                        pass
                    st.success("Profil sauvegardé !")
                except yaml.YAMLError as e:
                    st.error(f"YAML invalide : {e}")

        with col_d:
            st.download_button(
                "⬇️ Télécharger",
                data=st.session_state.profile_yaml,
                file_name="profile.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

        uploaded = st.file_uploader(
            "📤 Importer un profil (.yaml)",
            type=["yaml", "yml"],
        )
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8")
                yaml.safe_load(content)  # validate
                st.session_state.profile_yaml = content
                st.session_state.gap_report = None
                try:
                    save_profile(content)
                except Exception:
                    pass
                st.success("Profil importé !")
                st.rerun()
            except Exception as e:
                st.error(f"Fichier invalide : {e}")

    st.divider()

    # ── CV import ─────────────────────────────────────────────────────────────
    with st.expander("📄 Générer le profil depuis mon CV", expanded=False):
        st.caption("Upload ton CV (PDF natif) — l'IA extrait automatiquement ton profil.")
        cv_file = st.file_uploader("CV au format PDF", type=["pdf"], key="cv_upload")

        if cv_file is not None:
            if st.button("🤖 Extraire le profil", use_container_width=True, type="primary"):
                with st.spinner("Analyse de ton CV en cours..."):
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(cv_file)
                        cv_text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                        if not cv_text.strip():
                            st.error(
                                "Impossible d'extraire du texte. "
                                "Assure-toi d'utiliser un PDF natif (non scanné)."
                            )
                        else:
                            from src.chains import cv_to_profile_chain
                            st.session_state.generated_profile = cv_to_profile_chain(cv_text)
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        if st.session_state.generated_profile:
            st.markdown("**Profil extrait** — vérifie et corrige si besoin :")
            reviewed = st.text_area(
                "Profil extrait",
                value=st.session_state.generated_profile,
                height=320,
                key="cv_profile_editor",
                label_visibility="collapsed",
            )
            if st.button("✅ Appliquer ce profil", use_container_width=True, type="primary"):
                try:
                    yaml.safe_load(reviewed)
                    st.session_state.profile_yaml = reviewed
                    st.session_state.gap_report = None
                    st.session_state.generated_profile = None
                    try:
                        save_profile(reviewed)
                    except Exception:
                        pass
                    st.success("Profil appliqué !")
                    st.rerun()
                except yaml.YAMLError as e:
                    st.error(f"YAML invalide : {e}")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Analyseur d'Offres d'Emploi")
st.markdown("Colle des URLs d'offres, l'IA analyse et détecte tes lacunes.")
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    urls_input = st.text_area(
        "URLs des offres d'emploi (une par ligne)",
        height=180,
        placeholder="https://www.welcometothejungle.com/fr/companies/.../jobs/...",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Analyser", use_container_width=True, type="primary")
    st.markdown("<br>", unsafe_allow_html=True)
    manual_text = st.text_area(
        "Ou colle du texte d'offre directement",
        height=100,
        placeholder="Colle le contenu d'une offre ici si l'URL ne fonctionne pas…",
    )

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_btn:
    raw_urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]

    # Validate URLs
    invalid = [u for u in raw_urls if not u.startswith(("http://", "https://"))]
    if invalid:
        st.error(f"URLs invalides (doivent commencer par http:// ou https://) : {', '.join(invalid)}")
    elif not raw_urls and not manual_text.strip():
        st.warning("Colle au moins une URL ou un texte d'offre.")
    else:
        urls = raw_urls[:MAX_URLS]
        if len(raw_urls) > MAX_URLS:
            st.warning(f"Maximum {MAX_URLS} URLs — seules les {MAX_URLS} premières sont traitées.")

        urls_offers: list[dict] = []

        # Manual text input
        if manual_text.strip():
            from src.chains import analyze_offer_chain
            with st.spinner("Analyse du texte collé..."):
                result = analyze_offer_chain(manual_text.strip())
            if result:
                result["url"] = "Texte manuel"
                result["status"] = "success"
                urls_offers.append(result)
            else:
                urls_offers.append({"error": "LLM parsing failed", "url": "Texte manuel"})

        # URL batch — parallel
        if urls:
            progress = st.progress(0, text=f"Analyse de 0 / {len(urls)} offre(s)…")
            scraped: list[dict | None] = [None] * len(urls)
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_idx = {executor.submit(process_offer, url): i for i, url in enumerate(urls)}
                done = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    scraped[idx] = future.result()
                    done += 1
                    progress.progress(done / len(urls), text=f"Analyse de {done} / {len(urls)} offre(s)…")
            progress.empty()
            urls_offers.extend(scraped)

        # Compute match score per successful offer
        try:
            profile = yaml.safe_load(st.session_state.profile_yaml)
            for offer in urls_offers:
                if offer and offer.get("status") == "success":
                    offer["match_score"] = compute_match_score(offer, profile)
        except Exception:
            pass

        st.session_state.offers = urls_offers
        st.session_state.gap_report = None  # reset stale report
        ok = sum(1 for o in urls_offers if o and o.get("status") == "success")
        st.success(f"✅ {ok} offre(s) analysée(s) avec succès.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.offers:
    offers = st.session_state.offers
    successful = [o for o in offers if o and o.get("status") == "success"]

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Offres", "📊 Tendances", "🎯 Gap Analysis", "💾 Export"])

    # ── Tab 1: Offers ─────────────────────────────────────────────────────────
    with tab1:
        if successful:
            # Filters
            all_contracts = sorted({o.get("contract_type", "N/A") for o in successful})
            all_locations = sorted({o.get("location", "N/A") for o in successful})
            fc, fl = st.columns(2)
            with fc:
                sel_contracts = st.multiselect("Contrat", all_contracts, default=all_contracts)
            with fl:
                sel_locations = st.multiselect("Lieu", all_locations, default=all_locations)

            has_scores = any("match_score" in o for o in successful)
            min_score = 0
            if has_scores:
                min_score = st.slider(
                    "Match minimum (%)",
                    0, 100, 0, step=5,
                    help="N'afficher que les offres dont le score de correspondance avec ton profil est supérieur ou égal à ce seuil.",
                )

            filtered = [
                o for o in successful
                if o.get("contract_type", "N/A") in sel_contracts
                and o.get("location", "N/A") in sel_locations
                and o.get("match_score", 0) >= min_score
            ]

            df = offers_to_dataframe(filtered)
            has_scores = any("match_score" in o for o in filtered)
            if has_scores:
                df.insert(0, "Match %", [o.get("match_score", 0) for o in filtered])
            st.dataframe(df, use_container_width=True)

            st.divider()
            st.subheader("Détail des offres")

            # Sort by match score descending if available
            sorted_offers = sorted(filtered, key=lambda o: o.get("match_score", 0), reverse=True)
            for offer in sorted_offers:
                score = offer.get("match_score")
                badge = f" · {score}% match" if score is not None else ""
                with st.expander(f"📌 {offer.get('title', 'N/A')} — {offer.get('company', 'N/A')}{badge}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**📍 Lieu :** {offer.get('location', 'N/A')}")
                        st.markdown(f"**📄 Contrat :** {offer.get('contract_type', 'N/A')}")
                        st.markdown(f"**⏳ Expérience :** {offer.get('experience_required', 'N/A')}")
                        st.markdown(f"**💶 Salaire :** {offer.get('salary_range', 'N/A')}")
                    with col_b:
                        st.markdown(f"**🎓 Niveau :** {offer.get('level', 'N/A')}")
                        st.markdown(f"**🛠 Stack :** {', '.join(offer.get('tech_stack', []))}")
                        if score is not None:
                            st.progress(score / 100, text=f"Match profil : {score}%")
                    st.markdown("**📝 Résumé :**")
                    st.info(offer.get("summary", "N/A"))
                    st.markdown("**✅ Missions :**")
                    for m in offer.get("missions", []):
                        st.markdown(f"- {m}")
        else:
            st.warning("Aucune offre analysée avec succès.")

        errors = [o for o in offers if o and o.get("error")]
        if errors:
            with st.expander(f"⚠️ Erreurs ({len(errors)})"):
                for e in errors:
                    st.error(f"**{e['url']}**  \n{e['error']}")

    # ── Tab 2: Trends ─────────────────────────────────────────────────────────
    with tab2:
        if successful:
            st.subheader("🔥 Top Skills demandés sur le marché")
            top_skills = get_top_skills(successful, top_n=15)
            if top_skills:
                skills_df = pd.DataFrame(top_skills, columns=["Compétence", "Occurrences"])
                skills_df = skills_df.sort_values("Occurrences", ascending=False)
                st.bar_chart(skills_df.set_index("Compétence"))
                st.dataframe(skills_df, use_container_width=True, hide_index=True)
        else:
            st.info("Lance une analyse pour voir les tendances.")

    # ── Tab 3: Gap Analysis ───────────────────────────────────────────────────
    with tab3:
        if successful:
            # Profile summary
            try:
                profile = yaml.safe_load(st.session_state.profile_yaml)
                all_skills = [s for cat in profile.get("skills", {}).values() for s in cat]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Profil", profile.get("name", "—").split()[0])
                m2.metric("Rôle actuel", profile.get("current_role", "—"))
                m3.metric("Cible", profile.get("target_role", "—"))
                m4.metric("Compétences déclarées", len(all_skills))
                st.caption("Modifie ton profil dans le panneau latéral gauche.")
            except Exception:
                pass

            st.divider()
            st.subheader("🎯 Analyse des lacunes vs ton profil")

            if st.button("🧠 Générer le rapport de compétences", type="primary"):
                with st.spinner("L'IA analyse ton profil vs le marché..."):
                    try:
                        profile = yaml.safe_load(st.session_state.profile_yaml)
                    except Exception:
                        profile = load_profile()
                    st.session_state.gap_report = run_gap_analysis(successful, profile=profile)

            if st.session_state.gap_report:
                st.markdown(st.session_state.gap_report)
        else:
            st.info("Lance une analyse d'abord.")

    # ── Tab 4: Export ─────────────────────────────────────────────────────────
    with tab4:
        if successful:
            st.subheader("💾 Exporter les résultats")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                df_export = offers_to_dataframe(successful)
                if any("match_score" in o for o in successful):
                    df_export.insert(0, "Match %", [o.get("match_score", 0) for o in successful])
                csv = df_export.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ Télécharger CSV",
                    data=csv,
                    file_name="job_offers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_e2:
                json_str = json.dumps(successful, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Télécharger JSON",
                    data=json_str,
                    file_name="job_offers.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()
            st.subheader("🔄 Session")
            st.caption("Sauvegarde la session complète pour la recharger plus tard sans re-analyser.")
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

        # Session restore — always visible, even session vide
        if not successful:
            st.subheader("🔄 Session")
        restore_file = st.file_uploader(
            "📂 Charger une session sauvegardée (.json)",
            type=["json"],
            key="session_restore",
        )
        if restore_file is not None:
            try:
                data = json.loads(restore_file.read().decode("utf-8"))
                if "offers" not in data:
                    st.error("Fichier de session invalide (clé 'offers' manquante).")
                else:
                    st.session_state.offers = data["offers"]
                    st.session_state.gap_report = None
                    n = sum(1 for o in data["offers"] if o and o.get("status") == "success")
                    st.success(f"Session restaurée — {n} offre(s) chargée(s).")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")
