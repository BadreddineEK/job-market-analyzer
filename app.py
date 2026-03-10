import streamlit as st
import pandas as pd
from src.agent import process_batch, run_gap_analysis, get_top_skills
from src.utils import offers_to_dataframe
import json

st.set_page_config(
    page_title="Job Market Analyzer",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Job Market Analyzer")
    st.caption("Powered by LangChain + Groq (Llama 3.3 70B)")
    st.divider()

    st.subheader("📋 Comment utiliser")
    st.markdown("""
1. Colle les URLs d'offres d'emploi (une par ligne)
2. Clique sur **Analyser**
3. Consulte les résultats et le rapport de compétences
4. Exporte en CSV ou JSON
    """)

    st.divider()
    st.caption("💡 Modifier ton profil dans `config/profile.yaml`")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Analyseur d'Offres d'Emploi")
st.markdown("Colle des URLs d'offres, l'IA analyse et détecte tes lacunes.")

st.divider()

# Input URLs
col1, col2 = st.columns([3, 1])
with col1:
    urls_input = st.text_area(
        "URLs des offres d'emploi (une par ligne)",
        height=180,
        placeholder="https://www.welcometothejungle.com/fr/companies/.../jobs/...\nhttps://www.linkedin.com/jobs/view/...",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Analyser", use_container_width=True, type="primary")
    st.markdown("<br>", unsafe_allow_html=True)
    manual_text = st.text_area(
        "Ou colle du texte d'offre directement",
        height=100,
        placeholder="Colle le contenu d'une offre ici si l'URL ne fonctionne pas…"
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "offers" not in st.session_state:
    st.session_state.offers = []

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_btn:
    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]

    if not urls and not manual_text.strip():
        st.warning("Colle au moins une URL ou un texte d'offre.")
    else:
        if manual_text.strip():
            from src.chains import analyze_offer_chain
            with st.spinner("Analyse du texte collé..."):
                result = analyze_offer_chain(manual_text.strip())
                if result:
                    result["url"] = "manual_input"
                    result["status"] = "success"
                    urls_offers = [result]
                else:
                    urls_offers = [{"error": "LLM parsing failed", "url": "manual_input"}]
        else:
            urls_offers = []

        if urls:
            with st.spinner(f"Scraping et analyse de {len(urls)} offre(s)..."):
                scraped = process_batch(urls)
                urls_offers.extend(scraped)

        st.session_state.offers = urls_offers
        st.success(f"✅ {len([o for o in urls_offers if o.get('status') == 'success'])} offre(s) analysée(s) avec succès.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.offers:
    offers = st.session_state.offers
    successful = [o for o in offers if o.get("status") == "success"]

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Offres", "📊 Tendances", "🎯 Gap Analysis", "💾 Export"])

    # ── Tab 1: Offers table ───────────────────────────────────────────────────
    with tab1:
        if successful:
            df = offers_to_dataframe(successful)
            st.dataframe(df, use_container_width=True)

            st.divider()
            st.subheader("Détail des offres")
            for offer in successful:
                with st.expander(f"📌 {offer.get('title', 'N/A')} — {offer.get('company', 'N/A')}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**📍 Lieu :** {offer.get('location', 'N/A')}")
                        st.markdown(f"**📄 Contrat :** {offer.get('contract_type', 'N/A')}")
                        st.markdown(f"**⏳ Expérience :** {offer.get('experience_required', 'N/A')}")
                        st.markdown(f"**💶 Salaire :** {offer.get('salary_range', 'N/A')}")
                    with col_b:
                        st.markdown(f"**🎓 Niveau :** {offer.get('level', 'N/A')}")
                        st.markdown(f"**🛠 Stack :** {', '.join(offer.get('tech_stack', []))}")
                    st.markdown("**📝 Résumé :**")
                    st.info(offer.get("summary", "N/A"))
                    st.markdown("**✅ Missions :**")
                    for m in offer.get("missions", []):
                        st.markdown(f"- {m}")
        else:
            st.warning("Aucune offre analysée avec succès.")

        errors = [o for o in offers if o.get("error")]
        if errors:
            with st.expander("⚠️ Erreurs"):
                for e in errors:
                    st.error(f"{e['url']}: {e['error']}")

    # ── Tab 2: Trends ─────────────────────────────────────────────────────────
    with tab2:
        if successful:
            st.subheader("🔥 Top Skills demandés sur le marché")
            top_skills = get_top_skills(successful, top_n=15)
            if top_skills:
                skills_df = pd.DataFrame(top_skills, columns=["Compétence", "Occurrences"])
                st.bar_chart(skills_df.set_index("Compétence"))
                st.dataframe(skills_df, use_container_width=True)
        else:
            st.info("Lance une analyse pour voir les tendances.")

    # ── Tab 3: Gap Analysis ───────────────────────────────────────────────────
    with tab3:
        if successful:
            st.subheader("🎯 Analyse des lacunes vs ton profil")
            if st.button("🧠 Générer le rapport de compétences", type="primary"):
                with st.spinner("L'IA analyse ton profil vs le marché..."):
                    report = run_gap_analysis(successful)
                st.markdown(report)
        else:
            st.info("Lance une analyse d'abord.")

    # ── Tab 4: Export ─────────────────────────────────────────────────────────
    with tab4:
        if successful:
            st.subheader("💾 Exporter les résultats")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                df_export = offers_to_dataframe(successful)
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
