from collections import Counter
from typing import Optional
import re
import yaml
from pathlib import Path
from src.scraper import scrape_job_offer
from src.chains import analyze_offer_chain, gap_analysis_chain

CONFIG_DIR = Path(__file__).parent.parent / "config"

_BLOCKED_HOSTS = {
    "linkedin.com": (
        "LinkedIn bloque le scraping. "
        "Ouvre l'offre dans ton navigateur, sélectionne tout le texte (Ctrl+A) et colle-le dans "
        "la zone « Ou colle du texte d'offre directement »."
    ),
    "indeed.com": (
        "Indeed bloque le scraping. "
        "Colle le texte de l'offre manuellement via la zone de texte."
    ),
}


def load_profile() -> dict:
    """Load user profile from YAML config."""
    with open(CONFIG_DIR / "profile.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_profile(yaml_content: str) -> None:
    """Write raw YAML string back to config/profile.yaml."""
    with open(CONFIG_DIR / "profile.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)


def _blocked_reason(url: str) -> Optional[str]:
    for host, msg in _BLOCKED_HOSTS.items():
        if host in url:
            return msg
    return None


def process_offer(url: str, text: str | None = None) -> dict:
    """Scrape + analyze a single job offer URL.

    If `text` is provided (e.g. pre-fetched description from jobspy),
    the scraping step is skipped and `text` is fed directly to the LLM.
    """
    if text:
        content = text
    else:
        blocked = _blocked_reason(url)
        if blocked:
            return {"error": blocked, "url": url}
        content = scrape_job_offer(url)
        if not content:
            return {"error": "Scraping échoué : page inaccessible ou vide.", "url": url}

    analysis = analyze_offer_chain(content)
    if not analysis:
        return {"error": "LLM parsing failed", "url": url}

    analysis["url"] = url
    analysis["status"] = "success"
    return analysis


def compute_match_score(offer: dict, profile: dict) -> int:
    """
    Compute the % of offer skills already covered by the profile.
    Uses case-insensitive substring matching (bidirectional).
    """
    profile_skills = {
        s.lower()
        for cat in profile.get("skills", {}).values()
        for s in cat
        if s  # skip None and empty entries
    }
    offer_skills = [
        s.lower()
        for s in (offer.get("required_skills") or []) + (offer.get("tech_stack") or [])
        if s  # skip None / empty entries from LLM output
    ]
    if not offer_skills:
        return 0
    matches = sum(
        1 for s in offer_skills
        if any(p in s or s in p for p in profile_skills)
    )
    return round(matches / len(offer_skills) * 100)


def get_skill_match_detail(offer: dict, profile: dict) -> tuple[list[str], list[str]]:
    """Return (matched_skills, missing_skills) for an offer vs the profile."""
    profile_skills = {
        s.lower()
        for cat in profile.get("skills", {}).values()
        for s in (cat or [])
        if s
    }
    offer_skills = [
        s for s in (offer.get("required_skills") or []) + (offer.get("tech_stack") or [])
        if s
    ]
    matched = [s for s in offer_skills if any(p in s.lower() or s.lower() in p for p in profile_skills)]
    missing = [s for s in offer_skills if s not in matched]
    return matched, missing


def run_gap_analysis(offers: list[dict], profile: dict = None) -> str:
    """Aggregate skills from successful offers and run gap analysis."""
    if profile is None:
        profile = load_profile()

    all_skills = []
    for offer in offers:
        if offer.get("status") == "success":
            all_skills.extend(offer.get("required_skills") or [])
            all_skills.extend(offer.get("tech_stack") or [])

    if not all_skills:
        return "Aucune offre analysée avec succès pour générer un rapport."

    return gap_analysis_chain(profile, all_skills)


def get_top_skills(offers: list[dict], top_n: int = 15) -> list[tuple]:
    """Return the most requested skills across all analyzed offers."""
    all_skills = []
    for offer in offers:
        if offer.get("status") == "success":
            all_skills.extend(offer.get("required_skills") or [])
            all_skills.extend(offer.get("tech_stack") or [])
    return Counter(all_skills).most_common(top_n)


def get_blocking_skills(offers: list[dict], profile: dict) -> list[dict]:
    """
    Return skills ranked by market frequency, flagged as in/out of profile.
    Each entry: {skill, count, total_offers, pct, in_profile}
    """
    profile_skills = {
        s.lower()
        for cat in profile.get("skills", {}).values()
        for s in (cat or [])
        if s
    }
    successful = [o for o in offers if o.get("status") == "success"]
    total = len(successful)
    if total == 0:
        return []

    freq: Counter = Counter()
    for offer in successful:
        # Count each skill once per offer (set dedup)
        skills = {
            s for s in (offer.get("required_skills") or []) + (offer.get("tech_stack") or [])
            if s
        }
        for s in skills:
            freq[s] += 1

    result = []
    for skill, count in freq.most_common(30):
        in_profile = any(p in skill.lower() or skill.lower() in p for p in profile_skills)
        result.append({
            "skill": skill,
            "count": count,
            "total": total,
            "pct": round(count / total * 100),
            "in_profile": in_profile,
        })
    return result


def get_salary_stats(offers: list[dict]) -> dict | None:
    """Parse salary_range strings and return min/max/avg in euros."""
    values: list[float] = []
    for o in offers:
        if o.get("status") != "success":
            continue
        raw = (o.get("salary_range") or "").replace(" ", "")
        nums = re.findall(r'\d+(?:[.,]\d+)?', raw)
        for n in nums:
            try:
                v = float(n.replace(",", "."))
                if v < 500:      # likely in k€
                    v *= 1000
                if 15_000 <= v <= 500_000:
                    values.append(v)
            except ValueError:
                pass
    if not values:
        return None
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "avg": int(sum(values) / len(values)),
        "n_offers": len(values),
    }
