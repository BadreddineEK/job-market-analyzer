import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

load_dotenv(Path(__file__).parent.parent / ".env")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Cache prompt files — read once, never again
_PROMPTS: dict[str, str] = {}

# Cache LLM instances by (model, temperature)
_LLM_CACHE: dict[tuple, ChatGroq] = {}


def _get_prompt(name: str) -> str:
    if name not in _PROMPTS:
        _PROMPTS[name] = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return _PROMPTS[name]


def get_llm(model: str = None, temperature: float = 0.1) -> ChatGroq:
    resolved_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    key = (resolved_model, temperature)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatGroq(
            model=resolved_model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _LLM_CACHE[key]


def analyze_offer_chain(job_content: str) -> Optional[dict]:
    """Analyze a single job offer and return structured JSON."""
    prompt = PromptTemplate.from_template(_get_prompt("analyze_offer.txt"))
    chain = prompt | get_llm() | StrOutputParser()
    raw = chain.invoke({"job_content": job_content})
    try:
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cv_to_profile_chain(cv_text: str) -> Optional[str]:
    """Extract a profile YAML from raw CV text using the LLM."""
    prompt = PromptTemplate.from_template(_get_prompt("cv_to_profile.txt"))
    chain = prompt | get_llm(temperature=0.2) | StrOutputParser()
    raw = chain.invoke({"cv_text": cv_text[:8000]})
    raw = raw.strip()
    # If the model wrapped the YAML in a code fence, extract just the content
    fence_match = re.search(r"```(?:yaml)?\s*\n([\s\S]+?)\n```", raw)
    if fence_match:
        return fence_match.group(1).strip()
    # Otherwise strip any stray leading/trailing fence markers
    raw = re.sub(r"^```(?:yaml)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


def _offer_vars(offer: dict) -> dict:
    """Extract common template variables from an offer dict."""
    return {
        "title": offer.get("title", "N/A"),
        "company": offer.get("company", "N/A"),
        "location": offer.get("location", "N/A"),
        "missions": ", ".join(m for m in (offer.get("missions") or []) if m),
        "required_skills": ", ".join(s for s in (offer.get("required_skills") or []) if s),
        "tech_stack": ", ".join(s for s in (offer.get("tech_stack") or []) if s),
        "level": offer.get("level", "N/A"),
        "summary": offer.get("summary", "N/A"),
        "match_score": offer.get("match_score", 0),
    }


def cover_letter_chain(offer: dict, profile: dict) -> str:
    """Generate a personalized cover letter for a specific offer."""
    prompt = PromptTemplate.from_template(_get_prompt("cover_letter.txt"))
    chain = prompt | get_llm(temperature=0.5) | StrOutputParser()
    return chain.invoke({
        **_offer_vars(offer),
        "profile": json.dumps(profile, ensure_ascii=False, indent=2),
    })


def interview_prep_chain(offer: dict, profile: dict) -> str:
    """Generate interview preparation questions for a specific offer."""
    prompt = PromptTemplate.from_template(_get_prompt("interview_prep.txt"))
    chain = prompt | get_llm(temperature=0.3) | StrOutputParser()
    return chain.invoke({
        **_offer_vars(offer),
        "profile": json.dumps(profile, ensure_ascii=False, indent=2),
    })


def profile_adaptation_chain(offer: dict, profile: dict) -> str:
    """Generate profile adaptation advice for a specific offer."""
    prompt = PromptTemplate.from_template(_get_prompt("profile_adaptation.txt"))
    chain = prompt | get_llm(temperature=0.3) | StrOutputParser()
    return chain.invoke({
        **_offer_vars(offer),
        "profile": json.dumps(profile, ensure_ascii=False, indent=2),
    })


def gap_analysis_chain(profile: dict, aggregated_skills: list[str]) -> str:
    """Run a skill gap analysis comparing the user profile to market demand."""
    prompt = PromptTemplate.from_template(_get_prompt("gap_analysis.txt"))
    chain = prompt | get_llm(temperature=0.3) | StrOutputParser()
    profile_str = json.dumps(profile, ensure_ascii=False, indent=2)
    skills_str = ", ".join(set(aggregated_skills))
    return chain.invoke({"profile": profile_str, "aggregated_skills": skills_str})
