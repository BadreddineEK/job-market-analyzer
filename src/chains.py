import json
import os
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
    raw = chain.invoke({"cv_text": cv_text[:8000]})  # cap to avoid context overflow
    # Strip potential markdown code fences
    return raw.strip().removeprefix("```yaml").removeprefix("```").removesuffix("```").strip()
    """Run a skill gap analysis comparing the user profile to market demand."""
    prompt = PromptTemplate.from_template(_get_prompt("gap_analysis.txt"))
    chain = prompt | get_llm(temperature=0.3) | StrOutputParser()
    profile_str = json.dumps(profile, ensure_ascii=False, indent=2)
    skills_str = ", ".join(set(aggregated_skills))
    return chain.invoke({"profile": profile_str, "aggregated_skills": skills_str})
