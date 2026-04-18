"""
Opportunity Inbox Copilot — SOFTEC 2026 AI Hackathon
======================================================
A production-ready Streamlit MVP that parses raw emails/notices,
extracts structured opportunity data via Ollama (local LLM), and ranks them
deterministically against a student profile.

Architecture:
  - Frontend  : Streamlit (dashboard + sidebar form)
  - AI Layer  : Ollama (local LLM, zero cost, no API key) via ONE batched call
  - Ranking   : Deterministic Python scoring (no LLM randomness)
  - Output    : Priority-ranked cards with badges, reasons, and checklists
"""

import os, json, re, math
from datetime import datetime, date, timedelta
from typing import Optional
import streamlit as st
import requests  # for Ollama local HTTP API — zero cost, no key needed
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Opportunity Inbox Copilot",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --accent2:   #3fb950;
    --warn:      #d29922;
    --danger:    #f85149;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --radius:    10px;
}

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--text); }

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%;  left: -20%;
    width: 60%;  height: 200%;
    background: radial-gradient(ellipse, rgba(88,166,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 .3rem; letter-spacing: -.5px; }
.hero p  { color: var(--muted); margin: 0; font-size: .95rem; }

/* ── Opportunity card ── */
.opp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: .6rem;
    transition: border-color .2s;
}
.opp-card:hover { border-color: var(--accent); }
.opp-card.ineligible { border-left: 3px solid var(--danger); }
.opp-card .rank  { font-family: 'JetBrains Mono', monospace; font-size:.75rem; color: var(--muted); }
.opp-card .title { font-size: 1.15rem; font-weight: 600; margin: .3rem 0 .6rem; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: .15rem .6rem;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .3px;
    margin-right: .35rem;
    margin-bottom: .35rem;
}
.badge-high   { background: rgba(248,81,73,.15);  color: #f85149; border: 1px solid rgba(248,81,73,.3);  }
.badge-medium { background: rgba(210,153,34,.15); color: #d29922; border: 1px solid rgba(210,153,34,.3); }
.badge-low    { background: rgba(63,185,80,.15);  color: #3fb950; border: 1px solid rgba(63,185,80,.3);  }
.badge-type   { background: rgba(88,166,255,.12); color: #58a6ff; border: 1px solid rgba(88,166,255,.25);}
.badge-inelig { background: rgba(248,81,73,.1);   color: #f85149; border: 1px solid rgba(248,81,73,.2);  }
.badge-noise  { background: rgba(139,148,158,.1); color: #8b949e; border: 1px solid rgba(139,148,158,.2);}

/* ── Score bar ── */
.score-bar-wrap { background:#21262d; border-radius:999px; height:6px; margin:.4rem 0 .8rem; }
.score-bar      { height:6px; border-radius:999px; }

/* ── Why-box ── */
.why-box {
    background: rgba(88,166,255,.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: .6rem .9rem;
    font-size: .88rem;
    color: var(--muted);
    margin: .5rem 0;
}

/* ── Checklist ── */
.checklist { list-style: none; padding: 0; margin: .5rem 0 0; }
.checklist li { font-size:.87rem; color:var(--muted); padding:.2rem 0; }
.checklist li::before { content:"☐  "; color: var(--accent); }

/* ── Noise section ── */
.noise-card {
    background: rgba(139,148,158,.06);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .8rem 1.1rem;
    margin-bottom: .6rem;
    font-size:.88rem;
    color: var(--muted);
}

/* ── Metric tiles ── */
.metric-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.metric-tile {
    flex:1; min-width:120px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .9rem 1rem;
    text-align: center;
}
.metric-tile .num { font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.metric-tile .lbl { font-size:.75rem; color: var(--muted); margin-top:.1rem; }

/* ── Streamlit widget overrides ── */
.stTextArea textarea, .stTextInput input {
    background: #21262d !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .83rem !important;
}
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PYDANTIC DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class OpportunityModel(BaseModel):
    """
    Structured representation of ONE parsed email/notice.
    The LLM fills every field; defaults handle missing data gracefully.
    """
    is_opportunity: bool = Field(..., description="True if genuine opportunity, False if noise.")
    title: str = Field(default="Unknown", description="Short descriptive title.")
    type: str = Field(default="Other", description="Scholarship|Internship|Competition|Research|Job|Other")
    organization: str = Field(default="Unknown", description="Issuing organization name.")
    deadline_iso: Optional[str] = Field(default=None, description="Deadline as YYYY-MM-DD or null.")
    min_cgpa: Optional[float] = Field(default=None, description="Minimum CGPA required or null.")
    min_semester: Optional[int] = Field(default=None, description="Minimum semester number required (e.g. 5 means Semester 5+), or null.")
    eligible_programs: list[str] = Field(default_factory=list, description="Eligible degree programs; empty = all.")
    required_skills: list[str] = Field(default_factory=list, description="Explicitly required skills.")
    required_documents: list[str] = Field(default_factory=list, description="Documents needed to apply.")
    action_link: Optional[str] = Field(default=None, description="Application URL or email, or null.")
    summary: str = Field(default="", description="One-sentence summary of the opportunity.")
    noise_reason: Optional[str] = Field(default=None, description="Why this is classified as noise.")

    @field_validator("type")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        valid = {"Scholarship", "Internship", "Competition", "Research", "Job"}
        for t in valid:
            if t.lower() in v.lower():
                return t
        return "Other"


class RankedOpportunity(BaseModel):
    """Opportunity enriched with deterministic ranking metadata."""
    parsed: OpportunityModel
    score: float                    # 0.0–1.0 composite match score
    eligibility_score: float        # Profile Fit sub-component
    urgency_score: float            # Urgency sub-component
    preference_score: float         # Value/Preference sub-component
    urgency_label: str              # High | Medium | Low | Expired
    days_until_deadline: Optional[int]
    is_eligible: bool
    ineligibility_reason: Optional[str]
    why_it_matters: str             # Evidence-backed human-readable reason
    action_checklist: list[str]     # Step-by-step actions for student
    raw_email_snippet: str          # First 120 chars for reference


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GEMINI BATCH PARSER  (ONE API CALL for all emails)
# ─────────────────────────────────────────────────────────────────────────────

# ── Ollama defaults — overridden at runtime from sidebar, never mutated globally ──

PARSE_SYSTEM_PROMPT = """You are an expert email parser for a student opportunity tracker.
Input: a JSON array where each element has "index" (int) and "text" (the raw email).
Output: a raw JSON array of the SAME length — one object per input email. Nothing else.

STRICT: no markdown, no backticks, no explanation, no preamble. Pure JSON array only.

Each object must have exactly these keys:
{
  "is_opportunity": bool,
  "title": str,
  "type": "Scholarship|Internship|Competition|Research|Job|Other",
  "organization": str,
  "deadline_iso": "YYYY-MM-DD" or null,
  "min_cgpa": float or null,
  "min_semester": int or null,
  "eligible_programs": [str],
  "required_skills": [str],
  "required_documents": [str],
  "action_link": str or null,
  "summary": str,
  "noise_reason": str or null
}

Rules:
- is_opportunity=false for holidays, personal messages, admin/cafeteria notices.
- Ignore email signatures, footers, "Sent from iPhone", unsubscribe text.
- deadline_iso must be YYYY-MM-DD format or null — never a prose string.
- min_semester: extract the minimum semester number if stated (e.g. "Semester 5+" gives 5, "5th semester or above" gives 5). Set null if not mentioned.
- noise_reason is required when is_opportunity=false, null otherwise.
- Output array length MUST exactly match input array length."""

# ── Noise fallback used when an individual email parse fails ─────────────────
def _noise_fallback(reason: str = "Parse error") -> OpportunityModel:
    return OpportunityModel(is_opportunity=False, noise_reason=reason)


def _extract_json_array(raw: str) -> list[dict]:
    """
    Robustly extract a JSON array from model output.
    Handles: markdown fences, leading prose, single-object wraps,
    truncated arrays, and stray trailing commas.
    """
    # 1. Strip markdown code fences
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    # 2. Find the first '[' and last ']' — discard any surrounding prose
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    elif raw.strip().startswith("{"):
        # Model returned a single object instead of array — wrap it
        raw = f"[{raw.strip()}]"

    # 3. Fix trailing commas before ] or } (common model mistake)
    raw = re.sub(r",\s*([\]}])", r"\1", raw)

    return json.loads(raw)


def _call_ollama_single(email_text: str, ollama_url: str, model: str) -> OpportunityModel:
    """
    Parse ONE email with Ollama.
    Used as fallback when batch parsing fails, and for individual retries.
    Timeout: 120s per email (generous for slow hardware).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Parse this 1 email and return a JSON array with exactly 1 element:\n"
                f"{json.dumps([{'index': 0, 'text': email_text.strip()}])}"
            )},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(ollama_url, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()
    parsed = _extract_json_array(raw)
    return OpportunityModel(**parsed[0]) if parsed else _noise_fallback()


def parse_emails_with_ollama(
    emails: list[str],
    ollama_url: str,
    model: str,
    chunk_size: int = 3,
    progress_cb=None,
) -> list[OpportunityModel]:
    """
    Parse emails in small chunks to avoid LLM timeout on large batches.

    Strategy:
      1. Split emails into chunks of `chunk_size` (default 3).
      2. For each chunk, call Ollama with a batched request (fast path).
      3. If a chunk fails JSON parsing, fall back to one-by-one parsing (safe path).
      4. Always returns a list of same length as input — never raises.

    chunk_size=3 is the sweet spot: fast enough (~15-25s per chunk on CPU),
    small enough that even slow models don't truncate their JSON output.
    """
    results: list[OpportunityModel] = []
    total_chunks = math.ceil(len(emails) / chunk_size)

    for chunk_idx in range(total_chunks):
        chunk = emails[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        if progress_cb:
            progress_cb(chunk_idx, total_chunks, len(chunk))

        emails_payload = json.dumps(
            [{"index": i, "text": e.strip()} for i, e in enumerate(chunk)],
            ensure_ascii=False, separators=(",", ":"),
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Parse these {len(chunk)} emails and return the JSON array:\n"
                    f"{emails_payload}"
                )},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        try:
            # Per-chunk timeout: 90s × chunk_size — enough for slow hardware
            resp = requests.post(ollama_url, json=payload, timeout=90 * chunk_size)
            resp.raise_for_status()
            raw_text = resp.json()["message"]["content"].strip()
            parsed_list = _extract_json_array(raw_text)

            # Validate length — model sometimes drops items on large batches
            if len(parsed_list) != len(chunk):
                raise ValueError(
                    f"Length mismatch: expected {len(chunk)}, got {len(parsed_list)}"
                )

            for item in parsed_list:
                try:
                    results.append(OpportunityModel(**item))
                except Exception:
                    results.append(_noise_fallback("Validation error on this entry."))

        except (json.JSONDecodeError, ValueError, KeyError):
            # Batch failed — fall back to one-by-one for this chunk
            for email in chunk:
                try:
                    results.append(_call_ollama_single(email, ollama_url, model))
                except Exception as e:
                    results.append(_noise_fallback(f"Individual parse failed: {e}"))

        except requests.exceptions.Timeout:
            # Whole chunk timed out — fall back one-by-one
            for email in chunk:
                try:
                    results.append(_call_ollama_single(email, ollama_url, model))
                except Exception:
                    results.append(_noise_fallback("Timed out on this email."))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DETERMINISTIC RANKING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
#
#  Master formula:
#    score = (eligibility_score × 0.5)
#           + (urgency_score     × 0.3)
#           + (preference_score  × 0.2)
#
#  Weights rationale:
#    0.5 → eligibility: no point ranking something student can't apply for
#    0.3 → urgency:     time-sensitive opportunities must bubble up
#    0.2 → preference:  student's stated interests as a tie-breaker
# ─────────────────────────────────────────────────────────────────────────────

def compute_eligibility_score(
    opp: OpportunityModel, profile: dict
) -> tuple[float, bool, Optional[str]]:
    """
    Profile Fit sub-score (weight: 0.5).

    Returns:
        (score 0–1, is_eligible bool, ineligibility_reason or None)

    CGPA check is a hard gate: failing it caps the score at 0.1 and sets
    is_eligible=False so the card is clearly flagged in the UI.
    """
    score = 0.0
    is_eligible = True
    reason = None

    # ── Hard gate: Semester ──────────────────────────────────────────────────
    if opp.min_semester is not None and profile["semester"] < opp.min_semester:
        is_eligible = False
        reason = f"Requires Semester {opp.min_semester}+ (you are in Semester {profile['semester']})"
        return 0.05, False, reason  # hard cap — heavily penalised, still shown

    # ── Hard gate: CGPA ───────────────────────────────────────────────────────
    if opp.min_cgpa is not None and profile["cgpa"] < opp.min_cgpa:
        is_eligible = False
        reason = f"Requires CGPA ≥ {opp.min_cgpa:.1f} (yours: {profile['cgpa']:.2f})"
        return 0.1, False, reason   # hard cap — still shown for awareness

    score += 0.2  # Both hard gates passed — base credit

    # ── Program match (30% of eligibility sub-score) ──────────────────────────
    if not opp.eligible_programs:
        score += 0.3  # Open to all — full credit
    else:
        student_prog = profile["degree"].lower()
        matched = any(
            p.lower() in student_prog or student_prog in p.lower()
            for p in opp.eligible_programs
        )
        score += 0.3 if matched else 0.05

    # ── Skills match (50% of eligibility sub-score) ───────────────────────────
    if not opp.required_skills:
        score += 0.5  # No skills listed → open opportunity → full credit
    else:
        student_skills_lower = {s.lower() for s in profile["skills"]}
        matched_count = sum(
            1 for skill in opp.required_skills
            if skill.lower() in student_skills_lower
        )
        skill_ratio = matched_count / len(opp.required_skills)
        score += skill_ratio * 0.5

    return min(score, 1.0), is_eligible, reason


def compute_urgency_score(
    opp: OpportunityModel,
) -> tuple[float, str, Optional[int]]:
    """
    Urgency sub-score (weight: 0.3).

    Returns:
        (score 0–1, urgency_label, days_until_deadline or None)

    Urgency curve (non-linear decay):
      No deadline → 0.0   (always ranked lower on this dimension)
      Expired     → 0.0   (still shown, flagged as expired)
      ≤ 7 days   → 0.9–1.0  → "High"
      ≤ 30 days  → 0.5–0.89 → "Medium"
      > 30 days  → 0.1–0.49 → "Low"
    """
    if opp.deadline_iso is None:
        return 0.0, "Low", None

    try:
        deadline = datetime.strptime(opp.deadline_iso, "%Y-%m-%d").date()
        days = (deadline - date.today()).days
    except ValueError:
        return 0.0, "Low", None

    if days < 0:
        return 0.0, "Expired", days
    elif days <= 7:
        score = 1.0 - (days / 7) * 0.1   # 1.0 → 0.9
        return round(score, 4), "High", days
    elif days <= 30:
        score = 0.89 - ((days - 7) / 23) * 0.39  # 0.89 → 0.5
        return round(score, 4), "Medium", days
    else:
        capped = min(days, 90)
        score = 0.49 - ((capped - 30) / 60) * 0.39  # 0.49 → 0.1
        return round(max(score, 0.1), 4), "Low", days


def compute_preference_score(
    opp: OpportunityModel, profile: dict
) -> float:
    """
    Value / Preference Alignment sub-score (weight: 0.2).

    Factors:
      Preferred type match      → +0.50
      Financial need + Scholarship → +0.30
      Location preference match → +0.20
    """
    score = 0.0

    # ── Type preference ───────────────────────────────────────────────────────
    if opp.type in profile["preferred_types"]:
        score += 0.5

    # ── Financial need amplifier for scholarships ─────────────────────────────
    if profile["financial_need"] and opp.type == "Scholarship":
        score += 0.3

    # ── Location heuristic ────────────────────────────────────────────────────
    location_pref = profile.get("location_pref", "").lower()
    combined_text = (opp.summary + " " + opp.organization + " " + " ".join(opp.required_documents)).lower()
    if location_pref in ("remote/online", "remote", "online"):
        if any(kw in combined_text for kw in ["remote", "online", "virtual", "anywhere"]):
            score += 0.2
    elif location_pref:
        if location_pref in combined_text:
            score += 0.2
    else:
        score += 0.1  # No preference stated → neutral partial credit

    return min(score, 1.0)


def build_why_it_matters(
    item_data: dict,  # intermediate dict before RankedOpportunity creation
    opp: OpportunityModel,
    profile: dict,
) -> str:
    """Constructs an evidence-backed reason string for the UI."""
    if not item_data["is_eligible"]:
        return f"⚠️ Ineligible — {item_data['ineligibility_reason']}. Shown for awareness only."

    parts = []
    if opp.type in profile["preferred_types"]:
        parts.append(f"matches your preferred type ({opp.type})")
    days = item_data["days"]
    if days is not None and days >= 0:
        parts.append(f"deadline in {days} day{'s' if days != 1 else ''}")
    if profile["financial_need"] and opp.type == "Scholarship":
        parts.append("aligns with your financial need preference")
    if item_data["eligibility_score"] > 0.7:
        parts.append("strong skills/program match")
    elif not opp.required_skills and not opp.min_cgpa:
        parts.append("open to all students — no hard requirements")

    if parts:
        return "This opportunity " + ", ".join(parts) + "."
    return opp.summary or "Relevant based on your profile."


def build_action_checklist(opp: OpportunityModel) -> list[str]:
    """Generates a practical, ordered action checklist for the student."""
    steps = []

    if opp.required_documents:
        for doc in opp.required_documents[:4]:
            steps.append(f"Prepare: {doc}")
    else:
        steps.append("Prepare your CV and latest transcript")

    if opp.action_link:
        steps.append(f"Apply / Register at: {opp.action_link}")
    else:
        steps.append("Find & bookmark the official application portal")

    if opp.deadline_iso:
        steps.append(f"Submit before: {opp.deadline_iso}")
    else:
        steps.append("Confirm exact deadline with the issuing organization")

    steps.append("Email yourself a reminder 3 days before the deadline")
    steps.append("Send a confirmation / follow-up email after submitting")
    return steps


def rank_opportunities(
    paired: list[tuple[OpportunityModel, str]],
    profile: dict,
) -> tuple[list[RankedOpportunity], list[RankedOpportunity]]:
    """
    Master ranking function.

    Separates opportunities from noise, scores each opportunity using the
    three sub-components, then sorts descending by composite score.
    Ineligible items are pushed to the bottom of the ranked list (still shown).

    Returns: (ranked_opportunities, noise_items)
    """
    opportunities: list[RankedOpportunity] = []
    noise_items: list[RankedOpportunity] = []

    for opp, snippet in paired:
        if not opp.is_opportunity:
            noise_items.append(RankedOpportunity(
                parsed=opp, score=0, eligibility_score=0,
                urgency_score=0, preference_score=0,
                urgency_label="N/A", days_until_deadline=None,
                is_eligible=False, ineligibility_reason=None,
                why_it_matters="", action_checklist=[],
                raw_email_snippet=snippet,
            ))
            continue

        # ── Compute the three sub-scores ─────────────────────────────────────
        e_score, eligible, inelig_reason = compute_eligibility_score(opp, profile)
        u_score, urgency_label, days     = compute_urgency_score(opp)
        p_score                          = compute_preference_score(opp, profile)

        # ── Composite formula (judge-readable) ────────────────────────────────
        composite = (e_score * 0.5) + (u_score * 0.3) + (p_score * 0.2)

        item_data = {
            "is_eligible": eligible,
            "ineligibility_reason": inelig_reason,
            "eligibility_score": e_score,
            "days": days,
        }

        opportunities.append(RankedOpportunity(
            parsed=opp,
            score=round(composite, 4),
            eligibility_score=round(e_score, 4),
            urgency_score=round(u_score, 4),
            preference_score=round(p_score, 4),
            urgency_label=urgency_label,
            days_until_deadline=days,
            is_eligible=eligible,
            ineligibility_reason=inelig_reason,
            why_it_matters=build_why_it_matters(item_data, opp, profile),
            action_checklist=build_action_checklist(opp),
            raw_email_snippet=snippet,
        ))

    # ── Sort: eligible first, then by score descending ────────────────────────
    opportunities.sort(key=lambda x: (x.is_eligible, x.score), reverse=True)
    return opportunities, noise_items


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAMPLE EMAIL DATASET  (10 emails per specification)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_EMAILS = [
    # ─── Internship 1: Very close deadline ───────────────────────────────────
    """Subject: URGENT: Software Engineering Intern — Arbisoft Lahore [3-Day Deadline]
From: careers@arbisoft.com

Dear Student,

Arbisoft is hiring Software Engineering Interns for our Summer 2026 cohort.

Position: Software Engineering Intern
Location: Lahore, Pakistan (On-site)
Duration: 3 months (June–August 2026)
Stipend: PKR 50,000/month

Requirements:
- BS Computer Science or Software Engineering (Semester 6+)
- Proficient in Python, Django or React
- CGPA: 3.0 and above

Required Documents: CV, Unofficial Transcript, GitHub Profile link

Application Deadline: 2026-04-21

Apply at: https://arbisoft.com/careers/intern-2026

Recruitment Team, Arbisoft""",

    # ─── Internship 2: Normal deadline ───────────────────────────────────────
    """Subject: Data Science Internship Opportunity — Systems Limited
From: hr@systemslimited.pk

Hello FAST-NU Students,

Systems Limited invites applications for our Data Science & AI Internship.

Details:
- Role: Data Science Intern
- Skills Required: Python, Pandas, Machine Learning, SQL
- Eligibility: BS CS / SE / DS — minimum CGPA 2.75
- Semester: 5th or above
- Duration: 2 months
- Deadline: 2026-05-10

Documents needed: CV, Cover Letter, Transcript

Send applications to: internships@systemslimited.pk

HR Department, Systems Limited""",

    # ─── Internship 3: Rolling / no strict deadline ───────────────────────────
    """Subject: Front-End Developer Internship — Devsinc [Rolling Basis]
From: talent@devsinc.com

Hi there,

Devsinc is looking for passionate Front-End Developer interns.

Requirements:
- React.js, HTML/CSS, JavaScript experience
- Any CS/SE program, Semester 4+
- CGPA: 2.5 minimum

Perks: Mentorship, certificate, possible full-time offer

No strict deadline — we hire on a rolling basis.

Apply here: https://devsinc.com/jobs/frontend-intern

Talent Acquisition | Devsinc""",

    # ─── Scholarship 1: Accessible CGPA ─────────────────────────────────────
    """Subject: HEC Need-Based Scholarship 2026–27 — Applications Open
From: scholarships@hec.gov.pk

Dear Students,

HEC invites applications for Need-Based Scholarships for 2026-27.

Eligibility:
- Pakistani nationals enrolled in HEC-recognized universities
- Minimum CGPA: 2.50
- Demonstrated financial need (income affidavit required)
- All degree programs eligible

Award: Full tuition waiver + PKR 6,000/month stipend

Required Documents:
- Income Certificate / Affidavit
- CNIC copies (student & parent)
- Enrollment Certificate
- Latest Transcript

Deadline: 2026-05-30

Apply online: https://hec.gov.pk/scholarships/need-based-2026

HEC Scholarships Division""",

    # ─── Scholarship 2: High CGPA requirement (likely mismatch) ─────────────
    """Subject: LUMS Merit Scholarship for CS Students — CGPA 3.7+ Required
From: financial-aid@lums.edu.pk

Dear Applicant,

LUMS offers a prestigious Merit Scholarship for exceptional CS students.

Eligibility Criteria:
- Undergraduate CS/SE students (Final year preferred)
- Minimum CGPA: 3.70 — strict cut-off, no exceptions
- Strong research background preferred

Award: 75% tuition waiver + research stipend

Required Documents:
- Official Transcript
- Two Reference Letters
- Statement of Purpose
- Research Portfolio (if any)

Deadline: 2026-06-15

Apply at: https://lums.edu.pk/merit-scholarship-cs

LUMS Financial Aid Office""",

    # ─── Noise 1: University closure ─────────────────────────────────────────
    """Subject: University Closed on April 23 — Eid-ul-Fitr Holiday
From: registrar@university.edu.pk

Dear All,

This is to inform all students and faculty that the university will remain CLOSED
on April 23, 2026 (Wednesday) on account of Eid-ul-Fitr.

Regular classes will resume on April 24, 2026.

Eid Mubarak to you and your families!

Office of the Registrar""",

    # ─── Noise 2: Personal message ────────────────────────────────────────────
    """Subject: Hello
From: friend@gmail.com

Hey bro,

Are you coming to Ahsan's party this weekend? Let me know, it's at his place in DHA.
Bring snacks lol

– Bilal
Sent from my iPhone""",

    # ─── Noise 3: Admin notice ────────────────────────────────────────────────
    """Subject: Important: Cafeteria Menu Update for April 2026
From: admin@university.edu.pk

Dear Students,

Please note that the cafeteria on Block B will serve a revised menu starting April 20.

New items include: Nihari, Haleem, and a healthy salad bar.

Prices remain unchanged.

University Administration""",

    # ─── Competition 1: SOFTEC 2026 ──────────────────────────────────────────
    """Subject: SOFTEC 2026 — Pakistan's Largest CS Festival | AI Hackathon Inside!
From: softec@nu.edu.pk

Dear Students,

FAST-NUCES proudly presents SOFTEC 2026 — Pakistan's largest student-run CS festival!

Events include:
- Speed Programming (competitive coding)
- AI Hackathon (24-hour challenge — win up to PKR 200,000!)
- Project Exhibition
- Softec Hacks (hardware challenge)

Eligibility: Open to all university students in Pakistan
Team Size: 1–4 members

Registration Deadline: 2026-04-25

Register here: https://softec.nu.edu.pk/register

Cash prizes worth PKR 1,000,000+ up for grabs!

SOFTEC 2026 Organizing Committee""",

    # ─── Competition 2: ICPC ─────────────────────────────────────────────────
    """Subject: ICPC Asia Lahore Regional 2026 — Team Registration Open
From: icpc@icpc-lahore.pk

Dear Competitive Programmers,

The ICPC Asia Lahore Regional Contest 2026 is now accepting team registrations.

Details:
- Format: Team-based (exactly 3 members)
- Eligibility: Enrolled students; CGPA 2.0+
- Skills: C++, Data Structures & Algorithms, Problem Solving
- Prize: Top teams qualify for ICPC World Finals

Important Dates:
- Registration Deadline: 2026-05-05
- Contest Date: 2026-05-25

Register at: https://icpc.global/regionals/lahore-2026

ICPC Lahore Steering Committee""",
]

SAMPLE_EMAIL_TEXT = "\n\n────────────────────────────────────────────────────\n\n".join(SAMPLE_EMAILS)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UI COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

URGENCY_BADGE = {
    "High": "badge-high",
    "Medium": "badge-medium",
    "Low": "badge-low",
    "Expired": "badge-inelig",
    "N/A": "badge-noise",
}

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>📬 Opportunity Inbox Copilot</h1>
        <p>Paste raw emails &nbsp;·&nbsp; AI extracts & classifies &nbsp;·&nbsp;
           Ranked by your profile &nbsp;·&nbsp; <strong>One API call</strong></p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(opps: list[RankedOpportunity], noise: list[RankedOpportunity]):
    high = sum(1 for o in opps if o.urgency_label == "High")
    eligible_n = sum(1 for o in opps if o.is_eligible)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="num" style="color:#58a6ff">{len(opps)}</div>
            <div class="lbl">Opportunities Found</div>
        </div>
        <div class="metric-tile">
            <div class="num" style="color:#f85149">{high}</div>
            <div class="lbl">High Urgency</div>
        </div>
        <div class="metric-tile">
            <div class="num" style="color:#3fb950">{eligible_n}</div>
            <div class="lbl">You're Eligible</div>
        </div>
        <div class="metric-tile">
            <div class="num" style="color:#8b949e">{len(noise)}</div>
            <div class="lbl">Noise Filtered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_bar(score: float):
    pct = int(score * 100)
    color = "#f85149" if score < 0.35 else "#d29922" if score < 0.60 else "#3fb950"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:.7rem;margin:.2rem 0 .7rem;">
        <div class="score-bar-wrap" style="flex:1">
            <div class="score-bar" style="width:{pct}%;background:{color}"></div>
        </div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.8rem;color:{color};min-width:36px">{pct}%</span>
    </div>
    """, unsafe_allow_html=True)


def render_opportunity_card(rank: int, item: RankedOpportunity):
    opp = item.parsed
    card_cls = "opp-card ineligible" if not item.is_eligible else "opp-card"
    urg_cls = URGENCY_BADGE.get(item.urgency_label, "badge-low")

    badges = f'<span class="badge {urg_cls}">{item.urgency_label} Urgency</span>'
    badges += f'<span class="badge badge-type">{opp.type}</span>'
    if not item.is_eligible:
        if item.ineligibility_reason and 'Semester' in item.ineligibility_reason:
            badges += '<span class="badge badge-inelig">⚠ Semester Mismatch</span>'
        else:
            badges += '<span class="badge badge-inelig">⚠ CGPA Mismatch</span>'

    # Deadline display
    if item.days_until_deadline is not None:
        if item.days_until_deadline < 0:
            ddl_str = "🔴 Expired"
        elif item.days_until_deadline == 0:
            ddl_str = "🔴 Due TODAY"
        else:
            ddl_str = f"{opp.deadline_iso} &nbsp;({item.days_until_deadline}d left)"
    else:
        ddl_str = "N/A — deadline unknown"

    checklist_html = "".join(f"<li>{c}</li>" for c in item.action_checklist)
    e_pct = int(item.eligibility_score * 100)
    u_pct = int(item.urgency_score * 100)
    p_pct = int(item.preference_score * 100)

    # Card header (always visible)
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="rank">#{rank} &nbsp;·&nbsp; {opp.organization}</div>
        <div class="title">{opp.title}</div>
        <div>{badges}</div>
    </div>
    """, unsafe_allow_html=True)

    # Expandable detail panel
    with st.expander(f"📋 Details & Action Plan — Match Score: {int(item.score*100)}%", expanded=(rank == 1)):
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(f"**📅 Deadline:** {ddl_str}", unsafe_allow_html=True)
            if opp.min_semester:
                st.markdown(f"**📅 Min Semester Required:** Semester {opp.min_semester}+")
            if opp.min_cgpa:
                st.markdown(f"**🎓 Min CGPA Required:** {opp.min_cgpa:.1f}")
            if opp.action_link:
                st.markdown(f"**🔗 Apply / Register:** [{opp.action_link}]({opp.action_link})")
            if opp.summary:
                st.markdown(f"**📝 Summary:** {opp.summary}")

            st.markdown(f"""
            <div class="why-box">
                💡 <strong>Why it matters:</strong> {item.why_it_matters}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**✅ Practical Action Checklist:**")
            st.markdown(f'<ul class="checklist">{checklist_html}</ul>', unsafe_allow_html=True)

        with col_right:
            st.markdown("**Match Score Breakdown**")
            render_score_bar(item.score)
            st.markdown(f"""
            <div style="font-size:.82rem;color:#8b949e;line-height:1.9">
                Profile Fit &nbsp;<code style="color:#58a6ff">×0.5</code>: <strong>{e_pct}%</strong><br>
                Urgency &nbsp;<code style="color:#58a6ff">×0.3</code>: <strong>{u_pct}%</strong><br>
                Preference &nbsp;<code style="color:#58a6ff">×0.2</code>: <strong>{p_pct}%</strong>
            </div>
            """, unsafe_allow_html=True)

            if opp.required_skills:
                st.markdown(f"**Skills needed:** {', '.join(opp.required_skills[:6])}")
            if opp.required_documents:
                st.markdown(f"**Docs needed:** {', '.join(opp.required_documents[:4])}")
            if item.ineligibility_reason:
                st.error(f"🚫 {item.ineligibility_reason}")


def render_noise_section(noise_items: list[RankedOpportunity]):
    if not noise_items:
        return
    st.markdown("---")
    st.markdown("### 🗑️ Filtered as Noise")
    st.markdown(
        f"<div style='font-size:.85rem;color:#8b949e;margin-bottom:.8rem'>"
        f"{len(noise_items)} email(s) classified as irrelevant and excluded from ranking.</div>",
        unsafe_allow_html=True,
    )
    for item in noise_items:
        label = item.parsed.noise_reason or "Not an opportunity"
        snippet = item.raw_email_snippet[:100].replace("\n", " ").strip()
        st.markdown(f"""
        <div class="noise-card">
            <span class="badge badge-noise">NOISE</span>
            <strong>{label}</strong><br>
            <span style="color:#6e7681;font-size:.8rem">"{snippet}…"</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIDEBAR — STUDENT PROFILE FORM
# ─────────────────────────────────────────────────────────────────────────────

SKILL_OPTIONS = [
    "Python", "JavaScript", "React", "Django", "Node.js", "SQL",
    "Machine Learning", "Data Science", "C++", "Java", "Flutter",
    "Pandas", "TensorFlow", "Docker", "Git", "Figma", "HTML/CSS",
    "Problem Solving", "Data Structures & Algorithms",
]
DEGREE_OPTIONS = [
    "BS Computer Science", "BS Software Engineering",
    "BS Data Science", "BS Electrical Engineering",
    "BS Artificial Intelligence", "BBA / MBA",
    "BS Mathematics", "Other",
]
OPP_TYPES = ["Scholarship", "Internship", "Competition", "Research", "Job"]
LOCATION_OPTIONS = ["Any / No Preference", "Remote/Online", "Lahore", "Karachi", "Islamabad", "Abroad"]


def build_sidebar_profile() -> dict:
    """
    Build the student profile from sidebar widgets.

    BUG FIX: All widget values are stored in st.session_state via their `key`
    parameter. This means profile changes take effect immediately on the NEXT
    run WITHOUT needing a full page refresh. Previously the profile was read
    into local variables that were discarded on each Streamlit rerun triggered
    by the "Analyze" button, causing the sidebar to appear frozen.
    """

    # ── Initialise session_state defaults on very first load ─────────────────
    defaults = {
        "sb_model":      "llama3.2",
        "sb_url":        "http://localhost:11434/api/chat",
        "sb_degree":     DEGREE_OPTIONS[0],
        "sb_semester":   5,
        "sb_cgpa":       3.2,
        "sb_skills":     ["Python", "Machine Learning", "SQL"],
        "sb_pref_types": ["Internship", "Competition"],
        "sb_fin_need":   False,
        "sb_location":   LOCATION_OPTIONS[0],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.markdown("## 🎓 Student Profile")
        st.markdown("---")

        st.markdown("### 🤖 AI Model (Ollama)")
        st.selectbox(
            "Local Model",
            ["llama3.2", "mistral", "gemma3", "llama3.1", "qwen2.5"],
            key="sb_model",
            help="Run: ollama pull llama3.2  (recommended — fast & accurate)",
        )
        st.text_input(
            "Ollama URL",
            key="sb_url",
            help="Default if running Ollama locally. Change if using a remote host.",
        )

        st.markdown("### 🏫 Academic Info")
        st.selectbox("Degree / Program", DEGREE_OPTIONS, key="sb_degree")
        st.slider("Current Semester", 1, 8, key="sb_semester")
        st.slider("CGPA", 0.0, 4.0, step=0.01, format="%.2f", key="sb_cgpa")

        st.markdown("### 💻 Skills")
        st.multiselect("Your Skills", SKILL_OPTIONS, key="sb_skills")

        st.markdown("### 🎯 Preferences")
        st.multiselect("Preferred Opportunity Types", OPP_TYPES, key="sb_pref_types")
        st.checkbox(
            "Financial Need", key="sb_fin_need",
            help="Prioritizes scholarships if checked",
        )
        st.selectbox("Location Preference", LOCATION_OPTIONS, key="sb_location")

        st.markdown("---")

        # Live profile preview so student can confirm values are applied
        st.markdown(
            f"<div style='font-size:.76rem;color:#6e7681;line-height:1.6'>"
            f"<b style='color:#8b949e'>Active profile</b><br>"
            f"CGPA: <b style='color:#58a6ff'>{st.session_state.sb_cgpa:.2f}</b> &nbsp;·&nbsp; "
            f"Sem: <b style='color:#58a6ff'>{st.session_state.sb_semester}</b><br>"
            f"Skills: <b style='color:#58a6ff'>{len(st.session_state.sb_skills)}</b> selected<br>"
            f"Types: <b style='color:#58a6ff'>{', '.join(st.session_state.sb_pref_types) or 'Any'}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:.72rem;color:#484f58;margin-top:.5rem'>"
            "Ranking is fully deterministic Python — the LLM only extracts structured data."
            "</div>",
            unsafe_allow_html=True,
        )

    loc = st.session_state.sb_location
    return {
        "ollama_model":    st.session_state.sb_model,
        "ollama_url":      st.session_state.sb_url,
        "degree":          st.session_state.sb_degree,
        "semester":        st.session_state.sb_semester,
        "cgpa":            st.session_state.sb_cgpa,
        "skills":          st.session_state.sb_skills,
        "preferred_types": st.session_state.sb_pref_types,
        "financial_need":  st.session_state.sb_fin_need,
        "location_pref":   "" if loc.startswith("Any") else loc.lower(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_hero()
    profile = build_sidebar_profile()

    # ── Email input section ──────────────────────────────────────────────────
    st.markdown("### 📧 Paste Emails / Notices")
    st.markdown(
        "<div style='font-size:.84rem;color:#8b949e;margin-bottom:.6rem'>"
        "Paste 5–15 emails separated by blank lines or ──── dividers. "
        "Parsed locally via Ollama — zero cost, no API key, no rate limits."
        "</div>",
        unsafe_allow_html=True,
    )

    # Initialise on first load so the key always exists
    if "email_input" not in st.session_state:
        st.session_state["email_input"] = ""

    col_btn, _ = st.columns([1.2, 4])
    with col_btn:
        if st.button("📂 Load 10 Sample Emails", use_container_width=True):
            # Write directly into the widget's own session_state key.
            # A text_area with key= ignores value= on reruns —
            # the only correct fix is to set session_state[key] then rerun.
            st.session_state["email_input"] = SAMPLE_EMAIL_TEXT
            st.rerun()

    # No value= here — the widget reads from session_state["email_input"] automatically
    email_text = st.text_area(
        label="Emails",
        height=280,
        placeholder="Paste raw email texts here, or click 'Load Sample Emails' above…",
        label_visibility="collapsed",
        key="email_input",
    )

    col_run, col_rerank = st.columns([3, 1])
    with col_run:
        run = st.button("🚀 Analyze & Rank Opportunities", type="primary", use_container_width=True)
    with col_rerank:
        # Re-rank without calling Ollama again — profile changed, results cached
        rerank = st.button(
            "🔄 Re-rank with Current Profile",
            use_container_width=True,
            disabled="parsed_models" not in st.session_state,
            help="Re-applies your updated sidebar profile to already-parsed emails (no Ollama call).",
        )

    if not run and not rerank:
        if "ranked_results" not in st.session_state:
            st.markdown(
                "<div style='text-align:center;color:#6e7681;margin-top:3rem;font-size:.9rem'>"
                "⬆️ Make sure Ollama is running (<code>ollama serve</code>), "
                "fill your profile, paste emails, then click Analyze."
                "</div>",
                unsafe_allow_html=True,
            )
            return
        # Fall through to display cached results if they exist
        run = False
        rerank = True  # Show cached results on page load after first run

    # ── Parse path (calls Ollama) ─────────────────────────────────────────────
    if run:
        if not email_text.strip():
            st.error("📧 Please paste at least one email, or load the sample dataset.")
            return

        # Split emails by dividers or triple blank lines
        raw_emails = re.split(r"\n[─\-=]{3,}\n|\n{3,}", email_text.strip())
        raw_emails = [e.strip() for e in raw_emails if len(e.strip()) > 20]

        if not raw_emails:
            st.error("Could not detect separate emails. Use blank lines or ──── dividers between emails.")
            return

        if len(raw_emails) > 20:
            st.warning(f"Detected {len(raw_emails)} emails. Processing first 20 to keep response time reasonable.")
            raw_emails = raw_emails[:20]

        # ── Progress feedback during chunked parsing ──────────────────────────
        progress_placeholder = st.empty()
        status_text = st.empty()

        progress_bar = progress_placeholder.progress(0, text="Preparing to parse…")

        def update_progress(chunk_idx: int, total_chunks: int, chunk_len: int):
            pct = int((chunk_idx / total_chunks) * 100)
            emails_done = chunk_idx * 3  # chunk_size=3
            progress_bar.progress(
                pct,
                text=f"🦙 Parsing emails {emails_done+1}–{min(emails_done+chunk_len, len(raw_emails))} "
                     f"of {len(raw_emails)} (chunk {chunk_idx+1}/{total_chunks})…",
            )

        try:
            parsed_models = parse_emails_with_ollama(
                raw_emails,
                ollama_url=profile["ollama_url"],
                model=profile["ollama_model"],
                chunk_size=3,
                progress_cb=update_progress,
            )
        except requests.exceptions.ConnectionError:
            progress_placeholder.empty()
            st.error(
                "❌ Cannot connect to Ollama. Make sure it's running:\n\n"
                "```\nollama serve\n```\n"
                "Then pull a model if you haven't already:\n"
                "```\nollama pull llama3.2\n```"
            )
            return
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Ollama error: {e}")
            return

        progress_bar.progress(100, text="✅ Parsing complete!")

        # Cache parsed models and raw snippets — enables re-rank without re-calling LLM
        st.session_state["parsed_models"] = parsed_models
        st.session_state["raw_snippets"]  = [e[:120] for e in raw_emails]

    # ── Re-rank path (uses cached parsed models, no Ollama call) ─────────────
    if "parsed_models" not in st.session_state:
        return

    parsed_models = st.session_state["parsed_models"]
    raw_snippets  = st.session_state["raw_snippets"]

    paired = list(zip(parsed_models, raw_snippets))
    opportunities, noise_items = rank_opportunities(paired, profile)

    # Cache ranked results too
    st.session_state["ranked_results"] = (opportunities, noise_items)

    # ── Render results ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏆 Priority Inbox")

    render_metrics(opportunities, noise_items)

    if not opportunities:
        st.info("No genuine opportunities found in these emails. Try the sample dataset for a demo!")
    else:
        for rank, item in enumerate(opportunities, start=1):
            render_opportunity_card(rank, item)

    render_noise_section(noise_items)

    st.markdown(
        "<div style='text-align:center;color:#6e7681;font-size:.78rem;margin-top:3rem'>"
        "Opportunity Inbox Copilot · SOFTEC 2026 AI Hackathon · "
        "Ranking: deterministic Python · Extraction: Ollama (local LLM) · Zero cost"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
