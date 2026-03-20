"""
Interactive homework helper CLI using Azure OpenAI for all agents.

This version keeps the original Azure configuration style, but moves nearly all
semantic logic to AI agents:
- routing
- history scope checking
- smalltalk replies
- summary replies
- rejection wording
- profile acknowledgement wording

Local Python logic is kept only for:
- Azure initialization
- CLI loop
- lightweight user level normalization/storage
- conversation history assembly
- agent orchestration
"""

import asyncio
import os
import re
from typing import List, Literal, Optional

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Environment / settings
# --------------------------------------------------------------------------- #

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")

DEBUG = False

DEMOS = {
    "demo-history": "Who was the first president of the United States?",
    "demo-math": "Solve 2x + 3 = 15 for x.",
    "demo-life": "What is the meaning of life?",
    "demo-summary": "Can you summarize what we discussed?",
    "demo-thanks": "谢谢",
    "demo-hello": "bonjour",
    "demo-profile": "我是大一学生",
}


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #

def build_azure_model() -> OpenAIChatCompletionsModel:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        raise RuntimeError(
            "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY."
        )

    match = re.match(
        r"(https?://[^/]+)/openai/deployments/([^/]+)/.*api-version=([^&]+)",
        AZURE_OPENAI_ENDPOINT,
    )
    if not match:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT must include deployment name and api-version."
        )

    azure_endpoint = match.group(1)
    deployment_name = match.group(2)
    api_version = match.group(3)

    azure_client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    return OpenAIChatCompletionsModel(
        openai_client=azure_client,
        model=deployment_name,
    )


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class RouteDecision(BaseModel):
    route: Literal[
        "math",
        "physics",
        "chemistry",
        "history",
        "summary",
        "profile",
        "smalltalk",
        "reject",
    ]
    reason: str
    reject_reason: Optional[
        Literal[
            "not_homework_domain",
            "unsafe_or_inappropriate",
            "history_trivia_not_homework",
            "too_broad_or_ungrounded_history",
        ]
    ] = None
    extracted_level: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"


class HistoryScopeDecision(BaseModel):
    allowed: bool
    reason: str
    reject_reason: Optional[
        Literal[
            "history_trivia_not_homework",
            "too_broad_or_ungrounded_history",
            "not_homework_domain",
        ]
    ] = None
    confidence: Literal["high", "medium", "low"] = "medium"


class UserProfile(BaseModel):
    level: str = "general"


# --------------------------------------------------------------------------- #
# Minimal local helpers
# --------------------------------------------------------------------------- #

def normalize_level(level_text: str) -> str:
    """
    Minimal normalization just for internal storage.
    This is intentionally lightweight and not used for user-facing wording.
    """
    t = level_text.lower().strip()

    if any(x in t for x in ["primary", "elementary", "小学生", "小学", "child", "kids"]):
        return "child"
    if any(x in t for x in ["middle school", "junior high", "初中", "middle-school"]):
        return "middle_school"
    if any(x in t for x in ["high school", "secondary school", "高中", "high-school"]):
        return "high_school"
    if any(x in t for x in ["year one", "year 1", "first year", "freshman", "大一"]):
        return "university_year_1"
    if any(x in t for x in ["university", "college", "本科", "大学"]):
        return "university"
    return "general"


def describe_level(level: str) -> str:
    mapping = {
        "child": "child / primary-school level",
        "middle_school": "middle-school level",
        "high_school": "high-school level",
        "university_year_1": "first-year university level",
        "university": "university level",
        "general": "general level",
    }
    return mapping.get(level, "general level")


def build_history_text(history: List[dict], max_turns: int = 16) -> str:
    if not history:
        return "(empty conversation)"
    recent = history[-max_turns:]
    return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent)


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

azure_model = build_azure_model()

router_agent = Agent(
    name="Router",
    model=azure_model,
    output_type=RouteDecision,
    instructions="""
You are the classifier/router for a multi-turn tutoring assistant.

Classify the user's input into exactly one route:
- math
- physics
- chemistry
- history
- summary
- profile
- smalltalk
- reject

Definitions:
- math: computational math, theoretical math, logic, proofs, algebra, geometry, discrete math, abstract algebra
- physics: mechanics, electricity, magnetism, waves, optics, thermodynamics, physical problem solving
- chemistry: atoms, bonds, equations, stoichiometry, acids/bases, pH, thermochemistry
- history: educational history questions with meaningful historical context or significance
- summary: explicit request to summarize or recap the conversation
- profile: user states their level/background
- smalltalk: greeting, thanks, okay, bye, acknowledgment, or brief conversational message
- reject: anything else

Important rules:
1. Theoretical math is valid math.
2. Real-world quantitative reasoning is still math.
3. Current politics or current office-holders are NOT history homework.
4. Narrow institutional trivia is not valid history tutoring.
5. Overly broad pseudo-history questions with no clear historical grounding should be rejected.
6. Travel planning should be rejected.
7. Harmful or dangerous requests should be rejected.
8. If user says something like "I'm a university student", use profile and extract level if possible.
9. If user asks for a summary/recap, use summary.

When rejecting, provide one reject_reason from:
- not_homework_domain
- unsafe_or_inappropriate
- history_trivia_not_homework
- too_broad_or_ungrounded_history
""",
)

history_scope_agent = Agent(
    name="History Scope Checker",
    model=azure_model,
    output_type=HistoryScopeDecision,
    instructions="""
You decide whether a user question is an appropriate history tutoring question.

Return allowed=true if the question is a valid history homework-style question.

Allow:
- historically significant people, states, dynasties, wars, revolutions, governments
- questions about causes, effects, significance, comparison, interpretation
- foundational factual questions about major historical figures/events/states
- questions with educational value in history

Reject with allowed=false when:
1. the question is narrow institutional/local trivia
   examples:
   - first president/dean/head of a university
   - when a campus building or library was built
2. the question is too broad or weakly grounded historically
   examples:
   - vague development of humanity/society/entertainment with no time/place/course context
3. the question is about current politics or current office-holders
4. the question is not really history

Use reject_reason from:
- history_trivia_not_homework
- too_broad_or_ungrounded_history
- not_homework_domain
""",
)

smalltalk_agent = Agent(
    name="Smalltalk Responder",
    model=azure_model,
    instructions="""
You respond to short conversational user messages.

Requirements:
- reply naturally in the same language as the user
- keep the reply very short
- be polite and natural
- handle greetings, thanks, bye messages, acknowledgements, and brief conversational messages
- do not turn the reply into tutoring unless it is natural to briefly invite a homework question
- for greetings, you may briefly ask what math, physics, chemistry, or history question the user wants help with
- for thanks, reply with a short equivalent of "You're welcome"
- for bye, reply with a short goodbye
- for short acknowledgements like "ok", "got it", "好的", "d'accord", reply briefly and naturally

Do not add long explanations.
""",
)

reject_agent = Agent(
    name="Reject Responder",
    model=azure_model,
    instructions="""
You write a short refusal/rejection message for a tutoring assistant.

You will receive:
- the user's original input
- a reject reason code

Your requirements:
- reply in the same language as the user
- be brief, polite, and clear
- explain that the request is outside the assistant's tutoring scope
- if the reason is unsafe_or_inappropriate, make that clear
- if the reason is history_trivia_not_homework, explain that it is too narrow / fact-lookup-like
- if the reason is too_broad_or_ungrounded_history, explain that it is too broad or lacks clear historical grounding
- if the reason is not_homework_domain, explain that it does not fit math / physics / chemistry / history tutoring
- do not be overly verbose
- do not mention internal codes
""",
)

profile_reply_agent = Agent(
    name="Profile Reply Writer",
    model=azure_model,
    instructions="""
You write a very short acknowledgement after the user states their academic level/background.

You will receive:
- the user's original message
- the normalized internal level description

Requirements:
- reply in the same language as the user
- acknowledge the level naturally
- say future explanations will be adjusted to that level
- keep it short and natural
""",
)

summary_agent = Agent(
    name="Summary Agent",
    model=azure_model,
    instructions="""
Summarize the conversation so far in a natural, concise way.
If the conversation is short, keep it to 2-4 sentences.
Write in the same language the user is currently using, unless the conversation strongly suggests another language.
""",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    model=azure_model,
    instructions="""
You are a supportive math homework tutor.

You can help with:
- arithmetic
- algebra
- geometry
- distance and coordinates
- word problems
- estimation and quantitative reasoning
- introductory and advanced mathematical concepts
- mathematical logic, proofs, axioms, and theoretical foundations
- number systems, set theory, discrete mathematics, and abstract algebra

Style rules:
- Be encouraging, respectful, and human.
- Never shame the user.
- Never say things like "you should already know this."
- Adapt to the user's level when provided.
- When appropriate, you may begin with one short natural sentence identifying the question type,
  such as a practice question, a concept explanation question, or a calculation/solving question.
- Do this only when it sounds natural. Do not force it every time.
- Keep explanations concise, clear, and educational.
- Reply in the same language as the user unless the prompt explicitly indicates otherwise.
""",
)

physics_tutor_agent = Agent(
    name="Physics Tutor",
    model=azure_model,
    instructions="""
You are a supportive physics homework tutor.

You can help with:
- mechanics
- energy and momentum
- electricity and magnetism
- waves and optics
- thermodynamics
- units and dimensional analysis

Style rules:
- Be clear, respectful, and concise.
- Adapt to the user's level when provided.
- When appropriate, you may begin with one short natural sentence identifying the question type,
  such as a practice question, a concept explanation question, or a calculation/solving question.
- Do this only when it sounds natural. Do not force it every time.
- Explain the physical model and assumptions when useful.
- Use units consistently.
- Reply in the same language as the user unless the prompt explicitly indicates otherwise.
""",
)

chemistry_tutor_agent = Agent(
    name="Chemistry Tutor",
    model=azure_model,
    instructions="""
You are a supportive chemistry homework tutor.

You can help with:
- atomic structure
- chemical bonding
- balancing equations
- stoichiometry
- acids and bases
- pH
- basic thermochemistry

Style rules:
- Be clear, respectful, and concise.
- Adapt to the user's level when provided.
- When appropriate, you may begin with one short natural sentence identifying the question type,
  such as a practice question, a concept explanation question, or a calculation/solving question.
- Do this only when it sounds natural. Do not force it every time.
- Show steps for balancing and calculations.
- Do not provide dangerous lab instructions.
- Reply in the same language as the user unless the prompt explicitly indicates otherwise.
""",
)

history_tutor_agent = Agent(
    name="History Tutor",
    model=azure_model,
    instructions="""
You are a supportive history homework tutor.

Answer educational history questions clearly and concisely.

Scope:
- historical causes, effects, significance, comparison, context
- major countries, states, dynasties, wars, revolutions, leaders, and political systems
- foundational fact questions about historically significant figures/events are allowed

Style rules:
- Adapt to the user's level when provided.
- When appropriate, you may begin with one short natural sentence identifying the question type,
  such as a foundational history question, a cause-and-effect question, or a significance question.
- Do this only when it sounds natural. Do not force it every time.
- Reply in the same language as the user unless the prompt explicitly indicates otherwise.

Do not answer:
- current politics or current office-holders
- narrow local institutional trivia
- extremely broad and weakly grounded pseudo-history questions

If the user's premise is mistaken, politely correct it and answer the closest valid interpretation.
""",
)


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #

async def run_subject_agent(route: str, profile: UserProfile, user_input: str) -> str:
    prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}
"""

    if route == "math":
        result = await Runner.run(math_tutor_agent, prompt)
    elif route == "physics":
        result = await Runner.run(physics_tutor_agent, prompt)
    elif route == "chemistry":
        result = await Runner.run(chemistry_tutor_agent, prompt)
    elif route == "history":
        result = await Runner.run(history_tutor_agent, prompt)
    else:
        raise ValueError(f"Unsupported subject route: {route}")

    return result.final_output


async def run_smalltalk_agent(user_input: str) -> str:
    result = await Runner.run(smalltalk_agent, user_input)
    return result.final_output


async def run_reject_agent(user_input: str, reject_reason: Optional[str]) -> str:
    prompt = f"""
User input:
{user_input}

Reject reason:
{reject_reason or "not_homework_domain"}
"""
    result = await Runner.run(reject_agent, prompt)
    return result.final_output


async def run_profile_reply_agent(user_input: str, normalized_level: str) -> str:
    prompt = f"""
User input:
{user_input}

Normalized internal level:
{describe_level(normalized_level)}
"""
    result = await Runner.run(profile_reply_agent, prompt)
    return result.final_output


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def print_header() -> None:
    print("Welcome to SmartTutor.")
    print("Ask me a math, physics, chemistry, or history question.")
    print("Type 'exit' or 'quit' to stop.")
    print("Commands:")
    print("  demo-history")
    print("  demo-math")
    print("  demo-life")
    print("  demo-summary")
    print("  demo-thanks")
    print("  demo-hello")
    print("  demo-profile")
    print()


async def main() -> None:
    set_tracing_disabled(True)
    print_header()

    profile = UserProfile()
    history: List[dict] = []

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        if user_input.lower() in DEMOS:
            user_input = DEMOS[user_input.lower()]
            print(f"[INFO] Running demo: {user_input}")

        history.append({"role": "user", "content": user_input})

        try:
            route_result = await Runner.run(router_agent, user_input)
            decision = route_result.final_output_as(RouteDecision)

            if DEBUG:
                print(
                    f"[ROUTE] route={decision.route} | "
                    f"confidence={decision.confidence} | "
                    f"reject_reason={decision.reject_reason} | "
                    f"level={profile.level} | "
                    f"reason={decision.reason}"
                )

            if decision.route == "profile":
                raw_level = decision.extracted_level or user_input
                profile.level = normalize_level(raw_level)
                answer = await run_profile_reply_agent(user_input, profile.level)

            elif decision.route == "summary":
                prompt = f"""
Current internal user level: {describe_level(profile.level)}

Conversation:
{build_history_text(history)}
"""
                result = await Runner.run(summary_agent, prompt)
                answer = result.final_output

            elif decision.route == "smalltalk":
                answer = await run_smalltalk_agent(user_input)

            elif decision.route == "history":
                scope_result = await Runner.run(history_scope_agent, user_input)
                scope_decision = scope_result.final_output_as(HistoryScopeDecision)

                if DEBUG:
                    print(
                        f"[HISTORY_SCOPE] allowed={scope_decision.allowed} | "
                        f"confidence={scope_decision.confidence} | "
                        f"reject_reason={scope_decision.reject_reason} | "
                        f"reason={scope_decision.reason}"
                    )

                if not scope_decision.allowed:
                    answer = await run_reject_agent(user_input, scope_decision.reject_reason)
                else:
                    answer = await run_subject_agent("history", profile, user_input)

            elif decision.route in {"math", "physics", "chemistry"}:
                answer = await run_subject_agent(decision.route, profile, user_input)

            else:
                answer = await run_reject_agent(user_input, decision.reject_reason)

            print(f"Assistant: {answer}\n")
            history.append({"role": "assistant", "content": answer})

        except Exception as exc:
            print(f"[ERROR] Agent run failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())