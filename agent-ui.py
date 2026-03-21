import asyncio
import os
import re
from typing import List, Literal, Optional

import gradio as gr
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

class GuardrailDecision(BaseModel):
    allowed: bool
    reason: str
    reject_reason: Optional[
        Literal[
            "not_homework_domain",
            "unsafe_or_inappropriate",
            "history_trivia_not_homework",
            "too_broad_or_ungrounded_history",
        ]
    ] = None
    confidence: Literal["high", "medium", "low"] = "medium"


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


class ProfileLevelDecision(BaseModel):
    normalized_level: Literal[
        "child",
        "middle_school",
        "high_school",
        "university_year_1",
        "university",
        "general",
    ]
    reason: str


class UserProfile(BaseModel):
    level: str = "general"


# --------------------------------------------------------------------------- #
# Minimal local helpers
# --------------------------------------------------------------------------- #

def describe_level(level: str) -> str:
    mapping = {
        "child": "child / primary-school level",
        "middle_school": "middle-school level",
        "high-school": "high-school level",
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


def build_router_prompt(history: List[dict], user_input: str, max_turns: int = 12) -> str:
    return f"""
Conversation so far:
{build_history_text(history, max_turns=max_turns)}

Current user message:
{user_input}
""".strip()


def build_guardrail_prompt(history: List[dict], user_input: str, max_turns: int = 12) -> str:
    return f"""
Conversation so far:
{build_history_text(history, max_turns=max_turns)}

Current user message:
{user_input}
""".strip()


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

azure_model = build_azure_model()

guardrail_agent = Agent(
    name="Homework Guardrail",
    model=azure_model,
    output_type=GuardrailDecision,
    instructions="""
You are the front-door guardrail for a tutoring assistant.

Your job is ONLY to decide whether the assistant should continue processing
the user input or reject it immediately.

Return allowed=true when the message can continue into the tutoring workflow.

Return allowed=false when the message should be rejected immediately.

The assistant is designed for:
- math
- physics
- chemistry
- history
- summary of the conversation
- profile/background statements from the user
- short conversational smalltalk such as hello / thanks / bye / okay

Allow:
1. Valid tutoring questions in math, physics, chemistry, or history
2. Theoretical math questions
3. Real-world quantitative reasoning that is fundamentally mathematical
4. Requests to summarize the conversation
5. User profile/background statements like "I am a first-year university student"
6. Short smalltalk like greetings, thanks, okay, bye
7. Foundational history questions about major countries, rulers, political systems,
   dynasties, wars, revolutions, and historically important institutions
8. Brief history questions about first presidents, first kings, first queens,
   first emperors, or founders of major states, if they are historically meaningful
9. Questions with spelling mistakes if the intended meaning is clear
10. Questions like "How can I find the centre of a city?" because they can be treated as math
11. Questions like "Who is the first precident of uk?" should be allowed to continue,
    even if the premise is imperfect, because the assistant can later correct it

Reject:
1. Requests outside the tutoring/supported-assistant scope
2. Travel planning and tourism advice
3. Dangerous or unsafe requests
4. Current politics or current office-holders
5. Extremely broad, vague, or weakly grounded pseudo-history requests
6. Narrow local institutional trivia with little educational value in history

Important:
- This is only the first-pass guardrail.
- If a question is potentially valid history, allow it through even if it may later need
  more detailed history scope checking.
- Prefer allow=true when the request appears to fit one of the supported paths.
- Do not reject just because the user is brief.
- Do not reject just because the wording is imperfect.

When rejecting, provide one reject_reason from:
- not_homework_domain
- unsafe_or_inappropriate
- history_trivia_not_homework
- too_broad_or_ungrounded_history
""",
)

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

Use both the current user message and the recent conversation context.

Definitions:
- math: computational math, theoretical math, logic, proofs, algebra, geometry,
  discrete math, abstract algebra, quantitative reasoning, estimation, units,
  coordinates, distance calculation, rate problems, percentage problems,
  geometric reasoning, center/midpoint/centroid questions, map-based coordinate reasoning,
  and real-world mathematical modeling
- physics: mechanics, electricity, magnetism, waves, optics, thermodynamics,
  physical problem solving
- chemistry: atoms, bonds, equations, stoichiometry, acids/bases, pH,
  thermochemistry
- history: educational history questions with meaningful historical context or significance
- summary: explicit request to summarize or recap the conversation
- profile: user states their level/background
- smalltalk: greeting, thanks, okay, bye, acknowledgment, or brief conversational message
- reject: anything else

Important rules:
1. Theoretical math is valid math.
2. Real-world quantitative reasoning is still math.
3. Applied math in real-world settings is math, not a non-homework domain.
4. If the user asks how to compute or calculate something, prefer math when the core task is numerical reasoning.
5. Questions about finding a center, centre, midpoint, centroid, average position,
   geometric center, or representative center of a place, shape, region, or set of points
   are usually math if the user is asking how to determine or calculate it.
6. "How can I find the centre of a city?" should usually be routed to math.
7. "How do I calculate the geometric center of a city on a map?" should be routed to math.
8. Distinguish carefully between:
   - math: "How can I find the centre of a city?"
   - math: "How do I calculate the center of a region on a map?"
   - math: "How do I compute the distance between Hong Kong and Shenzhen?"
   - reject/travel: "What is the best way to travel from Hong Kong to Shenzhen?"
   - reject/travel: "Which district is the city center best for tourists?"
9. Current politics or current office-holders are NOT history homework.
10. Narrow institutional trivia is not valid history tutoring.
11. Overly broad pseudo-history questions with no clear historical grounding should be rejected.
12. Travel planning should be rejected.
13. Harmful or dangerous requests should be rejected.
14. If user says something like "I'm a university student", use profile and extract level if possible.
15. If user asks for a summary/recap, use summary.
16. If the current message is short but depends on recent conversation, infer the intended route from context.

History-specific rule:
17. Foundational factual questions about major countries, major political leaders,
    monarchs, emperors, presidents, prime ministers, dynasties, wars, revolutions,
    empires, republics, kings, queens, and first rulers should usually be routed to history.
18. Questions like "Who was the first president of France?" are history, not reject.
19. Questions about the first ruler, first president, first emperor, first king,
    or first queen of a historically significant state are usually valid history questions.
20. Treat major-country monarchy questions as history even if the wording is brief.
21. Spelling mistakes should not cause rejection if the intended meaning is clear.
22. Examples:
    - "Who is the first precident of uk?" -> history
    - "Who is the first queen of uk?" -> history
    - "Who was the first queen of England?" -> history
23. Do not confuse foundational history with narrow local trivia.

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

Return allowed=true if the question is a valid history tutoring or history-homework-style question.

Allow:
- questions about historically significant people, rulers, states, dynasties, wars, revolutions, governments, and political systems
- questions about causes, effects, significance, comparison, interpretation, and historical context
- foundational factual questions with educational value in history
- short factual questions when they concern historically important figures, institutions, or states
- questions whose wording is imperfect, imprecise, or partially mistaken, if the intended historical meaning is still reasonably clear

Reject with allowed=false only when:
1. the question is clearly outside history
2. the question is mainly about current politics or current office-holders
3. the question is too vague, too broad, or too weakly grounded to support a meaningful historical answer
4. the question is narrow local or institutional trivia with little broader historical value

Important:
- Do not reject merely because the user is brief.
- Do not reject merely because the question is factual.
- Do not reject merely because the wording is imprecise or conceptually imperfect.
- If the intended historical meaning is reasonably clear, allow it.
- Prefer allowing a question that can be answered through brief clarification over rejecting it.

Use reject_reason from:
- history_trivia_not_homework
- too_broad_or_ungrounded_history
- not_homework_domain
""",
)

profile_level_agent = Agent(
    name="Profile Level Normalizer",
    model=azure_model,
    output_type=ProfileLevelDecision,
    instructions="""
You normalize a user's stated academic level/background into one internal category.

Return exactly one normalized_level:
- child
- middle_school
- high_school
- university_year_1
- university
- general

Guidance:
- child: primary school / elementary school / young child level
- middle_school: junior high / middle school
- high_school: secondary school / high school
- university_year_1: explicitly first-year university / freshman / year 1 / 大一
- university: general university / college / undergraduate, when first-year is not explicit
- general: if the level is unclear

Do not write user-facing text here. Only return the structured result.
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
- if the reason is history_trivia_not_homework, explain that it is too narrow or only a fact lookup
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
- coordinates
- distance and coordinates
- center / centre / midpoint / centroid questions
- map-based geometric reasoning
- word problems
- estimation and quantitative reasoning
- introductory and advanced mathematical concepts
- mathematical logic, proofs, axioms, and theoretical foundations
- number systems, set theory, discrete mathematics, and abstract algebra
- real-world mathematical modeling

Style rules:
- Be encouraging, respectful, and human.
- Never shame the user.
- Never say things like "you should already know this."
- Adapt to the user's level when provided.
- When appropriate, you may begin with one short natural sentence identifying the question type.
- Keep explanations concise, clear, and educational.
- If a real-world question can be answered mathematically, explain the mathematical interpretation first.
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
- When appropriate, you may begin with one short natural sentence identifying the question type.
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
- When appropriate, you may begin with one short natural sentence identifying the question type.
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
- historical causes, effects, significance, comparison, and context
- major countries, states, dynasties, wars, revolutions, leaders, institutions, and political systems
- foundational fact questions about historically significant figures, events, and states

Style rules:
- Adapt to the user's level when provided.
- When appropriate, you may begin with one short natural sentence identifying the question type.
- Reply in the same language as the user unless the prompt explicitly indicates otherwise.

Do not answer:
- current politics or current office-holders
- narrow local institutional trivia
- extremely broad and weakly grounded pseudo-history questions

Important:
- If the user's historical premise is inaccurate, incomplete, or uses an imprecise label, do not reject for that reason alone.
- First determine whether there is a clear and educationally meaningful historical question behind the wording.
- If so, briefly correct the imprecision and answer the closest valid historical interpretation.
- Prefer helpful clarification over refusal when the user's intended historical meaning is reasonably clear.
- Reject only when the question is clearly outside history, purely about current politics, or too vague to support a meaningful historical answer.
""",
)


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #

async def run_guardrail(history: List[dict], user_input: str) -> GuardrailDecision:
    prompt = build_guardrail_prompt(history, user_input)
    result = await Runner.run(guardrail_agent, prompt)
    return result.final_output_as(GuardrailDecision)


async def run_router(history: List[dict], user_input: str) -> RouteDecision:
    prompt = build_router_prompt(history, user_input)
    result = await Runner.run(router_agent, prompt)
    return result.final_output_as(RouteDecision)


async def run_history_scope(user_input: str, history: List[dict]) -> HistoryScopeDecision:
    prompt = f"""
Recent conversation:
{build_history_text(history, max_turns=8)}

Current history question:
{user_input}
"""
    result = await Runner.run(history_scope_agent, prompt)
    return result.final_output_as(HistoryScopeDecision)


async def run_subject_agent(
        route: str,
        profile: UserProfile,
        user_input: str,
        history: List[dict],
) -> str:
    prompt = f"""
User level: {describe_level(profile.level)}

Recent conversation:
{build_history_text(history, max_turns=12)}

Current question:
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


async def run_smalltalk_agent(user_input: str, history: List[dict]) -> str:
    prompt = f"""
Recent conversation:
{build_history_text(history, max_turns=8)}

Current user message:
{user_input}
"""
    result = await Runner.run(smalltalk_agent, prompt)
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


async def run_profile_level_agent(user_input: str) -> str:
    result = await Runner.run(profile_level_agent, user_input)
    decision = result.final_output_as(ProfileLevelDecision)
    return decision.normalized_level


async def run_profile_reply_agent(user_input: str, normalized_level: str) -> str:
    prompt = f"""
User input:
{user_input}

Normalized internal level:
{describe_level(normalized_level)}
"""
    result = await Runner.run(profile_reply_agent, prompt)
    return result.final_output


async def run_summary_agent(profile: UserProfile, history: List[dict], user_input: str) -> str:
    prompt = f"""
Current internal user level: {describe_level(profile.level)}

Conversation:
{build_history_text(history, max_turns=16)}

Latest user request:
{user_input}
"""
    result = await Runner.run(summary_agent, prompt)
    return result.final_output


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #

def create_ui():
    """Create and return the Gradio interface."""
    set_tracing_disabled(True)

    with gr.Blocks(title="SmartTutor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 SmartTutor
            ### Your AI homework assistant for Math, Physics, Chemistry, and History
            Ask any question, and I'll help you understand step by step.
            """
        )

        # Chatbot display - using proper message format
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
        )

        # State to hold conversation history and user profile
        # Each element in state_history is a dict with role and content
        state_history = gr.State([])
        state_profile = gr.State(UserProfile().model_dump())

        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask me something... (e.g., 'Solve 2x+3=7', 'Who was the first president of the US?')",
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")

        # Define the asynchronous response function
        async def respond(user_input, history_list, profile_dict):
            if not user_input.strip():
                return "", history_list, profile_dict, []

            # Append user message to history
            history_list.append({"role": "user", "content": user_input})

            # Reconstruct UserProfile from dict
            profile = UserProfile(**profile_dict)

            try:
                # Step 1: guardrail
                guardrail_decision = await run_guardrail(history_list, user_input)

                if DEBUG:
                    print(
                        f"[GUARDRAIL] allowed={guardrail_decision.allowed} | "
                        f"confidence={guardrail_decision.confidence} | "
                        f"reject_reason={guardrail_decision.reject_reason}"
                    )

                if not guardrail_decision.allowed:
                    answer = await run_reject_agent(
                        user_input,
                        guardrail_decision.reject_reason,
                    )
                    history_list.append({"role": "assistant", "content": answer})
                    # Convert to messages format for Chatbot
                    messages = []
                    for msg in history_list:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    return "", history_list, profile.model_dump(), messages

                # Step 2: router
                decision = await run_router(history_list, user_input)

                if DEBUG:
                    print(
                        f"[ROUTE] route={decision.route} | "
                        f"confidence={decision.confidence} | "
                        f"reject_reason={decision.reject_reason}"
                    )

                # Step 3: specialized handling
                if decision.route == "profile":
                    normalized_level = await run_profile_level_agent(
                        decision.extracted_level or user_input
                    )
                    profile.level = normalized_level
                    answer = await run_profile_reply_agent(user_input, profile.level)

                elif decision.route == "summary":
                    answer = await run_summary_agent(profile, history_list, user_input)

                elif decision.route == "smalltalk":
                    answer = await run_smalltalk_agent(user_input, history_list)

                elif decision.route == "history":
                    scope_decision = await run_history_scope(user_input, history_list)

                    if DEBUG:
                        print(
                            f"[HISTORY_SCOPE] allowed={scope_decision.allowed} | "
                            f"confidence={scope_decision.confidence} | "
                            f"reject_reason={scope_decision.reject_reason}"
                        )

                    if not scope_decision.allowed:
                        answer = await run_reject_agent(
                            user_input,
                            scope_decision.reject_reason,
                        )
                    else:
                        answer = await run_subject_agent(
                            "history",
                            profile,
                            user_input,
                            history_list,
                        )

                elif decision.route in {"math", "physics", "chemistry"}:
                    answer = await run_subject_agent(
                        decision.route,
                        profile,
                        user_input,
                        history_list,
                    )

                else:
                    answer = await run_reject_agent(
                        user_input,
                        decision.reject_reason or "not_homework_domain",
                    )

                history_list.append({"role": "assistant", "content": answer})

            except Exception as exc:
                error_msg = f"⚠️ Sorry, an error occurred: {str(exc)}"
                history_list.append({"role": "assistant", "content": error_msg})
                print(f"[ERROR] {exc}")

            # Convert to messages format for Gradio Chatbot
            messages = []
            for msg in history_list:
                messages.append({"role": msg["role"], "content": msg["content"]})

            return "", history_list, profile.model_dump(), messages

        async def clear_conversation(profile_dict):
            # Reset history but keep profile
            new_profile = UserProfile(**profile_dict)
            return [], new_profile.model_dump(), []

        # Wire up the interface
        submit_event = submit_btn.click(
            fn=respond,
            inputs=[msg, state_history, state_profile],
            outputs=[msg, state_history, state_profile, chatbot],
            queue=True,
        )
        msg.submit(
            fn=respond,
            inputs=[msg, state_history, state_profile],
            outputs=[msg, state_history, state_profile, chatbot],
            queue=True,
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[state_profile],
            outputs=[state_history, state_profile, chatbot],
            queue=False,
        )

    return demo


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False)