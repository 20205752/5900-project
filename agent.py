import asyncio
import os
import re
from typing import Dict, List, Literal, Optional

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import BaseModel


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")

DEBUG = False

DEMOS: Dict[str, str] = {
    "demo-profile": "I'm a university year one student, provide your answers accordingly.",
    "demo-math": "How to solve x + 1 = 2 for x?",
    "demo-history": "Who was the first president of France?",
    "demo-distance": "I want to know how to compute the distance between two cities like Hong Kong and Shenzhen.",
    "demo-center": "How do I find the centre of a city?",
    "demo-peano": "Can you explain Peano arithmetic?",
    "demo-practice": "I want to practice calculus for my final in math101, can you give me a few exercises?",
    "demo-summary": "Can you summarise our conversation so far?",
    "demo-reject1": "What is the best way to travel from Hong Kong to London?",
    "demo-reject2": "What would happen if someone throws a firecracker on a busy street?",
    "demo-reject3": "Who was the first president of Hong Kong University of Science and Technology in Hong Kong?",
    "demo-thanks": "That's helpful, thank you.",
    "demo-false-premise": "the first president of the United Kindom",
    "demo-queen": "the first queen of the United Kindom",
}


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


class RouteDecision(BaseModel):
    route: Literal["math", "history", "summary", "profile", "smalltalk", "reject"]
    reason: str
    reject_reason: Optional[str] = None
    extracted_level: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"


class UserProfile(BaseModel):
    level: str = "general"


azure_model = build_azure_model()

router_agent = Agent(
    name="Router",
    model=azure_model,
    output_type=RouteDecision,
    instructions="""
You are the policy/router for a multi-turn homework tutoring agent.

Classify the user's input into exactly one route:
- math
- history
- summary
- profile
- smalltalk
- reject

Allowed:
- math questions, including applied math questions in real-world settings
  such as distance, geometry, estimation, percentages, rates, units,
  coordinates, city-centre modelling, and quantitative reasoning
- history questions
- requests to summarise the conversation so far
- statements about user background / level
- short conversational messages such as thanks / okay / got it / hello / bye

Return:
- route="math" for valid math tutoring questions
- route="history" for valid history tutoring questions
- route="summary" only when the user explicitly asks to summarise or recap the conversation
- route="profile" when the user specifies their level/background
- route="smalltalk" for greetings, thanks, acknowledgements, or short conversational replies
- route="reject" otherwise

Important:
1. A valid math/history question does NOT need to explicitly mention homework.
2. Real-world math questions are still math questions.
   Example: computing the distance between two cities is math, not travel advice.
3. Do NOT classify gratitude or acknowledgement as summary.
4. If a question is clearly about history but may contain a false or incorrect premise,
   still route it to history instead of reject.
5. Tolerate minor spelling mistakes and infer the most likely meaning.
6. Reject travel planning questions such as:
   "What is the best way to travel from Hong Kong to London?"
7. Reject harmful or dangerous requests.
8. Reject very local institutional trivia if it is not a suitable general history homework question.
9. If the user says something like "I'm a university year one student", use route="profile"
   and extract a short user_level.
10. Set confidence:
   - high: the route is very clear
   - medium: probably correct
   - low: ambiguous case

When rejecting, provide one reject_reason from:
- not_homework_domain
- unsafe_or_inappropriate
- too_local_or_not_general_history
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
- practice exercise generation

Style rules:
- Be encouraging and respectful.
- Never shame the user.
- Never say the user should already know something.
- If a topic is basic for the user's level, describe it as foundational rather than trivial.
- If a topic is advanced for the user's level, gently say it is more advanced than typical for that level, but still help.
- Prefer a warm and natural teaching tone.

Teaching rules:
- Explain clearly and step by step when solving.
- For applied questions, first identify the mathematical model, then solve or explain it.
- If the user asks for practice, generate a few suitable exercises.
- Unless the user asks otherwise, do not reveal the full solution immediately for every practice question;
  you may provide exercises first and then offer hints or solutions.
- Adapt the difficulty to the user's level if provided.
- Keep explanations concise, human, and educational.
""",
)

history_tutor_agent = Agent(
    name="History Tutor",
    model=azure_model,
    instructions="""
You are a supportive history homework tutor.

Style rules:
- Be clear, respectful, and concise.
- Adapt the explanation depth to the user's level if provided.
- Keep a helpful and natural tone.
- Tolerate minor spelling mistakes and interpret the user's likely meaning.

Scope rules:
- Answer general history homework questions clearly and accurately.
- If the question contains a false or impossible premise, do not force an answer.
  Instead, politely correct the premise and answer the closest valid interpretation if possible.
- Example: if a country never had a president, say so clearly.
- If a question is very local, narrow, or institutional trivia and is unlikely to be a suitable general history homework question,
  politely say so instead of forcing an answer.
- Do not answer dangerous or unrelated non-history questions.

Teaching rules:
- Give the direct answer first when appropriate.
- If needed, briefly explain why the original wording is inaccurate.
- Keep explanations educational and easy to follow.
""",
)

summary_agent = Agent(
    name="Summary Agent",
    model=azure_model,
    instructions="""
Summarise the conversation so far in a natural, conversational way.

Style rules:
- Be concise and human.
- Do not sound like a report unless the user explicitly asks for a structured summary.
- Mention the user's level only if it is actually relevant.
- Focus on the main topics and helpful answers already given.
- Do not mention minor failures or things the assistant could not help with unless the user explicitly asks for a full review.

If the conversation is short, keep the summary to 2-4 sentences.
""",
)


def print_header() -> None:
    print("--- SmartTutor CLI ---")
    print("Supports: math, history, profile adaptation, summary, smalltalk, and guarded rejection.")
    print("Type 'exit' or 'quit' to stop.")
    print("Demo commands:")
    for key in DEMOS:
        print(f"  {key}")


def build_history_text(history: List[Dict[str, str]], max_turns: int = 16) -> str:
    if not history:
        return "(empty conversation)"
    recent = history[-max_turns:]
    return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent)


def looks_like_thanks(text: str) -> bool:
    t = text.lower().strip()
    phrases = [
        "thanks",
        "thank you",
        "that's helpful",
        "that is helpful",
        "helpful, thank you",
        "got it, thanks",
        "ok thanks",
        "okay thanks",
        "thx",
    ]
    return any(p in t for p in phrases)


def looks_like_summary_request(text: str) -> bool:
    t = text.lower().strip()
    keywords = [
        "summarise",
        "summarize",
        "summary",
        "recap",
        "conversation so far",
        "what have we discussed",
        "what we've discussed",
        "summarise our conversation",
        "summarize our conversation",
    ]
    return any(k in t for k in keywords)


def smalltalk_reply(text: str) -> str:
    t = text.lower().strip()
    if looks_like_thanks(t):
        return "You're welcome."
    if t in {"hi", "hello", "hey"}:
        return "Hello. What math or history question would you like help with?"
    if t in {"bye", "goodbye", "see you"}:
        return "Goodbye."
    return "Okay."


def build_reject_message(reject_reason: Optional[str]) -> str:
    reason_map = {
        "not_homework_domain": (
            "Sorry, I can help with math and history homework, plus summarising our conversation, "
            "but I cannot help with that request."
        ),
        "unsafe_or_inappropriate": (
            "Sorry, I cannot help with that request."
        ),
        "too_local_or_not_general_history": (
            "Sorry, that does not look like a suitable general history homework question."
        ),
    }
    return reason_map.get(
        reject_reason,
        "Sorry, I cannot help with that request."
    )


async def main() -> None:
    set_tracing_disabled(True)
    print_header()

    profile = UserProfile()
    history: List[Dict[str, str]] = []

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
            if looks_like_thanks(user_input):
                answer = smalltalk_reply(user_input)
                print(f"Assistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                continue

            forced_summary = looks_like_summary_request(user_input)

            if forced_summary:
                decision = RouteDecision(
                    route="summary",
                    reason="Explicit summary request matched by local rule.",
                    confidence="high",
                )
            else:
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
                profile.level = decision.extracted_level or user_input
                answer = (
                    f"Understood. I will tailor future answers to this level: {profile.level}."
                )

            elif decision.route == "summary":
                prompt = f"""
Current user level: {profile.level}

Conversation:
{build_history_text(history)}
"""
                result = await Runner.run(summary_agent, prompt)
                answer = result.final_output

            elif decision.route == "smalltalk":
                answer = smalltalk_reply(user_input)

            elif decision.route == "math":
                math_prompt = f"""
User level: {profile.level}

Question:
{user_input}
"""
                result = await Runner.run(math_tutor_agent, math_prompt)
                answer = result.final_output

            elif decision.route == "history":
                history_prompt = f"""
User level: {profile.level}

Question:
{user_input}
"""
                result = await Runner.run(history_tutor_agent, history_prompt)
                answer = result.final_output

            else:
                if decision.confidence == "low":
                    fallback_prompt = f"""
User level: {profile.level}

Question:
{user_input}

If this looks like a valid math tutoring question, answer it as math help.
If it clearly does not belong to math tutoring, reply exactly with:
__REJECT__
"""
                    fallback_result = await Runner.run(math_tutor_agent, fallback_prompt)
                    fallback_answer = str(fallback_result.final_output).strip()
                    if fallback_answer != "__REJECT__":
                        answer = fallback_answer
                    else:
                        answer = build_reject_message(decision.reject_reason)
                else:
                    answer = build_reject_message(decision.reject_reason)

            print(f"Assistant: {answer}\n")
            history.append({"role": "assistant", "content": answer})

        except Exception as exc:
            print(f"[ERROR] Agent run failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())