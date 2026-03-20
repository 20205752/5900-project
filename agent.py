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
    route: Literal["math", "history", "summary", "profile", "reject"]
    reason: str
    reject_reason: Optional[str] = None
    extracted_level: Optional[str] = None


class UserProfile(BaseModel):
    level: str = "general"


azure_model = build_azure_model()

router_agent = Agent(
    name="Router",
    model=azure_model,
    output_type=RouteDecision,
    instructions="""
You are the policy/router for a multi-turn homework tutoring agent.

Allowed:
- math homework questions
- history homework questions
- requests to summarise the conversation so far
- statements about user background / level, e.g. "I'm a university year one student"

Return:
- route="math" for math tutoring
- route="history" for history tutoring
- route="summary" for summarisation requests
- route="profile" when the user is specifying their level/background
- route="reject" otherwise

When rejecting, provide a specific reject_reason:
- "not_homework_domain" for things outside math/history tutoring
- "unsafe_or_inappropriate" for harmful / dangerous requests
- "too_local_or_not_general_history" for very local / niche factual questions that do not fit the intended history-homework scope

Important:
- A valid math/history question does NOT need to explicitly mention "homework".
- Follow the project spirit: answer math and history, reject unrelated requests, and support summary/profile.
""",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    model=azure_model,
    instructions="""
You are a math homework tutor.
Explain clearly, step by step.
Adapt the difficulty to the user's level if provided.
If the user is year-1 university level, keep explanations appropriately introductory but not childish.
Be concise and educational.
""",
)

history_tutor_agent = Agent(
    name="History Tutor",
    model=azure_model,
    instructions="""
You are a history homework tutor.
Answer clearly and accurately.
Adapt the explanation depth to the user's level if provided.
If a question is too niche/local and not appropriate for general history homework, say so briefly.
Be concise and educational.
""",
)

summary_agent = Agent(
    name="Summary Agent",
    model=azure_model,
    instructions="""
Summarise the conversation so far for the user.
Include:
1. user profile/background if mentioned
2. main math/history topics discussed
3. key answers already given
Keep it short and clear.
""",
)


def print_header():
    print("--- SmartTutor CLI ---")
    print("Supports: math, history, profile adaptation, summary, and guarded rejection.")
    print("Type 'exit' or 'quit' to stop.")


def build_history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(empty conversation)"
    parts = []
    for item in history[-12:]:
        parts.append(f"{item['role'].upper()}: {item['content']}")
    return "\n".join(parts)


async def main():
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

        history.append({"role": "user", "content": user_input})

        # step 1: route
        route_result = await Runner.run(router_agent, user_input)
        decision = route_result.final_output_as(RouteDecision)

        # step 2: act by route
        if decision.route == "profile":
            if decision.extracted_level:
                profile.level = decision.extracted_level
            else:
                profile.level = user_input
            answer = f"Understood. I will tailor future answers to this level: {profile.level}."

        elif decision.route == "summary":
            prompt = f"""
Current user level: {profile.level}

Conversation:
{build_history_text(history)}
"""
            result = await Runner.run(summary_agent, prompt)
            answer = result.final_output

        elif decision.route == "math":
            prompt = f"""
User level: {profile.level}

Question:
{user_input}
"""
            result = await Runner.run(math_tutor_agent, prompt)
            answer = result.final_output

        elif decision.route == "history":
            prompt = f"""
User level: {profile.level}

Question:
{user_input}
"""
            result = await Runner.run(history_tutor_agent, prompt)
            answer = result.final_output

        else:
            # reject
            reason_map = {
                "not_homework_domain": "Sorry, I can only help with math and history homework, plus summarising our conversation.",
                "unsafe_or_inappropriate": "Sorry, I cannot help with that request.",
                "too_local_or_not_general_history": "Sorry, that does not look like a suitable general history homework question."
            }
            answer = reason_map.get(
                decision.reject_reason,
                "Sorry, I cannot help with that request."
            )

        print(f"Assistant: {answer}\n")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    asyncio.run(main())