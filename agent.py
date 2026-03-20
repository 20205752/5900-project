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


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")

DEBUG = False


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
- math questions, including both computational and theoretical math
- applied math questions in real-world settings such as distance, geometry,
  estimation, percentages, rates, units, coordinates, city-centre modelling,
  and quantitative reasoning
- abstract math topics such as number systems, logic, proofs, axioms,
  algebraic structures, discrete mathematics, and mathematical foundations
- history questions that resemble general homework topics
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
3. Theoretical mathematics questions are also valid math homework.
   Examples that should be accepted:
   - "Can you explain Peano arithmetic?"
   - "What is proof by induction?"
   - "What is a group in abstract algebra?"
4. Do NOT classify gratitude or acknowledgement as summary.
5. If a question is clearly about history but may contain a false or incorrect premise,
   still route it to history instead of reject.
6. Tolerate minor spelling mistakes and infer the most likely meaning.
7. A valid history homework question should reflect disciplinary features of history,
   not just any fact about the past.
8. Typical history-homework features include one or more of the following:
   - clear historical time/place/person/event context
   - asking for explanation, cause, effect, comparison, significance, or evaluation
   - involving historically significant events, states, dynasties, governments,
     wars, revolutions, social change, famous sites, or major historical figures
   - resembling a teachable classroom question rather than a trivia lookup
9. Foundational fact questions about major countries, major political leaders,
   dynasties, wars, revolutions, or historically significant states are valid history homework,
   even if they are short factual questions.
10. Example: "Who was the first president of France?" should be accepted.
11. Reject narrow historical trivia that looks like a fact lookup rather than homework tutoring.
    Examples include:
    - birthdays of niche institutional figures
    - the first president/dean/head of a specific university
    - the construction year of a local campus building or library
    - local administrative or institutional trivia with little broader historical significance
12. Reject questions that are too broad, vague, or weakly grounded historically.
    Examples include broad social or entertainment topics without a clear time/place/course context.
13. Reject travel planning questions such as:
    "What is the best way to travel from Hong Kong to London?"
14. Reject harmful or dangerous requests.
15. If the user says something like "I'm a university year one student", use route="profile"
    and extract a short user_level.
16. Set confidence:
   - high: the route is very clear
   - medium: probably correct
   - low: ambiguous case

When rejecting, provide one reject_reason from:
- not_homework_domain
- unsafe_or_inappropriate
- history_trivia_not_homework
- too_broad_or_ungrounded_history
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
- For theoretical questions, explain the core idea intuitively first, then add formal details if helpful.
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
- Do not behave like a general-purpose encyclopedia.
- Prioritise questions that resemble teachable homework topics.
- A good history-homework question usually has at least one of these features:
  1. clear historical context in time/place/person/event
  2. asks for cause, effect, comparison, significance, evaluation, or interpretation
  3. concerns historically significant people, states, dynasties, wars, revolutions, or sites
  4. has clear educational value beyond a narrow factual lookup
- Foundational fact questions about major countries, major political leaders,
  dynasties, wars, revolutions, or historically significant states should still be answered,
  even if they are short factual questions.
- If the question contains a false or impossible premise, do not force an answer.
  Instead, politely correct the premise and answer the closest valid interpretation if possible.
- If a question is only a narrow fact lookup with little educational value,
  especially about a school, library, campus building, department, or local institution,
  politely refuse it as outside the intended homework-tutoring scope.
- If a question is too broad, vague, or weakly grounded historically,
  politely refuse it as outside the intended homework-tutoring scope.
- If a question is about leadership or founding roles of a specific university or local institution,
  politely refuse it as not suitable general history homework.
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
    print("Welcome to SmartTutor, your personal math and history homework tutor.")
    print("What can I help you today?")
    print("Type 'exit' or 'quit' to stop.")
    print()


def build_history_text(history: List[dict], max_turns: int = 16) -> str:
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
        "谢谢",
        "多谢",
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
        "总结一下",
        "总结我们目前的对话",
        "总结一下我们聊了什么",
    ]
    return any(k in t for k in keywords)


def smalltalk_reply(text: str) -> str:
    t = text.lower().strip()
    if looks_like_thanks(t):
        return "You're welcome."
    if t in {"hi", "hello", "hey", "你好", "您好"}:
        return "Hello. What math or history homework question would you like help with?"
    if t in {"bye", "goodbye", "see you", "再见"}:
        return "Goodbye."
    return "Okay."


def build_reject_message(reject_reason: Optional[str]) -> str:
    reason_map = {
        "not_homework_domain": (
            "这是一个不属于数学作业或历史作业范畴的问题，所以我不能回答。"
        ),
        "unsafe_or_inappropriate": (
            "这是一个危险或不适当的问题，不属于数学作业或历史作业的正常辅导范围，所以我不能回答。"
        ),
        "history_trivia_not_homework": (
            "这是一个狭窄的历史事实查询问题，而不是典型的历史作业问题，所以我不能回答。"
        ),
        "too_broad_or_ungrounded_history": (
            "这是一个过于宽泛、缺少明确学科背景的问题，不属于具体的数学作业或历史作业问题，所以我不能回答。"
        ),
    }
    return reason_map.get(
        reject_reason,
        "这个问题不属于数学作业或历史作业的范畴，所以我不能回答。"
    )


def looks_like_theoretical_math_question(text: str) -> bool:
    t = text.lower().strip()

    theory_terms = [
        "peano arithmetic", "number theory", "set theory", "logic", "proof",
        "axiom", "axiomatic", "induction", "abstract algebra", "group theory",
        "ring", "field", "topology", "combinatorics", "discrete math",
        "数理逻辑", "公理", "公理化", "证明", "归纳法", "群论", "环", "域",
        "拓扑", "组合数学", "离散数学", "皮亚诺", "皮亚诺算术"
    ]

    math_terms = [
        "math", "mathematics", "algebra", "geometry", "theorem",
        "数学", "代数", "几何", "定理"
    ]

    has_theory = any(x in t for x in theory_terms)
    has_math = any(x in t for x in math_terms)

    return has_theory or has_math


def looks_like_foundational_history_question(text: str) -> bool:
    t = text.lower().strip()

    major_entity_terms = [
        "france", "french", "england", "britain", "united kingdom",
        "china", "chinese", "rome", "roman", "qing dynasty", "ming dynasty",
        "清朝", "明朝", "法国", "英国", "中国", "罗马", "秦朝", "汉朝", "唐朝", "宋朝",
        "royaume-uni", "chine"
    ]

    major_role_terms = [
        "president", "emperor", "king", "queen", "prime minister", "monarch",
        "总统", "皇帝", "国王", "女王", "首相", "君主",
        "président", "empereur", "roi", "reine", "premier ministre", "monarque"
    ]

    major_event_terms = [
        "revolution", "war", "dynasty", "empire", "republic",
        "革命", "战争", "王朝", "帝国", "共和国",
        "révolution", "guerre", "dynastie", "empire", "république"
    ]

    has_major_entity = any(x in t for x in major_entity_terms)
    has_major_role = any(x in t for x in major_role_terms)
    has_major_event = any(x in t for x in major_event_terms)

    return (has_major_entity and has_major_role) or has_major_event


def looks_like_history_trivia_not_homework(text: str) -> bool:
    t = text.lower().strip()

    institution_terms = [
        "university", "college", "school", "library", "campus",
        "department", "institute", "hkust", "tsinghua",
        "hong kong university of science and technology",
        "清华大学", "香港科技大学", "图书馆", "学院", "学校", "大学", "校园", "系"
    ]

    trivia_terms = [
        "first president", "first principal", "first dean", "first head",
        "founding president", "birthday", "date of birth",
        "when was it built", "when was it founded",
        "第一任校长", "生日", "哪年修建", "哪年建成", "哪年建的", "什么时候建",
        "第一任院长", "第一任负责人", "第一任馆长"
    ]

    analysis_terms = [
        "why", "how", "analyze", "analyse", "compare", "evaluation", "significance",
        "为什么", "如何", "分析", "比较", "评价", "意义", "影响", "原因", "后果"
    ]

    has_institution = any(x in t for x in institution_terms)
    has_trivia = any(x in t for x in trivia_terms)
    has_analysis = any(x in t for x in analysis_terms)

    return has_institution and has_trivia and not has_analysis


def looks_like_too_broad_nonhomework_history(text: str) -> bool:
    t = text.lower().strip()

    broad_topic_terms = [
        "娱乐圈", "演艺圈", "影视圈", "明星圈",
        "人类", "社会", "流行文化", "娱乐产业", "文化产业",
        "humanity", "humankind", "mankind", "entertainment industry",
        "pop culture", "society"
    ]

    development_terms = [
        "怎么发展", "如何发展", "是怎么发展的", "发展过程", "如何形成",
        "how did it develop", "how has it developed", "development of"
    ]

    grounding_terms = [
        "france", "china", "uk", "united kingdom", "rome",
        "法国", "中国", "英国", "罗马",
        "清朝", "明朝", "唐朝", "宋朝",
        "工业革命", "辛亥革命", "法国大革命", "鸦片战争",
        "19th century", "20th century", "19世纪", "20世纪",
        "war", "revolution", "dynasty", "empire",
        "战争", "革命", "王朝", "帝国", "圆明园", "故宫"
    ]

    has_broad_topic = any(x in t for x in broad_topic_terms)
    has_development = any(x in t for x in development_terms)
    has_grounding = any(x in t for x in grounding_terms)

    return has_broad_topic and has_development and not has_grounding


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

        history.append({"role": "user", "content": user_input})

        try:
            if looks_like_thanks(user_input):
                answer = smalltalk_reply(user_input)
                print(f"Assistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                continue

            if looks_like_theoretical_math_question(user_input):
                math_prompt = f"""
User level: {profile.level}

Question:
{user_input}
"""
                result = await Runner.run(math_tutor_agent, math_prompt)
                answer = result.final_output
                print(f"Assistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                continue

            if looks_like_too_broad_nonhomework_history(user_input):
                answer = build_reject_message("too_broad_or_ungrounded_history")
                print(f"Assistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                continue

            if looks_like_history_trivia_not_homework(user_input) and not looks_like_foundational_history_question(user_input):
                answer = build_reject_message("history_trivia_not_homework")
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
                if looks_like_too_broad_nonhomework_history(user_input):
                    answer = build_reject_message("too_broad_or_ungrounded_history")
                elif looks_like_history_trivia_not_homework(user_input) and not looks_like_foundational_history_question(user_input):
                    answer = build_reject_message("history_trivia_not_homework")
                else:
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