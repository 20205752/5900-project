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
    reject_reason: Optional[str] = None
    extracted_level: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"


class UserProfile(BaseModel):
    level: str = "general"


def normalize_level(level_text: str) -> str:
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


azure_model = build_azure_model()

router_agent = Agent(
    name="Router",
    model=azure_model,
    output_type=RouteDecision,
    instructions="""
You are the policy/router for a multi-turn homework tutoring agent.

Classify the user's input into exactly one route:
- math
- physics
- chemistry
- history
- summary
- profile
- smalltalk
- reject

Allowed:
- math questions, including both computational and theoretical math
- physics questions (high-school / university level), including mechanics,
  electricity and magnetism, waves and optics, thermodynamics, and
  quantitative problem solving
- chemistry questions (high-school / university level), including atomic and
  bonding structure, balancing chemical equations, stoichiometry, reaction
  types, and basic thermochemistry
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
- route="physics" for valid physics tutoring questions
- route="chemistry" for valid chemistry tutoring questions
- route="history" for valid history tutoring questions
- route="summary" only when the user explicitly asks to summarise or recap the conversation
- route="profile" when the user specifies their level/background
- route="smalltalk" for greetings, thanks, acknowledgements, or short conversational replies
- route="reject" otherwise

Important:
1. A valid math/physics/chemistry/history question does NOT need to explicitly mention homework.
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
- Be encouraging, respectful, and human.
- Never shame the user.
- Never say things like "you should already know this."
- If a topic is basic for the user's level, gently describe it as a foundational topic.
- If a topic is advanced for the user's level, say so gently and still help.

Age/level adaptation:
- child: use very simple words, short sentences, and concrete examples.
- middle_school: explain clearly step by step, avoid too much abstraction.
- high_school: explain both steps and the underlying idea.
- university_year_1: if the topic is very basic, briefly note that it is a foundational concept at this level, then explain efficiently.
- university: be concise, structured, and slightly more formal.
- general: be clear and balanced.

Human tone examples:
- For very basic questions at university_year_1:
  "This is a foundational algebra step, so let’s solve it carefully."
- For advanced topics at university_year_1:
  "This topic is a bit beyond typical first-year material, but I can still give you an intuitive explanation first."

Teaching rules:
- Explain clearly and step by step when solving.
- For applied questions, first identify the mathematical model, then solve or explain it.
- For theoretical questions, explain the core idea intuitively first, then add formal details if helpful.
- If the user asks for practice, generate a few suitable exercises.
- Unless the user asks otherwise, do not reveal the full solution immediately for every practice question;
  you may provide exercises first and then offer hints or solutions.
- Adapt the difficulty and tone to the user's level if provided.
- Keep explanations concise, human, and educational.
""",
)

physics_tutor_agent = Agent(
    name="Physics Tutor",
    model=azure_model,
    instructions="""
You are a supportive physics homework tutor.

You can help with:
- mechanics: kinematics, Newton's laws, forces, free-body diagrams
- energy and momentum: work-energy theorem, conservation ideas
- electricity and magnetism: circuits basics, EM induction concepts (as appropriate)
- waves and optics: wave equations, reflection/refraction (as appropriate)
- basic thermodynamics: heat, temperature, simple laws (as appropriate)
- units, dimensional analysis, and quantitative problem solving
- practice exercise generation

Style rules:
- Be encouraging, respectful, and human.
- Never shame the user.
- Keep explanations concise, human, and educational.

Age/level adaptation:
- child: use very simple words, short sentences, and concrete examples.
- middle_school: explain clearly step by step, avoid too much abstraction.
- high_school: explain both steps and the underlying idea.
- university_year_1: if the topic is very basic, briefly note that it is a foundational concept at this level, then explain efficiently.
- university: be concise, structured, and slightly more formal.
- general: be clear and balanced.

Teaching rules:
- Explain step by step. If needed, identify the physical model (e.g., projectile motion, forces, energy approach).
- When solving word problems, extract known/unknown variables and state key assumptions (idealizations).
- Use units consistently; show key conversions when relevant.
- If the user asks for practice, generate a few exercises (with hints first unless they request full solutions).

Safety / scope:
- If the user requests harmful or dangerous instructions (e.g., weaponization), refuse politely and steer back to safe physics study.
""",
:)

chemistry_tutor_agent = Agent(
    name="Chemistry Tutor",
    model=azure_model,
    instructions="""
You are a supportive chemistry homework tutor.

You can help with:
- atomic structure, periodic table trends, and chemical bonding
- balancing chemical equations
- stoichiometry and reaction calculations (moles, limiting reagent, yield)
- types of reactions and basic reaction mechanisms (at a homework-appropriate level)
- acids and bases, and basic pH reasoning (as appropriate)
- basic thermochemistry ideas (endothermic/exothermic, simple enthalpy reasoning)
- practice exercise generation

Style rules:
- Be clear, respectful, and concise.
- Never shame the user.
- Keep explanations educational and easy to follow.

Age/level adaptation:
- child: use very simple words, short sentences, and concrete examples.
- middle_school: explain the main ideas and steps clearly.
- high_school: explain both the procedure and the chemistry reasoning behind it.
- university_year_1: if the topic is very basic, briefly note that it is a foundational concept at this level, then explain efficiently.
- university: be more structured and slightly more formal.
- general: be clear and balanced.

Teaching rules:
- Start by identifying the topic (e.g., stoichiometry vs balancing equations).
- For reaction questions: show how to balance the equation, then do the stoichiometry steps.
- For calculations: show units and keep the math consistent.
- If the user asks for practice, generate a few exercises suitable for their level (with hints first unless they request full answers).

Safety / scope:
- Do not provide instructions for making harmful substances.
- If the user requests dangerous lab procedures, provide high-level educational guidance and safety-oriented refusal.
""",
:)

history_tutor_agent = Agent(
    name="History Tutor",
    model=azure_model,
    instructions="""
You are a supportive history homework tutor.

Style rules:
- Be clear, respectful, and concise.
- Keep a helpful and natural tone.
- Never shame the user.
- Tolerate minor spelling mistakes and interpret the user's likely meaning.

Age/level adaptation:
- child: use simple language and focus on the basic story.
- middle_school: explain the main event, people, and outcome clearly.
- high_school: explain context, causes, and consequences.
- university_year_1: if the question is very basic, briefly note that it is a foundational history topic, then answer clearly.
- university: be more analytical and structured.
- general: be clear and balanced.

Human tone examples:
- For basic questions at university_year_1:
  "This is a foundational topic in political history, so let’s answer it clearly."
- For more advanced questions:
  "This goes a bit beyond an introductory overview, but I can explain the main historical interpretation."

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
    print("Welcome to SmartTutor, your personal math, physics and chemistry homework tutor (plus history).")
    print("What can I help you today?")
    print("Type 'exit' or 'quit' to stop.")
    print()


def build_history_text(history: List[dict], max_turns: int = 16) -> str:
    if not history:
        return "(empty conversation)"
    recent = history[-max_turns:]
    return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent)


def detect_response_language(text: str) -> str:
    t = text.strip()

    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"

    lower_t = f" {t.lower()} "
    french_markers = [
        " bonjour ", " merci ", " président ", " premier ", " première ",
        " france ", " royaume ", " qui ", " été ", " le ", " la ", " de la ",
        " devoir ", " mathématiques ", " histoire ", " quel ", " quelle ",
        " comment ", " pourquoi ", " qu'"
    ]
    if any(marker in lower_t for marker in french_markers):
        return "fr"

    return "en"


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
        "merci",
        "merci beaucoup",
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
        "résume",
        "résumer",
        "résumé",
        "récapitule",
    ]
    return any(k in t for k in keywords)


def smalltalk_reply(text: str) -> str:
    lang = detect_response_language(text)
    t = text.lower().strip()

    if looks_like_thanks(t):
        if lang == "zh":
            return "不客气。"
        if lang == "fr":
            return "Je vous en prie."
        return "You're welcome."

    if t in {"hi", "hello", "hey", "你好", "您好", "bonjour", "salut"}:
        if lang == "zh":
            return "你好。你想让我帮你解答什么数学、物理、化学或历史作业问题？"
        if lang == "fr":
            return "Bonjour. Quelle question de devoir de mathématiques, de physique, de chimie ou d’histoire voulez-vous que j’explique ?"
        return "Hello. What math, physics, chemistry, or history homework question would you like help with?"

    if t in {"bye", "goodbye", "see you", "再见", "au revoir"}:
        if lang == "zh":
            return "再见。"
        if lang == "fr":
            return "Au revoir."
        return "Goodbye."

    if lang == "zh":
        return "好的。"
    if lang == "fr":
        return "D’accord."
    return "Okay."


def build_reject_message(reject_reason: Optional[str], user_input: str) -> str:
    lang = detect_response_language(user_input)

    reason_map = {
        "zh": {
            "not_homework_domain": "这是一个不属于数学、物理、化学或历史作业范畴的问题，所以我不能回答。",
            "unsafe_or_inappropriate": "这是一个危险或不适当的问题，不属于数学、物理、化学或历史作业的正常辅导范围，所以我不能回答。",
            "history_trivia_not_homework": "这是一个狭窄的历史事实查询问题，而不是典型的历史作业问题，所以我不能回答。",
            "too_broad_or_ungrounded_history": "这是一个过于宽泛、缺少明确学科背景的问题，不属于具体的数学、物理、化学或历史作业问题，所以我不能回答。",
        },
        "en": {
            "not_homework_domain": "This is not a math, physics, chemistry, or history homework question, so I cannot answer it.",
            "unsafe_or_inappropriate": "This is a dangerous or inappropriate question, and it does not belong to normal math/physics/chemistry/history homework tutoring, so I cannot answer it.",
            "history_trivia_not_homework": "This is a narrow historical fact-lookup question rather than a typical history homework question, so I cannot answer it.",
            "too_broad_or_ungrounded_history": "This question is too broad and does not have a clear math/physics/chemistry/history homework context, so I cannot answer it.",
        },
        "fr": {
            "not_homework_domain": "Ce n’est pas une question de devoir de mathématiques, de physique, de chimie ou d’histoire, donc je ne peux pas y répondre.",
            "unsafe_or_inappropriate": "C’est une question dangereuse ou inappropriée, qui ne relève pas d’un accompagnement normal en devoirs de mathématiques, de physique, de chimie ou d’histoire, donc je ne peux pas y répondre.",
            "history_trivia_not_homework": "C’est une question de fait historique très étroite, et non une question typique de devoir d’histoire, donc je ne peux pas y répondre.",
            "too_broad_or_ungrounded_history": "Cette question est trop large et n’a pas de contexte clair de devoir de mathématiques, de physique, de chimie ou d’histoire, donc je ne peux pas y répondre.",
        },
    }

    fallback_map = {
        "zh": "这个问题不属于数学、物理、化学或历史作业的范畴，所以我不能回答。",
        "en": "This question does not fall within math/physics/chemistry/history homework tutoring, so I cannot answer it.",
        "fr": "Cette question ne relève pas d’un devoir de mathématiques, de physique, de chimie ou d’histoire, donc je ne peux pas y répondre.",
    }

    lang_map = reason_map.get(lang, reason_map["en"])
    return lang_map.get(reject_reason, fallback_map.get(lang, fallback_map["en"]))


def looks_like_theoretical_math_question(text: str) -> bool:
    t = text.lower().strip()

    theory_terms = [
        "peano arithmetic", "number theory", "set theory", "logic", "proof",
        "axiom", "axiomatic", "induction", "abstract algebra", "group theory",
        "ring", "field", "topology", "combinatorics", "discrete math",
        "数理逻辑", "公理", "公理化", "证明", "归纳法", "群论", "环", "域",
        "拓扑", "组合数学", "离散数学", "皮亚诺", "皮亚诺算术",
        "arithmétique de peano", "théorie des nombres", "théorie des ensembles",
        "logique", "preuve", "axiome", "algèbre abstraite", "théorie des groupes",
        "anneau", "corps", "topologie", "combinatoire", "mathématiques discrètes"
    ]

    return any(x in t for x in theory_terms)


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
        "清华大学", "香港科技大学", "图书馆", "学院", "学校", "大学", "校园", "系",
        "université", "bibliothèque", "campus", "département", "institut"
    ]

    trivia_terms = [
        "first president", "first principal", "first dean", "first head",
        "founding president", "birthday", "date of birth",
        "when was it built", "when was it founded",
        "第一任校长", "生日", "哪年修建", "哪年建成", "哪年建的", "什么时候建",
        "第一任院长", "第一任负责人", "第一任馆长",
        "premier président", "premier doyen", "date de naissance",
        "quand a-t-il été construit", "quand a-t-il été fondé"
    ]

    analysis_terms = [
        "why", "how", "analyze", "analyse", "compare", "evaluation", "significance",
        "为什么", "如何", "分析", "比较", "评价", "意义", "影响", "原因", "后果",
        "pourquoi", "comment", "analyser", "comparer", "importance", "signification"
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
        "pop culture", "society",
        "industrie du divertissement", "culture populaire", "société", "humanité"
    ]

    development_terms = [
        "怎么发展", "如何发展", "是怎么发展的", "发展过程", "如何形成",
        "how did it develop", "how has it developed", "development of",
        "comment cela s’est développé", "comment s’est développé", "développement de"
    ]

    grounding_terms = [
        "france", "china", "uk", "united kingdom", "rome",
        "法国", "中国", "英国", "罗马",
        "清朝", "明朝", "唐朝", "宋朝",
        "工业革命", "辛亥革命", "法国大革命", "鸦片战争",
        "19th century", "20th century", "19世纪", "20世纪",
        "war", "revolution", "dynasty", "empire",
        "战争", "革命", "王朝", "帝国", "圆明园", "故宫",
        "19e siècle", "20e siècle", "guerre", "révolution", "dynastie", "empire"
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
User level: {describe_level(profile.level)}

Question:
{user_input}
"""
                result = await Runner.run(math_tutor_agent, math_prompt)
                answer = result.final_output
                print(f"Assistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                continue

            if looks_like_too_broad_nonhomework_history(user_input):
                answer = build_reject_message("too_broad_or_ungrounded_history", user_input)
                print(f"Assistant: {answer}\n")
                history.append({"role": "assistant", "content": answer})
                continue

            if looks_like_history_trivia_not_homework(user_input) and not looks_like_foundational_history_question(user_input):
                answer = build_reject_message("history_trivia_not_homework", user_input)
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
                lang = detect_response_language(user_input)
                raw_level = decision.extracted_level or user_input
                profile.level = normalize_level(raw_level)

                if lang == "zh":
                    answer = f"明白了。接下来我会按照这个水平来调整回答：{describe_level(profile.level)}。"
                elif lang == "fr":
                    answer = f"Compris. J’adapterai désormais mes réponses à ce niveau : {describe_level(profile.level)}."
                else:
                    answer = f"Understood. I will tailor future answers to this level: {describe_level(profile.level)}."

            elif decision.route == "summary":
                prompt = f"""
Current user level: {describe_level(profile.level)}

Conversation:
{build_history_text(history)}
"""
                result = await Runner.run(summary_agent, prompt)
                answer = result.final_output

            elif decision.route == "smalltalk":
                answer = smalltalk_reply(user_input)

            elif decision.route == "math":
                math_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}
"""
                result = await Runner.run(math_tutor_agent, math_prompt)
                answer = result.final_output

            elif decision.route == "physics":
                physics_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}
"""
                result = await Runner.run(physics_tutor_agent, physics_prompt)
                answer = result.final_output

            elif decision.route == "chemistry":
                chemistry_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}
"""
                result = await Runner.run(chemistry_tutor_agent, chemistry_prompt)
                answer = result.final_output

            elif decision.route == "history":
                if looks_like_too_broad_nonhomework_history(user_input):
                    answer = build_reject_message("too_broad_or_ungrounded_history", user_input)
                elif looks_like_history_trivia_not_homework(user_input) and not looks_like_foundational_history_question(user_input):
                    answer = build_reject_message("history_trivia_not_homework", user_input)
                else:
                    history_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}
"""
                    result = await Runner.run(history_tutor_agent, history_prompt)
                    answer = result.final_output

            else:
                if decision.confidence == "low":
                    # Router was uncertain and chose `reject`. Try each tutor in turn
                    # to recover legitimate homework questions.
                    answer = None

                    # 1) Math
                    math_fallback_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}

If this looks like a valid math tutoring question, answer it as math help.
If it clearly does not belong to math tutoring, reply exactly with:
__REJECT__
"""
                    math_fallback_result = await Runner.run(
                        math_tutor_agent, math_fallback_prompt
                    )
                    math_fallback_answer = str(
                        math_fallback_result.final_output
                    ).strip()
                    if math_fallback_answer != "__REJECT__":
                        answer = math_fallback_answer

                    # 2) Physics
                    if answer is None:
                        physics_fallback_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}

If this looks like a valid physics tutoring question, answer it as physics help.
If it clearly does not belong to physics tutoring, reply exactly with:
__REJECT__
"""
                        physics_fallback_result = await Runner.run(
                            physics_tutor_agent, physics_fallback_prompt
                        )
                        physics_fallback_answer = str(
                            physics_fallback_result.final_output
                        ).strip()
                        if physics_fallback_answer != "__REJECT__":
                            answer = physics_fallback_answer

                    # 3) Chemistry
                    if answer is None:
                        chemistry_fallback_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}

If this looks like a valid chemistry tutoring question, answer it as chemistry help.
If it clearly does not belong to chemistry tutoring, reply exactly with:
__REJECT__
"""
                        chemistry_fallback_result = await Runner.run(
                            chemistry_tutor_agent, chemistry_fallback_prompt
                        )
                        chemistry_fallback_answer = str(
                            chemistry_fallback_result.final_output
                        ).strip()
                        if chemistry_fallback_answer != "__REJECT__":
                            answer = chemistry_fallback_answer

                    # 4) History
                    if answer is None:
                        history_fallback_prompt = f"""
User level: {describe_level(profile.level)}

Question:
{user_input}

If this looks like a valid history tutoring question, answer it as history help.
If it clearly does not belong to history tutoring, reply exactly with:
__REJECT__
"""
                        history_fallback_result = await Runner.run(
                            history_tutor_agent, history_fallback_prompt
                        )
                        history_fallback_answer = str(
                            history_fallback_result.final_output
                        ).strip()
                        if history_fallback_answer != "__REJECT__":
                            answer = history_fallback_answer

                    if answer is None:
                        answer = build_reject_message(decision.reject_reason, user_input)
                else:
                    answer = build_reject_message(decision.reject_reason, user_input)

            print(f"Assistant: {answer}\n")
            history.append({"role": "assistant", "content": answer})

        except Exception as exc:
            print(f"[ERROR] Agent run failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())