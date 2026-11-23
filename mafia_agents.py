# =====================================
# mafia_agents.py : LLM 플레이어(Controller) + 사회자(Moderator)
# =====================================
# 원칙: LLM 은 "생각/말/선택"만 하고, 규칙 판정과 정보 분리는 mafia_core(코드)가 맡는다.
# LLMPlayer 는 PlayerView(본인이 볼 수 있는 정보)만 받는다.
# 추론 재료(투표표, 처형된 시민에게 투표한 사람 등)는 코드가 계산하고, LLM 은 해석만 한다.
import re
import dataclasses
from typing import Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import create_agent

import config
import rag
import mafia_core
from mafia_core import NightDecision, Speech, VoteDecision

# =====================================
# 1. LLM 설정 + 호출 횟수 카운터
# =====================================
LLM_CALLS = {"count": 0}   # 프로세스 전체 LLM 호출 수
CONSISTENCY = {"violations": 0, "retried": 0, "repeat_retries": 0}   # 일관성 위반 / 재질문 / 반복 재생성 수

ACT_TEMPERATURE = 0.3      # 판단 행동(밤 행동, 투표): 낮게
SPEAK_TEMPERATURE = 0.8    # 발언: 다양성


class CallCounter(BaseCallbackHandler):
    """LLM 이 실제로 호출될 때마다 횟수를 센다(에이전트 내부 호출 포함)."""

    def __init__(self, state=None):
        self.state = state

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # state.stats["llm_calls"] 는 mafia_core 가 올리므로 여기서는 모듈 카운터만 증가(중복 방지)
        LLM_CALLS["count"] += 1


def make_llm(temperature=config.LLM_TEMPERATURE):
    # reasoning=False : qwen3 의 think 를 꺼서 속도를 확보
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=temperature,
        reasoning=False,
        base_url=config.OLLAMA_BASE_URL,
        num_predict=400,
    )


# =====================================
# 2. 구조화 출력 스키마 (Pydantic)
# =====================================
class TargetChoice(BaseModel):
    target: str = Field(description="선택한 대상의 이름(후보 중 하나, 자기 자신이면 자기 이름)")
    reason: str = Field(description="한 문장 이유")


class SuspicionItem(BaseModel):
    name: str = Field(description="플레이어 이름")
    score: int = Field(description="마피아일 것 같은 정도 0~100")


class SpeakOut(BaseModel):
    thought: str = Field(description="속마음(남에게 안 보임) 1~2문장. 누가 왜 수상한지에 대한 솔직한 추리(마피아라면 위장 전략). 공개 발언(speech)을 그대로 반복하지 말 것")
    goal: Literal["지목", "변호", "질문", "정보공개"] = Field(description="이번 발언의 목표")
    suspicion: list[SuspicionItem] = Field(description="살아있는 다른 플레이어별 의심도")
    speech: str = Field(description="공개 발언 2~3문장, 한국어. 다른 사람 이름을 반드시 포함")


# =====================================
# 3. 코드가 계산하는 "추론 재료" (PlayerView 의 공개 정보만 사용)
# =====================================
LINE_RE = re.compile(r"^\[(\d+)일차 (낮|투표|재투표)\] (\S+?)(?:: | -> )(.*)$")


def parse_log(view):
    """public_log_text 를 (발언 목록, 투표 목록)으로 바꾼다."""
    speeches, votes = [], []
    for line in (view.public_log_text or "").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        day, kind, who, rest = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if kind == "낮":
            speeches.append({"day": day, "who": who, "text": rest, "line": line})
        else:
            votes.append({"day": day, "kind": kind, "voter": who, "target": rest.strip()})
    return speeches, votes


def bigram_sim(a, b):
    """문자 bigram 자카드 유사도(0~1). 반복 발언 감지와 평가에 사용."""
    def grams(s):
        s = re.sub(r"\s+", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def private_facts(view):
    """private_info 에서 (동료 마피아 목록, 마피아 아님 확인 목록, 마피아 확인 목록)을 뽑는다."""
    names = list(view.alive) + [d["name"] for d in view.dead]
    partners, cleared, found = [], [], []
    for line in view.private_info:
        hit = [n for n in names if n in line and n != view.name]
        if "동료" in line:
            partners += [n for n in hit if n not in partners]
        elif "조사 결과" in line and hit:
            n = hit[0]
            # 코어 문구: "...는 마피아가 아닙니다" / "...는 마피아입니다" ('아닙니다'에는 '아니'가 없으니 모두 확인)
            if any(k in line for k in ("아닙니다", "아니다", "아니에요", "아님", "아니")):
                if n not in cleared:
                    cleared.append(n)
            elif n not in found:
                found.append(n)
    return partners, cleared, found


def reasoning_material(view):
    """사망자/투표표/처형된 시민에게 투표한 사람 등 코드가 계산한 사실만 정리한다."""
    speeches, votes = parse_log(view)
    lines = []
    # 주의: "투표 기록 없음" 같은 문장을 넣으면 8B 모델이 "어제 아무도 투표 안 했다"는 환각을 만든다.
    #       그래서 첫 토론일 때는 없는 것을 나열하지 않고, 근거로 쓸 수 있는 것만 알려준다.
    if not any(s["day"] < view.day for s in speeches) and not votes:
        lines.append("오늘은 게임의 첫 낮입니다. 근거로 쓸 수 있는 것은 아침 발표(밤사이 사망자)와 오늘 나온 발언뿐입니다.")
    if view.dead:
        lines.append("사망/처형: " + ", ".join(
            "%s(%s,%s)" % (d["name"], d["role_ko"], d.get("cause", "")) for d in view.dead))
    by_day = {}
    for v in votes:
        by_day.setdefault((v["day"], v["kind"]), []).append(v)
    for (day, kind), vs_ in sorted(by_day.items()):
        lines.append("%d일차 %s: " % (day, kind) + ", ".join("%s->%s" % (v["voter"], v["target"]) for v in vs_))
    for d in view.dead:
        if d.get("cause") == "처형" and d["role_ko"] == "시민":
            who = sorted({v["voter"] for v in votes if v["target"] == d["name"]})
            if who:
                lines.append("처형된 시민 %s 에게 투표한 사람: %s" % (d["name"], ", ".join(who)))
    today = [s for s in speeches if s["day"] == view.day]
    if view.phase == "day" or today:
        lines.append("오늘 이미 발언한 사람: " + (", ".join(s["who"] for s in today) or "아직 없음(내가 첫 발언)"))
    return "\n".join(lines) if lines else "아직 없음"


def facts_block(view):
    """본인만 아는 확실한 사실을 맨 위에 강조한다."""
    partners, cleared, found = private_facts(view)
    lines = ["★★ 확실한 사실 (나만 앎, 절대 어기지 말 것) ★★", "- 내 직업: " + view.role_ko]
    for p in view.private_info:
        lines.append("- " + p)
    if view.role == "mafia" and partners:
        lines.append("- 내 동료 마피아: %s (의심 금지, 투표 금지, 의심도는 0~10으로 낮게)" % ", ".join(partners))
    if view.role == "police":
        if cleared:
            lines.append("- 마피아가 아님이 확정된 사람: %s (의심/투표 금지, 의심도 0~10)" % ", ".join(cleared))
        if found:
            lines.append("- 마피아로 확정된 사람: %s (조사 결과를 공개하고 이 사람에게 투표하도록 몰아가기)" % ", ".join(found))
    return "\n".join(lines)


# =====================================
# 4. LLMPlayer
# =====================================
ACTION_INFO = {
    "kill": ("choose_kill_target", "오늘 밤 죽일 대상을 고르세요."),
    "protect": ("protect", "오늘 밤 지킬 대상을 고르세요."),
    "investigate": ("investigate", "오늘 밤 조사할 대상을 고르세요."),
    "vote": ("vote", "처형할 사람에게 투표하세요."),
    "revote": ("vote", "재투표입니다. 처형할 사람에게 투표하세요."),
}
TOOL_DESC = {
    "choose_kill_target": "마피아가 오늘 밤 죽일 대상을 정한다.",
    "protect": "의사가 오늘 밤 보호할 대상을 정한다.",
    "investigate": "경찰이 오늘 밤 조사할 대상을 정한다.",
    "vote": "처형 투표 대상을 정한다.",
}
ACTION_HINT = {
    "kill": "경찰/의사로 보이는 사람, 논리적으로 추리해 주목받는 사람 등 시민팀에 위협이 되는 사람을 노리세요.",
    "protect": "마피아가 노릴 것 같은 사람(경찰로 보이는 사람, 논리적으로 추리해 주목받는 사람, 또는 자기 자신)을 보호하세요. 의심스러운 사람을 보호하는 것이 아닙니다.",
    "investigate": "아직 조사하지 않았고 발언이나 투표가 수상한 사람을 고르세요.",
    "vote": "발언과 투표표를 근거로 마피아일 가능성이 가장 높은 사람에게 투표하세요.",
    "revote": "발언과 투표표를 근거로 마피아일 가능성이 가장 높은 사람에게 투표하세요.",
}


class LLMPlayer:
    """mafia_core.Controller 구현체. 역할별 도구를 가진 LLM 플레이어."""

    _strategy_cache = {}   # 역할 -> 전략 텍스트 (플레이어 간 공유, 호출 절약)
    _persona_cache = {}    # 이름 -> 페르소나 텍스트

    def __init__(self, name, role, vs, llm=None, use_tools=True, state=None, speak_llm=None):
        self.name = name
        self.role = role
        self.vs = vs
        # 판단 행동은 낮은 온도, 발언은 높은 온도. llm 을 직접 넘기면 둘 다 그것을 쓴다(테스트용).
        self.llm = llm or make_llm(ACT_TEMPERATURE)
        self.speak_llm = speak_llm or llm or make_llm(SPEAK_TEMPERATURE)
        self.use_tools = use_tools
        self.state = state
        self.counter = CallCounter(state)
        self.cfg = {"callbacks": [self.counter]}
        self._view = None          # read_my_notes 도구가 참조하는 현재 view
        self.last_source = ""      # 마지막 결정 경로: agent / structured / none
        self.consistency_warnings = 0   # 끝까지 일관성을 어긴 횟수(공개 속성)
        self.persona = self._get_persona()
        self.strategy = self._get_strategy()
        self.tools = self._make_common_tools()

    # ----- 4-1. 캐시된 페르소나/전략 -----
    def _get_persona(self):
        if self.name not in LLMPlayer._persona_cache:
            try:
                text = rag.get_persona(self.vs, self.name)
            except Exception:
                text = ""
            LLMPlayer._persona_cache[self.name] = text[:450]
        return LLMPlayer._persona_cache[self.name]

    def _get_strategy(self):
        if self.role not in LLMPlayer._strategy_cache:
            try:
                docs = rag.search_strategy(self.vs, self.role, "토론 발언과 투표 전략", k=2)
                text = "\n".join(d.page_content for d in docs)
            except Exception:
                text = ""
            LLMPlayer._strategy_cache[self.role] = text[:600]
        return LLMPlayer._strategy_cache[self.role]

    # ----- 4-2. 공통 도구 -----
    def _make_common_tools(self):
        role, vs = self.role, self.vs

        @tool
        def search_strategy(query: str) -> str:
            """내 역할의 전략 가이드를 검색한다."""
            docs = rag.search_strategy(vs, role, query, k=2)
            return "\n".join(d.page_content for d in docs)

        @tool
        def read_my_notes() -> str:
            """내가 예전에 적어둔 속마음 기록을 읽는다."""
            v = self._view
            if v is None or not v.my_notes:
                return "기록 없음"
            return "\n".join("D%s %s" % (n.get("day"), n.get("thought", "")) for n in v.my_notes[-5:])

        return {"search_strategy": search_strategy, "read_my_notes": read_my_notes}

    # ----- 4-3. PlayerView -> 프롬프트 텍스트 -----
    def _log_without_today_speech(self, view):
        """오늘 발언은 따로 보여주므로 공개 기록에서는 뺀다."""
        speeches, _ = parse_log(view)
        today = {s["line"].strip() for s in speeches if s["day"] == view.day}
        return "\n".join(ln for ln in (view.public_log_text or "").splitlines() if ln.strip() not in today)

    def _view_text(self, view):
        dead = ", ".join("%s(%s,%s)" % (d["name"], d["role_ko"], d.get("cause", "")) for d in view.dead) or "없음"
        parts = [
            facts_block(view),
            "[상황] %d일차 %s | 나: %s (%s)" % (view.day, view.phase, view.name, view.role_ko),
            "생존자: " + ", ".join(view.alive),
            "사망자: " + dead,
        ]
        if view.partner_proposal:
            parts.append("[동료 마피아의 제안] 대상=%s, 이유=%s" % (
                view.partner_proposal.get("target"), view.partner_proposal.get("reason")))
        if view.my_notes:
            parts.append("[내 지난 속마음]\n" + "\n".join(
                "D%s %s" % (n.get("day"), n.get("thought", "")) for n in view.my_notes[-2:]))
        parts.append("[추론 재료(코드가 계산한 사실)]\n" + reasoning_material(view))
        parts.append("[공개 기록]\n" + (self._log_without_today_speech(view) or "아직 없음"))
        return "\n".join(parts)

    def _system_prompt(self, view):
        s = "당신은 마피아 게임 참가자 %s, 직업은 %s입니다. 한국어로만 답하세요.\n" % (view.name, view.role_ko)
        if view.role == "mafia":
            s += "정체를 절대 들키면 안 됩니다. 공개 발언에서 마피아라고 밝히지 마세요. 동료 마피아를 해치는 행동은 금지입니다.\n"
        if view.role == "police":
            s += "조사로 알게 된 사실은 확실한 정보입니다. 이를 어기는 투표나 의심은 금지입니다.\n"
        return s

    # ----- 4-4. 밤 행동 / 투표 : 에이전트 + 도구 1개 -----
    def _pick(self, text, candidates):
        """LLM 답을 후보 이름으로 바꾼다. '자기 자신', '나', '본인' 같은 표현은 내 이름으로 처리한다."""
        picked = mafia_core.normalize_choice(text, candidates)
        if picked is None and self.name in candidates:
            t = re.sub(r"[\s\.\(\)'\"]", "", str(text or ""))
            if t in ("자기자신", "자신", "나", "나자신", "본인", "저", "자기", "셀프", "self", "me", "myself"):
                picked = self.name
        return picked

    def _make_action_tool(self, action, holder, candidates):
        tool_name = ACTION_INFO[action][0]

        def _run(target: str, reason: str = "") -> str:
            picked = self._pick(target, candidates)
            if picked is None:
                return "잘못된 대상입니다. 후보: " + ", ".join(candidates)
            holder["target"], holder["reason"] = picked, reason
            return "%s 선택 완료. 한 줄로 마무리하세요." % picked

        return tool(tool_name, description=TOOL_DESC[tool_name] + " target 은 후보 이름 중 하나, reason 은 한 문장 이유.")(_run)

    def _decide(self, view, action, warning=""):
        """공통 결정 로직: 에이전트(도구) -> 구조화 출력 -> None"""
        self._view = view
        cands = list(view.candidates)
        task = ACTION_INFO[action][1] + " " + ACTION_HINT[action]
        if action == "kill" and view.partner_proposal:
            task += " 동료 제안에 동의하더라도 reason 은 복사하지 말고 당신의 말로 20자 안팎으로 새로 쓰세요."
        if warning:
            task += " " + warning
        base = (self._view_text(view) + "\n\n후보: " + ", ".join(cands) + "\n"
                "[전략 요약]\n" + self.strategy[:150] + "\n\n" + task)
        holder = {}
        if self.use_tools:
            try:
                agent = create_agent(
                    self.llm,
                    tools=[self._make_action_tool(action, holder, cands)],
                    system_prompt=self._system_prompt(view),
                )
                agent.invoke({"messages": [("user", base + " 도구를 정확히 한 번 호출하세요.")]}, config=self.cfg)
            except Exception:
                holder = {}
            if holder.get("target") in cands:
                self.last_source = "agent"
                return holder["target"], holder.get("reason", "")
        # 폴백: 구조화 출력
        try:
            out = self.llm.with_structured_output(TargetChoice).invoke(
                [("system", self._system_prompt(view)),
                 ("user", base + " target 과 reason 을 답하세요.")],
                config=self.cfg)
            picked = self._pick(out.target, cands)
            if picked:
                self.last_source = "structured"
                return picked, out.reason
        except Exception:
            pass
        self.last_source = "none"
        return None

    def night_action(self, view):
        action = {"mafia": "kill", "police": "investigate", "doctor": "protect"}.get(view.role)
        if action is None:
            return None
        res = self._decide(view, action)
        if res is None:
            return None
        target, reason = res
        # 두 번째 마피아가 동료 이유를 그대로 복사하면 짧은 자기 말로 바꾼다
        pp = view.partner_proposal
        if action == "kill" and pp and reason.strip() == str(pp.get("reason", "")).strip():
            if target == pp.get("target"):
                reason = "동료 의견에 동의하고, 내 판단으로도 %s님이 가장 위험합니다." % target
            else:
                reason = "%s님이 더 위험하다고 판단했습니다." % target
        return NightDecision(target=target, reason=reason)

    # ----- 4-5. 투표 + 자기 일관성 검사 -----
    def _violation(self, view, target):
        """마피아가 동료에게, 경찰이 확인된 비마피아에게 투표하면 위반."""
        partners, cleared, _ = private_facts(view)
        if view.role == "mafia" and target in partners:
            return "경고: %s님은 당신의 동료 마피아입니다. 동료에게 투표하면 안 됩니다. 다른 사람을 고르세요." % target
        if view.role == "police" and target in cleared:
            return "경고: %s님은 당신이 조사해 마피아가 아님을 확인한 사람입니다. 다른 사람을 고르세요." % target
        return ""

    def vote(self, view):
        action = "revote" if view.phase == "revote" else "vote"
        res = self._decide(view, action)
        if res is None:
            return None
        target, reason = res
        warn = self._violation(view, target)
        if warn:
            CONSISTENCY["retried"] += 1
            # 재질문 때는 확정 정보와 모순되는 사람(마피아의 동료, 경찰이 확인한 시민)을 후보에서 뺀다.
            partners, cleared, _ = private_facts(view)
            banned = set(partners) if view.role == "mafia" else set(cleared)
            safe_cands = [c for c in view.candidates if c not in banned]
            if safe_cands:
                view = dataclasses.replace(view, candidates=safe_cands)
            res2 = self._decide(view, action, warning=warn)
            if res2 is not None:
                target, reason = res2
            if self._violation(view, target):
                CONSISTENCY["violations"] += 1
                self.consistency_warnings += 1
        return VoteDecision(target=target, reason=reason)

    # ----- 4-6. 발언 : 구조화 출력 한 번 -----
    def _speak_prompt(self, view):
        others = [n for n in view.alive if n != self.name]
        partners, cleared, found = private_facts(view)
        if view.day <= 1 and len(view.dead) <= 1:
            day_guide = ("첫날 가이드: 정보가 적으니 성향대로 첫 의심이나 질문을 던지세요. 앞 발언자가 있으면 그 1명의 실제 말에 이름을 부르며 반응하고,"
                         " 없으면 특정인에게 질문하세요. 사망 사실을 되풀이하지 마세요(모두 압니다).")
        else:
            day_guide = "투표표와 처형된 시민에게 투표한 사람 같은 추론 재료를 근거로 구체적으로 말하세요."
        role_guide = ""
        if view.role == "police":
            if found:
                role_guide = " goal 은 정보공개로 하고, 마피아로 확정된 %s 의 조사 결과를 공개하고 투표로 몰아가세요." % ", ".join(found)
            elif cleared and view.day >= 2:
                role_guide = " 확인된 시민 %s 을(를) 변호하거나, 필요하면 경찰임을 밝히고 조사 결과를 공개하는 것을 고려하세요." % ", ".join(cleared)
        if view.role == "mafia":
            role_guide = " 동료 %s 은(는) 의심하지 말고 시민 중 한 명을 그럴듯한 근거로 지목하세요. 마피아임을 드러내면 안 됩니다." % (
                ", ".join(partners) or "마피아")
        return ("\n\n[내 역할 전략]\n" + self.strategy[:150] + "\n\n"
                "[발언 규칙]\n"
                "1) 목표(goal)를 하나 고르세요: 지목(특정인 1명 이름+구체 근거) / 변호(누군가 옹호) / 질문(특정인에게) / 정보공개.\n"
                "2) 이름 없는 '누가 숨기는 것 같다' 같은 말 금지. 다른 사람 이름을 반드시 넣으세요.\n"
                "3) 오늘 이미 나온 발언과 같은 문장/틀을 반복하지 말고 새 근거를 1개 이상 말하세요.\n"
                "4) " + day_guide + role_guide + "\n"
                "5) 다른 사람이 한 말/행동은 [오늘 이미 나온 발언]과 [공개 기록]에 실제로 있는 것만 인용하세요. 없는 일(누가 누구를 의심했다 등)을 지어내면 안 됩니다.\n"
                "6) 2~3문장, %d자 이내. suspicion 에는 %s 각각의 의심도(0~100)를 적으세요.\n"
                "7) 당신은 %s입니다. 자기 자신을 이름(%s)으로 부르거나 의심하지 마세요. '나'라고 말하세요.\n"
                "8) 지목/의심 대상은 살아있는 사람(%s) 중에서만 고르세요. 죽은 사람은 지목 대상이 아닙니다."
                % (config.SPEECH_MAX_CHARS, ", ".join(others), self.name, self.name, ", ".join(others)))

    def speak(self, view):
        self._view = view
        others = [n for n in view.alive if n != self.name]
        sys_msg = (self._system_prompt(view) +
                   "[내 캐릭터 설정]\n" + self.persona + "\n"
                   "말투와 성격을 캐릭터답게 유지하세요.")
        speeches, _ = parse_log(view)
        today = [s for s in speeches if s["day"] == view.day]
        today_text = "\n".join("%s: %s" % (s["who"], s["text"]) for s in today) or "아직 없음(내가 첫 발언자)"
        user_msg = (self._view_text(view) + "\n\n[오늘 이미 나온 발언]\n" + today_text + self._speak_prompt(view))
        best, best_pen = None, None
        for attempt in range(2):
            try:
                out = self.speak_llm.with_structured_output(SpeakOut).invoke(
                    [("system", sys_msg), ("user", user_msg)], config=self.cfg)
            except Exception:
                continue
            text = (out.speech or "").strip()
            if not text or mafia_core.contains_role_leak(text, view.role):
                user_msg += " (이전 답변은 정체를 노출했거나 비어 있었습니다. 다시 작성하세요.)"
                continue
            # 오늘 나온 남의 발언 + 내가 지난날 한 발언과 비교 (날을 넘겨 같은 말 반복 방지)
            mine_before = [s for s in speeches if s["who"] == self.name and s["day"] < view.day]
            sim = max([bigram_sim(text, s["text"]) for s in today + mine_before] or [0.0])
            has_name = any(n in text for n in others)
            self_ref = self.name in text   # 자기 이름을 3인칭으로 부르는 실수
            # 첫 토론인데 '어제/지난번' 일을 지어내는 환각
            fake_past = (not any(s["day"] < view.day for s in speeches)) and any(
                w in text for w in ("어제", "지난번", "지난 토론", "전날", "앞선 날", "이전 날", "지난날", "지난 날", "앞선 투표", "지난 투표"))
            # 문제가 적을수록 점수가 낮다. 두 번 시도 중 가장 나은 발언을 쓴다.
            pen = sim + (0 if has_name else 0.5) + (0.5 if self_ref else 0) + (0.5 if fake_past else 0)
            if best is None or pen < best_pen:
                best, best_pen = out, pen
            if has_name and sim < 0.6 and not self_ref and not fake_past:
                break
            CONSISTENCY["repeat_retries"] += 1
            if fake_past:
                why = "오늘이 첫 토론인데 없던 '어제' 일을 지어냈습니다. [오늘 이미 나온 발언]에 있는 내용만 인용하세요"
            elif self_ref:
                why = "자기 이름(%s)을 3인칭으로 불렀습니다. 자신은 '나'라고 하세요" % self.name
            elif not has_name:
                why = "살아있는 다른 사람 이름이 없었습니다"
            else:
                why = "앞 발언과 너무 비슷했습니다"
            user_msg += " (이전 답변은 %s. 앞 발언과 다른 새 근거로, 다른 사람 이름을 넣어 다시 작성하세요.)" % why
        if best is None:
            return Speech(text="아직은 판단하기 어렵네요. 조금 더 지켜보겠습니다.", thought="(발언 생성 실패)")
        sus = {}
        for it in best.suspicion:
            if it.name in others:
                sus[it.name] = max(0, min(100, int(it.score)))
        text = best.speech.strip()
        if len(text) > config.SPEECH_MAX_CHARS:
            # 단어 중간에서 끊기지 않도록 마지막 문장 끝(. ? !)까지만 남긴다
            cut = text[:config.SPEECH_MAX_CHARS]
            ends = [cut.rfind(t) + 1 for t in (".", "?", "!") if cut.rfind(t) != -1]
            end = max(ends) if ends else 0
            text = cut[:end] if end >= config.SPEECH_MAX_CHARS // 2 else cut.rstrip()
        return Speech(text=text, thought=best.thought, suspicion=sus)


# =====================================
# 5. 사회자 (Moderator)
# =====================================
class Moderator:
    def __init__(self, vs, llm=None):
        self.vs = vs
        self.llm = llm or make_llm(0.2)
        self.counter = CallCounter()

    def explain_rule(self, question):
        """룰북 RAG(type=rule 필터) + LLM. 답 끝에 출처 페이지를 붙인다."""
        docs = rag.search_rules(self.vs, question, k=3)
        ctx = "\n\n".join("(룰북 p.%d) %s" % (d.metadata["page"] + 1, d.page_content) for d in docs)
        msg = [("system", "당신은 마피아 게임 사회자입니다. 아래 룰북 발췌만 근거로 한국어로 간결히(3문장 이내) 답하세요. "
                          "룰북에 없으면 모른다고 하세요."),
               ("user", "[룰북 발췌]\n" + ctx + "\n\n[질문] " + question)]
        answer = self.llm.invoke(msg, config={"callbacks": [self.counter]}).content.strip()
        pages = sorted({d.metadata["page"] + 1 for d in docs})
        return answer + "\n" + " ".join("(룰북 p.%d)" % p for p in pages)

    def narrate(self, event):
        """이벤트를 사회자 멘트로 바꾼다. 템플릿 기반이라 LLM 을 쓰지 않는다."""
        t = event.get("type", "")
        actor, target, text = event.get("actor"), event.get("target"), event.get("text", "")
        day = event.get("day", "")
        if t == "night_start":
            return "밤이 되었습니다. 모두 눈을 감아 주세요."
        if t == "morning":
            return "%s일차 아침이 밝았습니다. %s" % (day, text)
        if t == "death":
            return "%s님이 사망했습니다. %s" % (target, text)
        if t == "no_death":
            return "간밤에는 아무도 죽지 않았습니다."
        if t == "speech":
            return "%s: %s" % (actor, text)
        if t == "vote":
            return "%s님이 %s님에게 투표했습니다." % (actor, target)
        if t == "execution":
            return "투표 결과 %s님이 처형되었습니다. %s" % (target, text)
        if t == "no_execution":
            return "동점으로 아무도 처형되지 않았습니다."
        if t == "end":
            return "게임이 끝났습니다. " + text
        return text or "[%s]" % t


# =====================================
# 6. 컨트롤러 생성
# =====================================
def build_controllers(state, vs, human_ask_rule=True):
    """플레이어 이름 -> Controller. AI 는 LLMPlayer, 사람은 HumanController."""
    act_llm = make_llm(ACT_TEMPERATURE)       # 판단용(밤 행동/투표)
    speak_llm = make_llm(SPEAK_TEMPERATURE)   # 발언용
    mod = Moderator(vs) if human_ask_rule else None
    ctrls = {}
    for p in state.players:
        if p.is_human:
            ctrls[p.name] = mafia_core.HumanController(
                p.name, ask_rule=(mod.explain_rule if mod else None))
        else:
            ctrls[p.name] = LLMPlayer(p.name, p.role, vs, llm=act_llm, state=state, speak_llm=speak_llm)
    return ctrls
