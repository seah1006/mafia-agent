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
# 공개 발언 속 "나는 경찰" 같은 경찰 주장
CLAIM_RE = re.compile(r"(나는|난|저는|전|내가|제가|나|저)\s*(진짜\s*)?경찰|경찰\s*(입니다|이에요|이예요|이야|이다|임|맞)")
# 밤 행동은 아무도 볼 수 없는데 "밤에 움직였다"처럼 지어내는 환각 문구
# 첫 토론인데 '어제' 일을 지어내는 표현 -> 같은 날 일로 고친다(재생성보다 빠르고 확실)
FAKE_PAST_FIX = (("지난 토론", "아까 토론"), ("지난번", "아까"), ("지난날", "아까"), ("지난 날", "아까"), ("앞선 날", "아까"),
                 ("이전 날", "아까"), ("앞선 투표", "아까"), ("지난 투표", "아까"), ("전날", "아까"), ("어제", "아까"))
NIGHT_CLAIM_WORDS = ("밤에 움직", "밤에 뭐", "밤에 무엇", "밤에 어디", "밤에 행동", "밤에 나갔", "밤사이 움직", "밤새 움직")


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
        # 시민팀(시민/경찰/의사)이 처형됐다면, 그 사람에게 표를 던진 사람이 수상하다
        if d.get("cause") == "처형" and d["role_ko"] != "마피아":
            who = sorted({v["voter"] for v in votes if v["target"] == d["name"]})
            if who:
                lines.append("처형된 %s %s 에게 투표한 사람: %s" % (d["role_ko"], d["name"], ", ".join(who)))
    # 공개된 경찰 주장 목록 (사망하면 직업이 공개되므로 진위도 코드가 알려준다)
    claims = {}
    for s in speeches:
        if CLAIM_RE.search(s["text"]) and s["who"] not in claims:
            claims[s["who"]] = s
    dead_role = {d["name"]: d["role_ko"] for d in view.dead}
    for who, s in claims.items():
        r = dead_role.get(who)
        status = ("진짜 경찰로 확인됨(사망 후 직업 공개)" if r == "경찰" else
                  "거짓 주장으로 확인됨(사망 후 %s로 공개)" % r if r else "아직 진위 미확인")
        lines.append("경찰이라고 주장한 사람: %s (%d일차, %s) \"%s\"" % (who, s["day"], status, s["text"][:70]))
    if len(claims) >= 2:
        lines.append("경찰 주장자가 %d명입니다. 진짜 경찰은 1명뿐이니 나머지는 거짓말(마피아 가능성 높음)입니다." % len(claims))
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
    "protect": "마피아가 노릴 것 같은 사람(경찰로 보이는 사람, 논리적으로 추리해 주목받는 사람, 또는 자기 자신)을 보호하세요. 의심스러운 사람을 보호하는 것이 아닙니다. 공개적으로 경찰이라고 밝힌 사람이 있으면 그 사람을 우선 보호하세요.",
    "investigate": "아직 조사하지 않았고 발언이나 투표가 수상한 사람을 고르세요.",
    "vote": "발언과 투표표를 근거로 마피아일 가능성이 가장 높은 사람에게 투표하세요. 공개된 경찰 주장과 조사 결과가 있으면 반드시 고려하세요.",
    "revote": "발언과 투표표를 근거로 마피아일 가능성이 가장 높은 사람에게 투표하세요. 공개된 경찰 주장과 조사 결과가 있으면 반드시 고려하세요.",
}
ACTION_KO = {"kill": "밤 처치", "protect": "밤 보호", "investigate": "밤 조사",
             "vote": "투표", "revote": "재투표", "speak": "발언"}
SOURCE_KO = {"agent": "에이전트가 도구를 호출해 결정", "structured": "도구 호출 실패 -> 구조화 출력(Pydantic)으로 결정",
             "none": "결정 실패 -> 코어가 재시도 후 무작위 대체"}


def speakers_mentioning(today, name):
    """오늘 발언 중 name 을 언급한 사람(본인 제외) 목록."""
    return sorted({s["who"] for s in today if s["who"] != name and name in s["text"]})


def pile_on_target(today, alive):
    """오늘 3명 이상이 같은 사람을 언급했으면 (그 이름, 인원), 아니면 (None, 0)."""
    best, cnt = None, 0
    for n in alive:
        c = len(speakers_mentioning(today, n))
        if c > cnt:
            best, cnt = n, c
    return (best, cnt) if cnt >= 3 else (None, 0)


def replace_self_name(text, name):
    """자기 이름 3인칭(예: '윤서진은')을 1인칭('나는')으로 바꾼다."""
    rep = {"은": "나는", "는": "나는", "이": "내가", "가": "내가", "을": "나를", "를": "나를", "의": "나의"}
    return re.sub(r"(?:(?:저|나)\s+)?" + re.escape(name) + r"(?:님|씨)?(은|는|이|가|을|를|의)?",
                  lambda m: rep.get(m.group(1) or "", "나"), text)


def fix_self_reference(text, name):
    """자기 이름이 들어간 문장을 지운다(남는 문장이 없거나 경찰 주장 문장이면 1인칭으로 바꾼다)."""
    sents = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]
    kept = [s if name not in s else (replace_self_name(s, name) if "경찰" in s else None) for s in sents]
    kept = [s for s in kept if s]
    if kept:
        return " ".join(kept)
    return replace_self_name(text, name)


class LLMPlayer:
    """mafia_core.Controller 구현체. 역할별 도구를 가진 LLM 플레이어."""

    _strategy_cache = {}   # 역할 -> 전략 텍스트 (플레이어 간 공유, 호출 절약)
    _persona_cache = {}    # 이름 -> 페르소나 텍스트
    _strategy_pages = {}   # 역할 -> [(룰북 쪽수, 제목)]  (사고 과정 보기에서 RAG 출처 표시용)

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
        self.trace_out = None      # 사고 과정 출력 함수(print 등). None 이면 출력하지 않는다(관전 모드 전용)
        self._steps = []           # 이번 결정에서 일어난 일(도구 호출, 안전장치 작동 등)
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
            pages = []
            try:
                docs = rag.search_strategy(self.vs, self.role, "토론 발언과 투표 전략", k=2)
                text = "\n".join(d.page_content for d in docs)
                for d in docs:
                    p = (d.metadata.get("page", -1) + 1, d.metadata.get("title", ""))
                    if p not in pages:
                        pages.append(p)
            except Exception:
                text = ""
            LLMPlayer._strategy_cache[self.role] = text[:600]
            LLMPlayer._strategy_pages[self.role] = pages
        return LLMPlayer._strategy_cache[self.role]

    # ----- 4-1b. 사고 과정 보기 (관전 모드 전용) -----
    def _rag_sources(self, with_persona=False):
        """이번 판단에 들어간 룰북 페이지(메타데이터 필터 검색 결과)."""
        out = []
        pages = LLMPlayer._strategy_pages.get(self.role) or []
        if pages:
            out.append("%s (role=%s 필터)" % (", ".join("p.%d %s" % (p, t) for p, t in pages), self.role))
        if with_persona:
            for idx, meta in config.PAGE_META.items():
                if meta.get("name") == self.name:
                    out.append("p.%d %s (name 필터)" % (idx + 1, meta["title"]))
        return " / ".join(out) or "없음"

    def _emit_trace(self, view, action, result_line, extra=None, compact=False):
        """결정 하나의 사고 과정을 상자 모양으로 출력한다. 출력 실패는 게임에 영향을 주지 않는다."""
        if not self.trace_out:
            return
        try:
            head = "    ┌─ 사고 과정 · %s %s · %s" % (view.role_ko, self.name, ACTION_KO.get(action, action))
            body = []
            if compact:
                body += self._steps
            else:
                body.append("받은 정보(PlayerView): 생존 %d명 · 공개 기록 %d줄 · 지난 속마음 %d개 · 후보 %d명" % (
                    len(view.alive), len((view.public_log_text or "").splitlines()), len(view.my_notes), len(view.candidates)))
                body.append("나만 아는 정보: " + (" / ".join(view.private_info) or "없음"))
                if view.partner_proposal:
                    body.append("동료 마피아의 제안: %s" % view.partner_proposal.get("target"))
                body.append("RAG 검색: " + self._rag_sources(with_persona=(action == "speak")))
                body += self._steps
            body += list(extra or [])
            lines = [head] + ["    │ " + b for b in body] + ["    └─ " + result_line]
            self.trace_out("\n".join(lines))
        except Exception:
            pass

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
        if view.phase == "night" and not (view.public_log_text or "").count("일차 낮]"):
            # 첫날 밤에는 아무도 말한 적이 없다. 이유에 없는 발언/행동을 지어내지 않게 한다.
            task += " 아직 아무도 발언하거나 투표한 적이 없습니다. reason 에 있지도 않은 발언이나 행동을 지어내지 말고, 캐릭터 성향이나 전략을 근거로 쓰세요."
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
                res = agent.invoke({"messages": [("user", base + " 도구를 정확히 한 번 호출하세요.")]}, config=self.cfg)
                self._record_tool_calls(res)
            except Exception:
                holder = {}
                self._steps.append("에이전트 호출 오류 -> 구조화 출력으로 다시 질문")
            if holder.get("target") in cands:
                self.last_source = "agent"
                return holder["target"], holder.get("reason", "")
            if not any("오류" in s for s in self._steps):
                self._steps.append("유효한 도구 호출 없음 -> 구조화 출력(Pydantic)으로 다시 질문")
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

    def _record_tool_calls(self, res):
        """에이전트 실행 결과 메시지에서 도구 호출과 도구 응답을 뽑아 사고 과정에 남린다."""
        for m in (res or {}).get("messages", []):
            for tc in getattr(m, "tool_calls", None) or []:
                args = tc.get("args") or {}
                self._steps.append("도구 호출: %s(target=%r)" % (tc.get("name"), args.get("target")))
            if getattr(m, "type", "") == "tool":
                self._steps.append("  └ 도구 응답: " + str(m.content)[:60])

    def night_action(self, view):
        action = {"mafia": "kill", "police": "investigate", "doctor": "protect"}.get(view.role)
        if action is None:
            return None
        self._steps = []
        res = self._decide(view, action)
        if res is None:
            self._emit_trace(view, action, "결정: 없음 (" + SOURCE_KO["none"] + ")")
            return None
        target, reason = res
        self._emit_trace(view, action, "결정: %s (%s)" % (target, SOURCE_KO.get(self.last_source, self.last_source)),
                         extra=["이유: " + reason[:80]] if reason else None)
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
        self._steps = []
        res = self._decide(view, action)
        if res is None:
            self._emit_trace(view, action, "결정: 없음 (" + SOURCE_KO["none"] + ")", compact=True)
            return None
        target, reason = res
        warn = self._violation(view, target)
        if warn:
            self._steps.append("일관성 검사 작동: %s 투표는 확정 정보와 모순 -> 후보에서 빼고 재질문" % target)
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
        self._emit_trace(view, action, "투표: %s (%s)" % (target, SOURCE_KO.get(self.last_source, self.last_source)),
                         extra=["근거: " + reason[:80]] if reason else None, compact=True)
        return VoteDecision(target=target, reason=reason)

    # ----- 4-6. 발언 : 구조화 출력 한 번 -----
    def _police_should_claim(self, view, today):
        """경찰이 지금 정체와 조사 결과를 공개해야 하는가.
        마피아를 찾았거나, 확인된 시민이 있는데 2일차 이후이거나, 오늘 2명 이상에게 의심받고 있으면 공개한다."""
        if view.role != "police":
            return False
        _p, cleared, found = private_facts(view)
        under_fire = len(speakers_mentioning(today, self.name)) >= 2
        return bool(found) or (bool(cleared) and (view.day >= 2 or under_fire))

    def _speak_prompt(self, view, today, claim_now):
        others = [n for n in view.alive if n != self.name]
        partners, cleared, found = private_facts(view)
        if view.day <= 1 and len(view.dead) <= 1:
            day_guide = ("첫날 가이드: 정보가 적으니 성향대로 첫 의심이나 질문을 던지세요. 앞 발언자가 있으면 그 1명의 실제 말에 이름을 부르며 반응하고,"
                         " 없으면 특정인에게 질문하세요. 사망 사실을 되풀이하지 마세요(모두 압니다).")
        else:
            day_guide = "투표표와 처형된 사람에게 투표한 사람, 경찰 주장 같은 추론 재료를 근거로 구체적으로 말하세요."
        role_guide = ""
        if view.role == "police" and claim_now:
            results = ["%s는 마피아" % n for n in found] + ["%s는 마피아가 아님" % n for n in cleared]
            role_guide = (" 지금은 정체를 밝힐 때입니다. goal 은 정보공개로 하고, '나는 경찰이다'라고 말한 뒤 조사 결과(%s)를 공개하세요."
                          % ", ".join(results))
            if found:
                role_guide += " 그리고 %s 에게 투표하자고 하세요." % ", ".join(found)
        if view.role == "mafia":
            role_guide = " 동료 %s 은(는) 의심하지 말고 시민 중 한 명을 그럴듯한 근거로 지목하세요. 마피아임을 드러내면 안 됩니다." % (
                ", ".join(partners) or "마피아")
        # 몰아가기 방지: 오늘 3명 이상이 이미 같은 사람을 언급했으면 같은 말 반복 대신 새 관점을 요구
        pile, cnt = pile_on_target(today, view.alive)
        pile_guide = ""
        if pile == self.name:
            pile_guide = "\n9) 오늘 %d명이 나를 의심했습니다. 감정적으로 반박하지 말고 내 결백 근거를 말하거나, 나를 몰아가는 사람 중 한 명을 지목하세요." % cnt
        elif pile:
            pile_guide = ("\n9) 오늘 이미 %d명이 %s 을(를) 언급했습니다. 같은 질문/의심을 되풀이하지 마세요. 동의한다면 아직 아무도 말하지 않은 새 근거를, "
                          "아니라면 다른 사람을 지목하세요." % (cnt, pile))
        return ("\n\n[내 역할 전략]\n" + self.strategy[:150] + "\n\n"
                "[발언 규칙]\n"
                "1) 목표(goal)를 하나 고르세요: 지목(특정인 1명 이름+구체 근거) / 변호(누군가 옹호) / 질문(특정인에게) / 정보공개.\n"
                "2) 이름 없는 '누가 숨기는 것 같다' 같은 말 금지. 다른 사람 이름을 반드시 넣으세요.\n"
                "3) 오늘 이미 나온 발언과 같은 문장/틀을 반복하지 말고 새 근거를 1개 이상 말하세요.\n"
                "4) " + day_guide + role_guide + "\n"
                "5) 다른 사람이 한 말/행동은 [오늘 이미 나온 발언]과 [공개 기록]에 실제로 있는 것만 인용하세요. 없는 일(누가 누구를 의심했다 등)을 지어내면 안 됩니다.\n"
                "6) 2~3문장, %d자 이내. suspicion 에는 %s 각각의 의심도(0~100)를 적으세요.\n"
                "7) 당신은 %s입니다. 자기 자신을 이름(%s)으로 부르거나 의심하지 마세요. '나'라고 말하세요.\n"
                "8) 지목/의심 대상은 살아있는 사람(%s) 중에서만 고르세요. 죽은 사람은 지목 대상이 아닙니다. "
                "밤에 누가 무엇을 했는지는 아무도 볼 수 없으니 '밤에 움직였다' 같은 말은 근거가 될 수 없습니다.\n"
                "thought 는 speech 와 다른 문장으로, 남에게 말하지 않는 진짜 판단(마피아라면 실제 의도와 위장 계획)을 쓰세요."
                % (config.SPEECH_MAX_CHARS, ", ".join(others), self.name, self.name, ", ".join(others)) + pile_guide)

    def speak(self, view):
        self._view = view
        self._steps = []
        others = [n for n in view.alive if n != self.name]
        sys_msg = (self._system_prompt(view) +
                   "[내 캐릭터 설정]\n" + self.persona + "\n"
                   "말투와 성격을 캐릭터답게 유지하세요.")
        speeches, _ = parse_log(view)
        today = [s for s in speeches if s["day"] == view.day]
        claim_now = self._police_should_claim(view, today)
        today_text = "\n".join("%s: %s" % (s["who"], s["text"]) for s in today) or "아직 없음(내가 첫 발언자)"
        user_msg = (self._view_text(view) + "\n\n[오늘 이미 나온 발언]\n" + today_text + self._speak_prompt(view, today, claim_now))
        if claim_now:
            self._steps.append("전략 판단(코드): 경찰 정체와 조사 결과를 공개할 시점 -> 정보공개 지시")
        pile, cnt = pile_on_target(today, view.alive)
        if pile:
            self._steps.append("몰아가기 감지(코드): 오늘 %d명이 %s 언급 -> 같은 말 반복 금지 지시" % (cnt, pile))
        best, best_pen = None, None
        for attempt in range(2):
            try:
                out = self.speak_llm.with_structured_output(SpeakOut).invoke(
                    [("system", sys_msg), ("user", user_msg)], config=self.cfg)
            except Exception:
                self._steps.append("%d차 생성: 형식 오류 -> 재생성" % (attempt + 1))
                continue
            text = (out.speech or "").strip()
            if not text or mafia_core.contains_role_leak(text, view.role):
                self._steps.append("%d차 생성: 정체 노출/빈 답변 감지 -> 재생성" % (attempt + 1))
                user_msg += " (이전 답변은 정체를 노출했거나 비어 있었습니다. 다시 작성하세요.)"
                continue
            # 오늘 나온 남의 발언 + 내가 지난날 한 발언과 비교 (날을 넘겨 같은 말 반복 방지)
            mine_before = [s for s in speeches if s["who"] == self.name and s["day"] < view.day]
            sim = max([bigram_sim(text, s["text"]) for s in today + mine_before] or [0.0])
            has_name = any(n in text for n in others)
            self_ref = self.name in text   # 자기 이름을 3인칭으로 부르는 실수
            # 첫 토론인데 '어제/지난번' 일을 지어내는 환각
            # (재생성하지 않고 아래 후처리에서 '아까'로 고친다)
            first_talk = not any(s["day"] < view.day for s in speeches)
            fake_past = first_talk and any(w in text for w, _r in FAKE_PAST_FIX)
            # 아무도 볼 수 없는 밤 행동을 근거로 대는 환각
            night_claim = any(w in text for w in NIGHT_CLAIM_WORDS)
            # 경찰이 공개해야 할 때 공개하지 않음
            missing_claim = claim_now and not CLAIM_RE.search(text)
            # 문제가 적을수록 점수가 낮다. 두 번 시도 중 가장 나은 발언을 쓴다.
            pen = (sim + (0 if has_name else 0.5) + (0.5 if self_ref else 0) + (0.1 if fake_past else 0)
                   + (0.5 if night_claim else 0) + (0.5 if missing_claim else 0))
            if best is None or pen < best_pen:
                best, best_pen = out, pen
            if has_name and sim < 0.6 and not self_ref and not night_claim and not missing_claim:
                break
            CONSISTENCY["repeat_retries"] += 1
            if night_claim:
                why, short = "아무도 볼 수 없는 밤 행동('밤에 움직였다' 등)을 근거로 댔습니다. 발언과 투표만 근거로 쓰세요", "볼 수 없는 밤 행동을 근거로 씀"
            elif self_ref:
                why, short = "자기 이름(%s)을 3인칭으로 불렀습니다. 자신은 '나'라고 하세요" % self.name, "자기 이름 3인칭"
            elif missing_claim:
                why, short = "경찰임을 밝히고 조사 결과를 공개하라는 지시를 따르지 않았습니다. '나는 경찰이다'로 시작하세요", "경찰 정보공개 누락"
            elif not has_name:
                why, short = "살아있는 다른 사람 이름이 없었습니다", "다른 사람 이름 없음"
            else:
                why, short = "앞 발언과 너무 비슷했습니다", "앞 발언과 유사도 %.2f" % sim
            self._steps.append("%d차 생성: 안전장치 작동(%s) -> 재생성" % (attempt + 1, short))
            user_msg += " (이전 답변은 %s. 앞 발언과 다른 새 근거로, 다른 사람 이름을 넣어 다시 작성하세요.)" % why
        if best is None:
            self._emit_trace(view, "speak", "발언 생성 실패 -> 중립 문장")
            return Speech(text="아직은 판단하기 어렵네요. 조금 더 지켜보겠습니다.", thought="(발언 생성 실패)")
        sus = {}
        for it in best.suspicion:
            if it.name in others:
                sus[it.name] = max(0, min(100, int(it.score)))
        text = best.speech.strip()
        if not any(s["day"] < view.day for s in speeches) and any(w in text for w, _r in FAKE_PAST_FIX):
            for w, r in FAKE_PAST_FIX:
                text = text.replace(w, r)
            self._steps.append("후처리(코드): 첫 토론인데 '어제' 표현 -> 같은 날 '아까'로 교정")
        if self.name in text:
            text = fix_self_reference(text, self.name)
            self._steps.append("후처리(코드): 자기 이름 3인칭 문장 정리")
        if len(text) > config.SPEECH_MAX_CHARS:
            # 단어 중간에서 끊기지 않도록 마지막 문장 끝(. ? !)까지만 남긴다
            cut = text[:config.SPEECH_MAX_CHARS]
            ends = [cut.rfind(t) + 1 for t in (".", "?", "!") if cut.rfind(t) != -1]
            end = max(ends) if ends else 0
            text = cut[:end] if end >= config.SPEECH_MAX_CHARS // 2 else cut.rstrip()
        thought = (best.thought or "").strip()
        if not thought or bigram_sim(thought, best.speech or "") >= 0.6:
            # 속마음이 공개 발언을 그대로 베꼈으면 의심도와 목표로 코드가 요약한다(없는 생각을 지어내지 않음)
            top = sorted(sus.items(), key=lambda kv: -kv[1])[:2]
            thought = "[코드 요약] 의심 순위: %s / 발언 목표: %s" % (
                ", ".join("%s %d" % kv for kv in top) or "없음", best.goal)
            self._steps.append("후처리(코드): 속마음이 발언과 같아서 의심도 요약으로 대체")
        top3 = sorted(sus.items(), key=lambda kv: -kv[1])[:3]
        self._emit_trace(view, "speak", "발언 목표: %s" % best.goal, extra=[
            "속마음(비공개): " + thought[:90],
            "의심도(비공개): " + (", ".join("%s %d" % kv for kv in top3) or "없음"),
        ])
        return Speech(text=text, thought=thought, suspicion=sus)


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
def build_controllers(state, vs, human_ask_rule=True, show_thinking=False, printer=print):
    """플레이어 이름 -> Controller. AI 는 LLMPlayer, 사람은 HumanController.

    show_thinking=True : 관전 모드에서 AI 의 사고 과정(받은 정보 -> RAG -> 도구 호출 -> 안전장치 -> 결정)을 출력한다.
    사람이 참여하는 판에서는 비밀 정보가 보이면 안 되므로 자동으로 땈다(정보 분리 원칙).
    """
    if show_thinking and any(p.is_human for p in state.players):
        printer("[알림] 사람이 참여하는 판에서는 사고 과정 보기를 끕니다(다른 사람의 비밀이 보이기 때문).")
        show_thinking = False
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
            if show_thinking:
                ctrls[p.name].trace_out = printer
    return ctrls
