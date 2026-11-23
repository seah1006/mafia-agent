# =====================================
# mafia_core.py : 마피아 게임 규칙 엔진 (LLM / LangChain 의존성 0)
# =====================================
# 이 모듈은 "코드가 진실을 지킨다" 는 원칙의 중심이다.
#  - 역할 배정, 밤 처리, 투표 집계, 승리 판정은 전부 이 파일의 순수 파이썬 코드가 한다.
#  - AI 에게는 build_view() 가 만든 PlayerView 만 전달된다 (정보 분리).
#  - 컨트롤러(LLM/사람/랜덤)가 이상한 값을 돌려줘도 safe_decide() 가 걸러서 게임은 멈추지 않는다.
#
# [LangGraph 노드에서 쓰는 단계별 함수 (GameEngine 메서드)]
#   engine.announce_start()                    게임 시작 안내 이벤트
#   engine.begin_night()                       day += 1, phase="night"
#   engine.collect_mafia_decision()            -> (target, info)  마피아 순차 결정
#   engine.collect_doctor_decision()           -> target | None
#   engine.collect_police_decision()           -> (police, target) | None
#   engine.apply_night(kill, protect, inv)     -> events  (아침 발표 + 승리 판정까지)
#   engine.run_night()                         위 단계를 모두 실행 -> events
#   engine.begin_day()                         phase="day"
#   engine.discussion_order()                  -> 이번 라운드 발언 순서(무작위)
#   engine.run_discussion_turn(name)           -> events (한 명 발언)
#   engine.run_day()                           토론 전체 -> events
#   engine.collect_votes(kind, tie_candidates) -> {투표자: 대상}
#   engine.apply_votes(votes, kind)            -> result dict (처형까지 처리)
#   engine.run_vote()                          투표 + 재투표 + 승리 판정 -> events
#   engine.check_end()                         승리 판정 후 end 이벤트
#   engine.run(max_days)                       전체 게임 -> winner

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import config

# =====================================
# 1. 데이터 구조
# =====================================


@dataclass
class Player:
    name: str
    role: str
    alive: bool = True
    is_human: bool = False


@dataclass
class GameState:
    players: list
    day: int = 0
    phase: str = "setup"          # setup | night | day | vote | end
    public_log: list = field(default_factory=list)      # 공개 사건/발언/투표
    private_info: dict = field(default_factory=dict)    # 이름 -> 본인만 아는 정보
    notes: dict = field(default_factory=dict)           # 이름 -> AI 속마음 기록
    night_history: list = field(default_factory=list)
    vote_history: list = field(default_factory=list)
    winner: Optional[str] = None  # "citizen" | "mafia" | "draw"
    seed: Optional[int] = None
    stats: dict = field(default_factory=lambda: {
        "retries": 0, "fallbacks": 0, "leak_blocked": 0, "llm_calls": 0})
    rng: random.Random = field(default_factory=random.Random, repr=False)


@dataclass
class PlayerView:
    """정보 분리의 핵심: AI 에게는 이 객체만 전달된다."""
    name: str
    role: str
    role_ko: str
    day: int
    phase: str
    alive: list
    dead: list                    # [{"name", "role_ko", "cause"}]
    public_log_text: str
    private_info: list
    my_notes: list
    candidates: list
    partner_proposal: Optional[dict] = None   # 두 번째 마피아에게만 {"target","reason"}


@dataclass
class NightDecision:
    target: str
    reason: str = ""


@dataclass
class Speech:
    text: str
    thought: str = ""
    suspicion: dict = field(default_factory=dict)   # 이름 -> 0~100


@dataclass
class VoteDecision:
    target: str
    reason: str = ""


class Controller(Protocol):
    name: str

    def night_action(self, view: PlayerView) -> Optional[NightDecision]: ...
    def speak(self, view: PlayerView) -> Speech: ...
    def vote(self, view: PlayerView) -> VoteDecision: ...


# =====================================
# 2. 보조 함수
# =====================================

NEUTRAL_LINES = [
    "아직 판단하기 어렵네요. 조금 더 지켜보겠습니다.",
    "지금은 정보가 부족해서 말을 아끼겠습니다.",
    "다른 분들의 발언을 더 들어보고 판단하겠습니다.",
    "섣불리 의심하기보다 투표 흐름을 보고 싶습니다.",
]


def _josa(name: str, with_batchim: str, without_batchim: str) -> str:
    """호환용: 받침 여부에 따라 조사 선택."""
    return josa(name, with_batchim + "/" + without_batchim)


def josa(word: str, kind: str) -> str:
    """단어 마지막 글자의 받침으로 조사 선택. kind: '이/가' | '은/는' | '을/를' | '과/와' | '으로/로'.

    예: josa("시민", "이/가") -> "이", josa("의사", "이/가") -> "가". 단어만 돌려주지 않고 조사만 돌려준다.
    """
    with_b, without_b = kind.split("/")
    if not word:
        return without_b
    ch = word[-1]
    if "가" <= ch <= "힣":
        final = (ord(ch) - 0xAC00) % 28
        if kind == "으로/로":
            return with_b if final not in (0, 8) else without_b   # ㄹ 받침은 '로'
        return with_b if final != 0 else without_b
    return without_b


def get_player(state: GameState, name: str) -> Optional[Player]:
    for p in state.players:
        if p.name == name:
            return p
    return None


def alive_players(state: GameState) -> list:
    return [p for p in state.players if p.alive]


def alive_names(state: GameState) -> list:
    return [p.name for p in state.players if p.alive]


def _log(state: GameState, type_: str, text: str, actor: str = "", target: str = "",
         phase: Optional[str] = None) -> dict:
    entry = {"day": state.day, "phase": phase or state.phase, "type": type_,
             "actor": actor, "target": target, "text": text}
    state.public_log.append(entry)
    return entry


# =====================================
# 3. 게임 생성
# =====================================


def new_game(ai_names: list, human_name: Optional[str] = None,
             seed: Optional[int] = None) -> GameState:
    """역할을 무작위 배정하고 마피아끼리 서로의 정체를 private_info 에 기록한다."""
    names = list(ai_names) + ([human_name] if human_name else [])
    if len(names) != config.TOTAL_PLAYERS:
        raise ValueError("참가자는 %d명이어야 합니다 (현재 %d명)" % (config.TOTAL_PLAYERS, len(names)))
    if len(set(names)) != len(names):
        raise ValueError("참가자 이름이 중복됩니다")
    rng = random.Random(seed)
    roles = []
    for role, cnt in config.ROLE_DISTRIBUTION.items():
        roles.extend([role] * cnt)
    rng.shuffle(roles)
    players = [Player(name=n, role=r, is_human=(human_name is not None and n == human_name))
               for n, r in zip(names, roles)]
    state = GameState(players=players, seed=seed, rng=rng)
    state.private_info = {n: [] for n in names}
    state.notes = {n: [] for n in names}
    mafias = [p.name for p in players if p.role == "mafia"]
    for m in mafias:
        for other in mafias:
            if other != m:
                state.private_info[m].append("동료 마피아: " + other)
    return state


# =====================================
# 4. 대상 / 정규화
# =====================================


def valid_targets(state: GameState, name: str, action: str, tie_candidates: Optional[list] = None) -> list:
    """행동별 유효 대상 목록."""
    me = get_player(state, name)
    alive = alive_names(state)
    if action == "kill":
        return [p.name for p in state.players if p.alive and p.role != "mafia"]
    if action == "protect":
        return alive
    if action in ("investigate", "speak", "vote"):
        return [n for n in alive if n != name]
    if action == "revote":
        pool = tie_candidates if tie_candidates else alive
        return [n for n in pool if n != name and n in alive]
    return [n for n in alive if me is None or n != name]


_NOISE = re.compile(r"[\s\"'`.,!?()\[\]{}<>:;~*_\-]")


def normalize_choice(text, candidates: list) -> Optional[str]:
    """'강태호님', '태호', '3번 강태호', '3' 같은 답을 후보 이름 하나로 맞춘다. 실패하면 None."""
    if text is None or not candidates:
        return None
    t = text if isinstance(text, str) else str(text)
    t = t.strip()
    if not t:
        return None
    if t in candidates:
        return t
    # 후보 이름이 문장 안에 들어있는 경우 (가장 먼저 나오는 이름)
    hits = [(t.index(c), c) for c in candidates if c in t]
    if hits:
        hits.sort(key=lambda x: (x[0], -len(x[1])))
        return hits[0][1]
    s = _NOISE.sub("", t)
    s = re.sub(r"\d+번?", "", s)
    s = re.sub(r"(님|씨)$", "", s)
    if s in candidates:
        return s
    if len(s) >= 2:
        sub = [c for c in candidates if len(c) >= 3 and c[1:] in s]
        if len(sub) == 1:
            return sub[0]
        sub = [c for c in candidates if s in c]
        if len(sub) == 1:
            return sub[0]
    m = re.fullmatch(r"\s*(\d+)\s*(번|번째)?\s*", t)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
    return None


# =====================================
# 5. 정보 분리: PlayerView
# =====================================


def _render_entry(e: dict) -> str:
    d = e.get("day", 0)
    t = e.get("type")
    if t == "speech":
        return "[%d일차 낮] %s: %s" % (d, e["actor"], e["text"])
    if t == "vote":
        return "[%d일차 투표] %s -> %s" % (d, e["actor"], e["target"])
    if t == "revote":
        return "[%d일차 재투표] %s -> %s" % (d, e["actor"], e["target"])
    if t in ("death", "no_death"):
        return "[%d일차 아침] %s" % (d, e["text"])
    if t in ("execution", "no_execution"):
        return "[%d일차 처형] %s" % (d, e["text"])
    return "[%d일차] %s" % (d, e.get("text", ""))


def _dead_list(state: GameState) -> list:
    dead = []
    for e in state.public_log:
        if e["type"] in ("death", "execution"):
            p = get_player(state, e["target"])
            if p is not None:
                dead.append({"name": p.name, "role_ko": config.ROLE_KO[p.role],
                             "cause": "밤 사망" if e["type"] == "death" else "처형"})
    return dead


def _public_log_text(state: GameState, max_chars: int) -> str:
    entries = state.public_log
    lines = [_render_entry(e) for e in entries]
    n = len(lines)
    for s in range(n + 1):
        kept = lines[s:]
        head = []
        if s > 0:
            by_day = {}
            for e in entries[:s]:
                if e["type"] in ("death", "execution"):
                    p = get_player(state, e["target"])
                    role_ko = config.ROLE_KO[p.role] if p else "?"
                    kind = "사망" if e["type"] == "death" else "처형"
                    by_day.setdefault(e["day"], []).append("%s(%s) %s" % (e["target"], role_ko, kind))
            days = sorted(set(e["day"] for e in entries[:s]))
            for d in days:
                head.append("[요약 %d일차] %s" % (d, ", ".join(by_day.get(d, ["사망/처형 없음"]))))
            head.append("(이전 발언과 투표는 생략됨)")
        text = "\n".join(head + kept)
        if len(text) <= max_chars:
            return text
    return text[-max_chars:]


def build_view(state: GameState, name: str, action: str, partner_proposal: Optional[dict] = None,
               tie_candidates: Optional[list] = None) -> PlayerView:
    """AI 에게 줄 정보를 만든다. 공개 정보 + 본인의 비공개 정보 + 본인의 메모만 들어간다."""
    me = get_player(state, name)
    if me is None:
        raise KeyError(name)
    return PlayerView(
        name=name,
        role=me.role,
        role_ko=config.ROLE_KO[me.role],
        day=state.day,
        phase=state.phase,
        alive=alive_names(state),
        dead=_dead_list(state),
        public_log_text=_public_log_text(state, config.PUBLIC_LOG_MAX_CHARS),
        private_info=list(state.private_info.get(name, [])),
        my_notes=[dict(n) for n in state.notes.get(name, [])],
        candidates=valid_targets(state, name, action, tie_candidates),
        partner_proposal=dict(partner_proposal) if (partner_proposal and me.role == "mafia") else None,
    )


# =====================================
# 6. 밤 처리 / 투표 집계 / 승리 판정
# =====================================


def resolve_night(state: GameState, kill_target: Optional[str], protect_target: Optional[str],
                  investigation: Optional[tuple] = None) -> list:
    """밤 결과를 상태에 반영하고 공개 이벤트(사망/무사망) 목록을 돌려준다.

    investigation = (경찰 이름, 조사 대상 이름). 결과는 경찰의 private_info 에만 기록된다.
    """
    events = []
    saved = bool(kill_target) and kill_target == protect_target
    killed = None
    if kill_target and not saved:
        victim = get_player(state, kill_target)
        if victim is not None and victim.alive:
            victim.alive = False
            killed = victim.name
    inv_rec = None
    if investigation:
        police_name, target_name = investigation
        tp = get_player(state, target_name)
        if tp is not None:
            is_mafia = tp.role == "mafia"
            result = "마피아입니다" if is_mafia else "마피아가 아닙니다"
            state.private_info.setdefault(police_name, []).append(
                "%d일차 밤 조사 결과: %s%s %s" % (state.day, target_name, josa(target_name, "은/는"), result))
            inv_rec = {"police": police_name, "target": target_name, "is_mafia": is_mafia}
    # 비공개 기록 (마피아 처치 시도 / 의사 보호)
    for p in state.players:
        if p.role == "mafia" and kill_target:
            state.private_info[p.name].append("%d일차 밤 처치 시도: %s (%s)" % (
                state.day, kill_target, "의사가 살려 생존" if saved else "사망"))
        if p.role == "doctor" and protect_target:
            state.private_info[p.name].append("%d일차 밤 보호: %s" % (state.day, protect_target))
    state.night_history.append({"day": state.day, "kill": kill_target, "protect": protect_target,
                                "saved": saved, "killed": killed, "investigation": inv_rec})
    if killed:
        vp = get_player(state, killed)
        role_ko = config.ROLE_KO[vp.role]
        text = "%s(%s)%s 밤에 사망" % (killed, role_ko, josa(role_ko, "이/가"))
        ev = _log(state, "death", text, target=killed, phase="day")
        events.append(ev)
    else:
        events.append(_log(state, "no_death", "밤사이 아무도 사망하지 않았습니다", phase="day"))
    return events


def tally_votes(votes: dict) -> dict:
    """{투표자: 대상} -> {"executed": 이름|None, "tie": [...], "counts": {...}}"""
    counts = {}
    for target in votes.values():
        counts[target] = counts.get(target, 0) + 1
    if not counts:
        return {"executed": None, "tie": [], "counts": {}}
    top = max(counts.values())
    tops = [n for n, c in counts.items() if c == top]
    if len(tops) == 1:
        return {"executed": tops[0], "tie": [], "counts": counts}
    return {"executed": None, "tie": sorted(tops), "counts": counts}


def check_winner(state: GameState) -> Optional[str]:
    mafia = sum(1 for p in state.players if p.alive and p.role == "mafia")
    others = sum(1 for p in state.players if p.alive and p.role != "mafia")
    if mafia == 0:
        return "citizen"
    if mafia >= others:
        return "mafia"
    if state.day > config.MAX_DAYS:
        return "draw"
    return None


# =====================================
# 7. 정체 노출 필터
# =====================================

_SELF = r"(?:나는|나도|난|내가|저는|저도|전|제가|우리는|우린|우리가|나\s)"
_MAFIA_SELF = re.compile(_SELF + r"\s*(?:사실\s*|사실은\s*|진짜\s*|바로\s*)?마피아")
_MAFIA_TEAM = re.compile(r"(동료\s*마피아|마피아\s*동료|마피아\s*파트너|마피아끼리|같은\s*마피아|우리\s*마피아|마피아\s*팀|마피아\s*편)")
_META = re.compile(r"(시스템\s*프롬프트|private_info|PlayerView|프롬프트에\s*따르면|나는\s*AI|저는\s*AI)", re.I)
_NEGATION = re.compile(r"(아니|아냐|아닙|않|없)")


def contains_role_leak(text: str, role: str) -> bool:
    """공개 발언이 비공개 정보를 직접 드러내는지 단순 규칙으로 검사한다.

    - 모든 역할: 프롬프트/시스템 용어 노출
    - 마피아: "나는 마피아", "동료 마피아", "마피아끼리" 같은 자기/팀 정체 노출 (부정문은 거짓말이므로 허용)
    """
    if not text:
        return False
    if _META.search(text):
        return True
    if role == "mafia":
        if _MAFIA_TEAM.search(text):
            return True
        for m in _MAFIA_SELF.finditer(text):
            tail = text[m.end():m.end() + 10]
            if tail[:1] in ("를", "을", "의", "에", "들", "와", "과", "라"):
                if not tail.startswith("라고"):
                    continue
            if _NEGATION.search(tail):
                continue
            return True
    return False


# =====================================
# 8. 안전 호출 (safe_decide)
# =====================================


def _extract(result):
    """컨트롤러 반환값에서 (대상 텍스트, 이유) 추출."""
    if result is None:
        return None, ""
    if isinstance(result, str):
        return result, ""
    if isinstance(result, dict):
        return result.get("target"), str(result.get("reason", "") or "")
    target = getattr(result, "target", None)
    reason = getattr(result, "reason", "") or ""
    return target, str(reason)


def decide_full(fn, view, candidates, state, kind):
    """(target, reason, used_fallback). 절대 예외를 던지지 않는다. (LangGraph 노드에서 한 명 단위로 사용)"""
    if not candidates:
        return None, "", True
    attempts = 1 + max(0, int(config.LLM_MAX_RETRIES))
    for i in range(attempts):
        if i > 0:
            state.stats["retries"] += 1
        state.stats["llm_calls"] += 1
        try:
            result = fn(view)
        except Exception:
            continue
        raw, reason = _extract(result)
        try:
            chosen = normalize_choice(raw, candidates)
        except Exception:
            chosen = None
        if chosen is not None:
            return chosen, reason, False
    state.stats["fallbacks"] += 1
    return state.rng.choice(candidates), "", True


_decide_full = decide_full   # 호환용 별칭


def _clean_speech(raw, state: GameState):
    """Speech 호환 객체 -> 검증된 Speech 또는 None."""
    text = getattr(raw, "text", None) if not isinstance(raw, (str, dict)) else (
        raw if isinstance(raw, str) else raw.get("text"))
    if not isinstance(text, str) or not text.strip():
        return None
    thought = getattr(raw, "thought", "") if not isinstance(raw, (str, dict)) else (
        "" if isinstance(raw, str) else raw.get("thought", ""))
    susp_raw = getattr(raw, "suspicion", {}) if not isinstance(raw, (str, dict)) else (
        {} if isinstance(raw, str) else raw.get("suspicion", {}))
    susp = {}
    if isinstance(susp_raw, dict):
        for k, v in susp_raw.items():
            if get_player(state, k) is None:
                continue
            try:
                susp[k] = max(0, min(100, int(v)))
            except (TypeError, ValueError):
                continue
    return Speech(text=text.strip(), thought=str(thought or ""), suspicion=susp)


def safe_speak(fn, view: PlayerView, state: GameState, check_leak: bool = True):
    """발언 호출. (Speech, used_fallback). 형식 오류는 재시도, 정체 노출은 1회 재요청 후 중립 문장."""
    attempts = 1 + max(0, int(config.LLM_MAX_RETRIES))
    leak_retry_used = False
    i = 0
    speech = None
    while i < attempts:
        if i > 0:
            state.stats["retries"] += 1
        state.stats["llm_calls"] += 1
        i += 1
        try:
            raw = fn(view)
        except Exception:
            continue
        cand = _clean_speech(raw, state)
        if cand is None:
            continue
        if check_leak and contains_role_leak(cand.text, view.role):
            state.stats["leak_blocked"] += 1
            if not leak_retry_used:
                leak_retry_used = True
                attempts += 1       # 노출 시 1회 더 요청
                continue
            speech = Speech(text=state.rng.choice(NEUTRAL_LINES), thought=cand.thought,
                            suspicion=cand.suspicion)
            return speech, True
        return cand, False
    state.stats["fallbacks"] += 1
    return Speech(text=state.rng.choice(NEUTRAL_LINES)), True


def safe_decide(fn, view: PlayerView, candidates: list, state: GameState, kind: str):
    """컨트롤러 호출 -> 정규화 -> 재시도 -> 무작위 대체.

    kind: "kill" | "protect" | "investigate" | "vote" | "revote" -> (대상 이름, used_fallback)
          "speak" -> (Speech, used_fallback)
    stats 의 retries / fallbacks / llm_calls 를 갱신한다 (llm_calls 는 컨트롤러 호출마다 +1).
    """
    if kind == "speak":
        return safe_speak(fn, view, state)
    target, _reason, fb = _decide_full(fn, view, candidates, state, kind)
    return target, fb


# =====================================
# 9. 컨트롤러: Random / Human
# =====================================


class RandomController:
    """테스트용. LLM 없이 유효 대상 중 무작위 선택."""

    def __init__(self, name: str, seed: Optional[int] = None):
        self.name = name
        self.rng = random.Random(seed)

    def night_action(self, view: PlayerView) -> Optional[NightDecision]:
        if not view.candidates:
            return None
        return NightDecision(self.rng.choice(view.candidates), "무작위 선택")

    def speak(self, view: PlayerView) -> Speech:
        others = [n for n in view.alive if n != view.name]
        susp = {n: self.rng.randint(0, 100) for n in others}
        text = self.rng.choice([
            "저는 아직 잘 모르겠어요.", "조금 수상한 사람이 있는 것 같아요.",
            "투표 전에 의견을 더 들어보고 싶습니다.", "어젯밤 일이 마음에 걸리네요."])
        return Speech(text=text, thought="무작위 발언", suspicion=susp)

    def vote(self, view: PlayerView) -> VoteDecision:
        return VoteDecision(self.rng.choice(view.candidates), "무작위 투표")


class HumanController:
    """사람 플레이어. input() 기반. input_fn / print_fn 을 바꿔 끼우면 테스트에서 stdin 없이 쓸 수 있다.

    ask_rule: 입력이 '?질문' 이면 호출되는 규칙 질문 hook (str -> str).
    """

    def __init__(self, name: str, ask_rule: Optional[Callable] = None,
                 input_fn: Optional[Callable] = None, print_fn: Optional[Callable] = None,
                 max_invalid: int = 10):
        self.name = name
        self.ask_rule = ask_rule
        self.input_fn = input_fn or input
        self.print_fn = print_fn or print
        self.max_invalid = max_invalid

    # ----- 입력 처리 -----
    def _read(self, prompt: str) -> Optional[str]:
        """한 줄 입력. '?질문' 은 규칙 hook 으로 처리하고 다시 묻는다. EOF 면 None."""
        while True:
            try:
                s = self.input_fn(prompt)
            except (EOFError, StopIteration):
                return None
            if s is None:
                return None
            s = str(s).strip()
            if s.startswith("?"):
                q = s[1:].strip()
                if self.ask_rule is not None and q:
                    try:
                        self.print_fn("[사회자] " + str(self.ask_rule(q)))
                    except Exception as exc:
                        self.print_fn("[사회자] 답변 중 오류: %s" % exc)
                else:
                    self.print_fn("[사회자] 규칙 질문 기능이 연결되어 있지 않습니다.")
                continue
            return s

    def _show_status(self, view: PlayerView) -> None:
        p = self.print_fn
        p("-" * 40)
        p("[%d일차 %s] 당신은 %s 입니다." % (view.day, view.phase, view.role_ko))
        p("생존자: " + ", ".join(view.alive))
        if view.dead:
            p("사망자: " + ", ".join("%s(%s)" % (d["name"], d["role_ko"]) for d in view.dead))
        for info in view.private_info:
            p("[비공개] " + info)
        if view.partner_proposal:
            p("[동료 제안] %s (이유: %s)" % (view.partner_proposal.get("target"),
                                           view.partner_proposal.get("reason", "")))

    def _choose(self, view: PlayerView, title: str) -> Optional[NightDecision]:
        self._show_status(view)
        self.print_fn(title)
        for i, c in enumerate(view.candidates, 1):
            self.print_fn("  %d. %s" % (i, c))
        for _ in range(self.max_invalid):
            s = self._read("번호 또는 이름 (?질문 으로 규칙 문의) > ")
            if s is None:
                return None
            chosen = normalize_choice(s, view.candidates)
            if chosen is not None:
                reason = ""
                return NightDecision(chosen, reason)
            self.print_fn("올바른 번호나 이름이 아닙니다. 다시 입력하세요.")
        return None

    # ----- Controller 프로토콜 -----
    def night_action(self, view: PlayerView) -> Optional[NightDecision]:
        titles = {"mafia": "처치할 대상을 고르세요.", "doctor": "보호할 대상을 고르세요.",
                  "police": "조사할 대상을 고르세요."}
        if view.role not in titles:
            return None
        return self._choose(view, titles[view.role])

    def speak(self, view: PlayerView) -> Speech:
        self._show_status(view)
        s = self._read("발언을 입력하세요 (?질문 으로 규칙 문의, 빈 입력은 패스) > ")
        if not s:
            return Speech(text="이번에는 패스하겠습니다.")
        return Speech(text=s)

    def vote(self, view: PlayerView) -> VoteDecision:
        d = self._choose(view, "투표할 대상을 고르세요.")
        if d is None:
            return None
        return VoteDecision(d.target, d.reason)


# =====================================
# 10. 게임 엔진
# =====================================


def default_printer(event: dict) -> None:
    """이벤트를 한국어 한 줄로 출력."""
    t = event.get("type")
    text = event.get("text", "")
    if t == "phase":
        print("\n===== %s =====" % text)
    elif t == "speech":
        print("[%s] %s" % (event.get("actor"), text))
    elif t == "vote":
        print("[투표] %s -> %s" % (event.get("actor"), event.get("target")))
    elif t in ("private", "secret", "night_action"):
        who = event.get("to") or event.get("actor") or ""
        role = event.get("role_ko", "")
        print("  [비밀 · %s %s] %s" % (role, who, text) if role else "  [비밀 · %s] %s" % (who, text))
    elif t == "end":
        print("\n***** %s *****" % text)
    else:
        print(text)


class GameEngine:
    """규칙 진행기. 컨트롤러(사람/LLM/랜덤)는 이름 -> Controller 딕셔너리로 받는다.

    이벤트 가시성(event["visibility"]):
      public  : 항상 on_event 로 전달
      private : event["to"] 가 사람 플레이어일 때만 전달 (관전 모드에서는 비밀 이벤트로 취급)
      secret  : 관전 모드(사람 없음)에서만 전달 (마피아 밤 제안, 의사/경찰 선택 등)
    """

    def __init__(self, state: GameState, controllers: dict, on_event: Optional[Callable] = None):
        self.state = state
        self.controllers = dict(controllers)
        self.on_event = on_event or default_printer
        self.events = []
        self.rng = state.rng
        for p in state.players:
            if p.name not in self.controllers:
                self.controllers[p.name] = RandomController(p.name, seed=self.rng.randint(0, 10 ** 9))
        self.human_names = [p.name for p in state.players if p.is_human]
        self.spectator = not self.human_names

    # ----- 이벤트 -----
    def _emit(self, type_: str, text: str = "", visibility: str = "public", **kw) -> Optional[dict]:
        ev = {"type": type_, "day": self.state.day, "phase": self.state.phase,
              "text": text, "visibility": visibility}
        ev.update(kw)
        if visibility == "private" and ev.get("to") not in self.human_names:
            ev["visibility"] = visibility = "secret"
        if visibility == "secret" and not self.spectator:
            return None
        who = ev.get("to") or ev.get("actor")
        pl = get_player(self.state, who) if who else None
        if pl is not None and type_ in ("private", "night_action"):
            ev["role_ko"] = config.ROLE_KO[pl.role]
        self.events.append(ev)
        try:
            self.on_event(ev)
        except Exception:
            pass
        return ev

    def _emit_logged(self, entry: dict) -> None:
        """공개 로그 항목을 이벤트로도 내보낸다."""
        self._emit(entry["type"], entry["text"], actor=entry.get("actor", ""), target=entry.get("target", ""))

    def _since(self, mark: int) -> list:
        return self.events[mark:]

    def _controller(self, name: str):
        return self.controllers[name]

    # ----- 시작 -----
    def announce_start(self) -> list:
        mark = len(self.events)
        s = self.state
        self._emit("phase", "마피아 게임 시작 (%d명)" % len(s.players))
        self._emit("start", "참가자: " + ", ".join(p.name for p in s.players))
        for p in s.players:
            self._emit("private", "당신의 직업: %s" % config.ROLE_KO[p.role], visibility="private", to=p.name)
            for info in s.private_info.get(p.name, []):
                self._emit("private", info, visibility="private", to=p.name)
        return self._since(mark)

    # ----- 밤 -----
    def begin_night(self) -> list:
        mark = len(self.events)
        self.state.day += 1
        self.state.phase = "night"
        self._emit("phase", "%d일차 밤" % self.state.day)
        return self._since(mark)

    def _alive_with_role(self, role: str) -> list:
        return [p.name for p in self.state.players if p.alive and p.role == role]

    def note(self, name: str, thought: str, suspicion: Optional[dict] = None) -> None:
        if not thought and not suspicion:
            return
        self.state.notes.setdefault(name, []).append(
            {"day": self.state.day, "phase": self.state.phase, "thought": thought,
             "suspicion": dict(suspicion or {})})

    _note = note   # 호환용 별칭

    def collect_single_vote(self, voter: str, kind: str = "vote", tie_candidates: Optional[list] = None):
        """한 명의 투표 수집. 반환: 대상 이름 또는 None (속마음 note 기록 포함, 공개는 apply_votes 에서)."""
        p = get_player(self.state, voter)
        if p is None or not p.alive:
            return None
        view = build_view(self.state, voter, kind, tie_candidates=tie_candidates)
        target, reason, _fb = decide_full(self._controller(voter).vote, view,
                                          view.candidates, self.state, kind)
        if target is not None:
            self.note(voter, "[%s] %s: %s" % ("재투표" if kind == "revote" else "투표", target, reason))
        return target

    def collect_mafia_step(self, name: str, proposal: Optional[dict] = None):
        """마피아 한 명의 처치 결정. proposal={"target","reason"} 은 두 번째 마피아에게만 전달.

        반환: (target, reason). 실패 시 target=None. 노트와 [제안]/[최종] secret 이벤트까지 처리한다.
        """
        mafias = self._alive_with_role("mafia")
        view = build_view(self.state, name, "kill", partner_proposal=proposal)
        target, reason, _fb = decide_full(self._controller(name).night_action, view,
                                          view.candidates, self.state, "kill")
        if target is None:
            return None, reason
        self.note(name, "[밤 처치 선택] %s: %s" % (target, reason))
        label = ""
        if len(mafias) > 1:
            label = " [제안]" if proposal is None else " [최종]"
        self._emit("night_action", "%s%s 노립니다%s" % (target, josa(target, "을/를"),
                   " (이유: %s)" % reason if reason else "") + label,
                   visibility="secret", actor=name, target=target)
        return target, reason

    def collect_mafia_decision(self):
        """살아있는 마피아가 순서대로 결정. 반환: (최종 대상, {"proposals": [...]})"""
        proposal = None
        final = None
        proposals = []
        for name in self._alive_with_role("mafia"):
            target, reason = self.collect_mafia_step(name, proposal)
            if target is None:
                continue
            proposals.append({"mafia": name, "target": target, "reason": reason})
            proposal = {"target": target, "reason": reason}
            final = target
        return final, {"proposals": proposals}

    def collect_doctor_decision(self) -> Optional[str]:
        docs = self._alive_with_role("doctor")
        if not docs:
            return None
        name = docs[0]
        view = build_view(self.state, name, "protect")
        target, reason, _fb = _decide_full(self._controller(name).night_action, view,
                                           view.candidates, self.state, "protect")
        if target is not None:
            self._note(name, "[밤 보호 선택] %s: %s" % (target, reason))
            self._emit("night_action", "%s 보호" % target, visibility="secret",
                       actor=name, target=target)
        return target

    def collect_police_decision(self):
        pol = self._alive_with_role("police")
        if not pol:
            return None
        name = pol[0]
        view = build_view(self.state, name, "investigate")
        target, reason, _fb = _decide_full(self._controller(name).night_action, view,
                                           view.candidates, self.state, "investigate")
        if target is None:
            return None
        self._note(name, "[밤 조사 선택] %s: %s" % (target, reason))
        return (name, target)

    def apply_night(self, kill_target, protect_target, investigation=None) -> list:
        """밤 결과 반영 + 아침 발표 + 경찰 결과 전달(사람이면) + 승리 판정."""
        mark = len(self.events)
        for ev in resolve_night(self.state, kill_target, protect_target, investigation):
            self._emit_logged(ev)
        if investigation:
            police = investigation[0]
            rec = self.state.night_history[-1]["investigation"]
            if rec:
                info = "%s 조사 결과: %s" % (rec["target"], "마피아다" if rec["is_mafia"] else "마피아가 아니다")
                self._emit("private", info, visibility="private", to=police)
        self.check_end()
        return self._since(mark)

    def run_night(self) -> list:
        mark = len(self.events)
        self.begin_night()
        kill, _info = self.collect_mafia_decision()
        protect = self.collect_doctor_decision()
        inv = self.collect_police_decision()
        self.apply_night(kill, protect, inv)
        return self._since(mark)

    # ----- 낮 -----
    def begin_day(self) -> list:
        mark = len(self.events)
        self.state.phase = "day"
        self._emit("phase", "%d일차 낮 토론" % self.state.day)
        return self._since(mark)

    def discussion_order(self) -> list:
        order = alive_names(self.state)
        self.rng.shuffle(order)
        return order

    def run_discussion_turn(self, name: str) -> list:
        """한 명의 발언 1회. 속마음은 notes 에, 발언은 공개 로그에만 남는다."""
        mark = len(self.events)
        p = get_player(self.state, name)
        if p is None or not p.alive:
            return []
        view = build_view(self.state, name, "speak")
        speech, _fb = safe_speak(self._controller(name).speak, view, self.state, check_leak=not p.is_human)
        text = speech.text
        if not p.is_human and len(text) > config.SPEECH_MAX_CHARS * 2:
            text = text[:config.SPEECH_MAX_CHARS * 2].rstrip() + "..."
        self._note(name, speech.thought, speech.suspicion)
        entry = _log(self.state, "speech", text, actor=name, phase="day")
        self._emit_logged(entry)
        return self._since(mark)

    def run_day(self) -> list:
        mark = len(self.events)
        if self.state.winner is not None:
            return []
        self.begin_day()
        for _ in range(max(1, config.DISCUSSION_ROUNDS)):
            for name in self.discussion_order():
                self.run_discussion_turn(name)
        return self._since(mark)

    # ----- 투표 -----
    def collect_votes(self, kind: str = "vote", tie_candidates: Optional[list] = None) -> dict:
        """살아있는 모두의 투표 수집 (공개는 apply_votes 에서 한꺼번에). kind: vote | revote"""
        votes = {}
        for name in alive_names(self.state):
            target = self.collect_single_vote(name, kind, tie_candidates)
            if target is not None:
                votes[name] = target
        return votes

    def apply_votes(self, votes: dict, kind: str = "vote") -> dict:
        """투표 공개 + 집계 + 처형. 동점이면 result["tie"] 가 채워진다(재투표는 호출자가)."""
        s = self.state
        s.phase = "vote"
        clean = {}
        for v, t in votes.items():
            pv, pt = get_player(s, v), get_player(s, t)
            if pv is not None and pt is not None and pv.alive and pt.alive and v != t:
                clean[v] = t
        for v, t in clean.items():
            entry = _log(s, kind, "%s -> %s" % (v, t), actor=v, target=t, phase="vote")
            self._emit("vote", entry["text"], actor=v, target=t, kind=kind)
        result = tally_votes(clean)
        s.vote_history.append({"day": s.day, "kind": kind, "votes": dict(clean),
                               "counts": dict(result["counts"]), "executed": result["executed"],
                               "tie": list(result["tie"])})
        events = []
        if result["executed"]:
            victim = get_player(s, result["executed"])
            victim.alive = False
            role_ko = config.ROLE_KO[victim.role]
            text = "%s(%s)%s 투표로 처형됨" % (victim.name, role_ko, josa(role_ko, "이/가"))
            entry = _log(s, "execution", text, target=victim.name, phase="vote")
            self._emit_logged(entry)
        elif result["tie"] and kind == "revote":
            entry = _log(s, "no_execution", "재투표도 동점이어서 아무도 처형되지 않음", phase="vote")
            self._emit_logged(entry)
        elif result["tie"]:
            self._emit("phase", "동점: " + ", ".join(result["tie"]) + " 재투표")
        elif not clean:
            entry = _log(s, "no_execution", "유효한 투표가 없어 아무도 처형되지 않음", phase="vote")
            self._emit_logged(entry)
        result["events"] = events
        return result

    def run_vote(self) -> list:
        mark = len(self.events)
        if self.state.winner is not None:
            return []
        self.state.phase = "vote"
        self._emit("phase", "%d일차 투표" % self.state.day)
        res = self.apply_votes(self.collect_votes("vote"), "vote")
        if res["tie"]:
            votes2 = self.collect_votes("revote", res["tie"])
            self.apply_votes(votes2, "revote")
        self.check_end()
        return self._since(mark)

    # ----- 종료 -----
    def check_end(self) -> Optional[str]:
        s = self.state
        if s.winner is not None:
            return s.winner
        w = check_winner(s)
        if w is not None:
            s.winner = w
            s.phase = "end"
            label = {"citizen": "시민팀 승리", "mafia": "마피아팀 승리", "draw": "무승부"}[w]
            self._emit("end", label, winner=w)
        return w

    def run(self, max_days: Optional[int] = None) -> str:
        max_days = max_days or config.MAX_DAYS
        s = self.state
        if s.day == 0:
            self.announce_start()
        while s.winner is None:
            if s.day >= max_days:
                s.winner = "draw"
                s.phase = "end"
                self._emit("end", "무승부 (%d일 경과)" % max_days, winner="draw")
                break
            self.run_night()
            if s.winner is not None:
                break
            self.run_day()
            self.run_vote()
        return s.winner


# =====================================
# 11. 로그 저장 / 결과 공개
# =====================================


def save_log(state: GameState, path=None) -> Path:
    """게임 전체(역할, 속마음 포함)를 JSON 으로 저장한다."""
    if path is None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = config.LOG_DIR / ("mafia_%s_seed%s.json" % (time.strftime("%Y%m%d_%H%M%S"), state.seed))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "players": [asdict(p) for p in state.players],
        "day": state.day, "winner": state.winner, "seed": state.seed,
        "public_log": state.public_log, "private_info": state.private_info,
        "notes": state.notes, "night_history": state.night_history,
        "vote_history": state.vote_history, "stats": state.stats,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def reveal_report(state: GameState) -> str:
    """게임 종료 후: 승자, 전체 역할, 밤 기록, 플레이어별 속마음 요약."""
    label = {"citizen": "시민팀 승리", "mafia": "마피아팀 승리", "draw": "무승부", None: "진행 중"}[state.winner]
    lines = ["===== 게임 결과: %s (%d일차) =====" % (label, state.day), "", "[역할 공개]"]
    for p in state.players:
        lines.append("  %s : %s (%s)%s" % (p.name, config.ROLE_KO[p.role], "생존" if p.alive else "사망",
                                           " [사람]" if p.is_human else ""))
    lines += ["", "[밤 기록]"]
    for h in state.night_history:
        inv = h.get("investigation")
        inv_t = ""
        if inv:
            inv_t = ", 경찰 조사 %s(%s)" % (inv["target"], "마피아" if inv["is_mafia"] else "비마피아")
        lines.append("  %d일차: 처치 %s, 보호 %s, %s%s" % (
            h["day"], h["kill"], h["protect"],
            "의사가 살림" if h["saved"] else ("사망 " + str(h["killed"])), inv_t))
    lines += ["", "[속마음 요약]"]
    for p in state.players:
        notes = state.notes.get(p.name, [])
        if not notes:
            continue
        lines.append("  <%s / %s>" % (p.name, config.ROLE_KO[p.role]))
        for n in notes[-5:]:
            t = (n.get("thought") or "").replace("\n", " ")
            if len(t) > 80:
                t = t[:80] + "..."
            top = sorted((n.get("suspicion") or {}).items(), key=lambda x: -x[1])[:2]
            sus = (" 의심: " + ", ".join("%s %d" % kv for kv in top)) if top else ""
            lines.append("    %d일차 %s: %s%s" % (n["day"], n["phase"], t, sus))
    lines += ["", "[통계] " + ", ".join("%s=%s" % kv for kv in state.stats.items())]
    return "\n".join(lines)
