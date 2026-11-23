# =====================================
# mafia_graph.py : LangGraph 로 만든 마피아 게임 진행 그래프
# =====================================
# - 규칙 판정(밤 결과, 투표 집계, 승리 판정)은 전부 mafia_core 의 함수/GameEngine 을 그대로 재사용한다.
#   이 파일은 "무엇을 어떤 순서로, 어떤 조건에서 다음에 실행할까" 라는 진행 흐름만 정의한다.
# - 그래프 구조
#     START -> setup -> night_start -> night_mafia_first -> night_mafia_second -> night_doctor -> night_police -> morning
#     morning -(승부 결정)-> ending / -(계속)-> day_start -> speak(반복) -> vote_start -> vote(반복)
#     -> tally -(동점 재투표)-> vote / -> execution -(승부 결정)-> ending / -(계속)-> night_start
# - 사람 입력은 langgraph.types.interrupt() 로 그래프를 멈추고, Command(resume=...) 로 재개한다.
# - interrupt 로 재개하면 "멈췄던 노드를 처음부터 다시 실행"한다. 그래서 AI 호출이 많은 일을
#   한 노드에 몰지 않고, 발언/투표를 한 명 단위 노드로 쪼갰다 (사람 차례에 AI 호출이 반복되지 않도록).

from typing import Any, Callable, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt

import config
import mafia_core as core
from mafia_core import (
    GameEngine,
    NightDecision,
    Speech,
    VoteDecision,
    alive_names,
    build_view,
    default_printer,
)


# =====================================
# 1. 그래프 상태 (State)
# =====================================
class MafiaState(TypedDict, total=False):
    """노드들이 공유하는 상태. 각 노드는 바꾼 키만 딕셔너리로 돌려준다."""

    game: core.GameState   # 게임 전체 상태 (진실은 이 안에 있고, 변경은 mafia_core 함수만 한다)
    kill: Optional[str]    # 이번 밤 마피아가 정한 처치 대상
    proposal: Optional[dict]  # 첫 마피아의 제안 {"mafia","target","reason"} (둘째 마피아가 본다)
    protect: Optional[str]  # 이번 밤 의사가 정한 보호 대상
    inv: Optional[tuple]   # (경찰, 조사 대상)
    order: list            # 이번 단계(발언/투표) 진행 순서
    idx: int               # order 에서 현재 차례 번호
    vote_kind: str         # "vote" | "revote"
    tie: list              # 재투표 후보
    votes: dict            # {투표자: 대상}
    result: dict           # 마지막 개표 결과 (표시용)


# =====================================
# 2. 사람 입력용 컨트롤러 (interrupt 사용)
# =====================================
def _human_payload(view, kind: str, prompt: str) -> dict:
    return {
        "kind": kind,                  # kill | protect | investigate | speak | vote | revote
        "player": view.name,
        "prompt": prompt,
        "day": view.day,
        "role_ko": view.role_ko,
        "alive": list(view.alive),
        "dead": list(view.dead),
        "private_info": list(view.private_info),
        "partner_proposal": view.partner_proposal,
        "candidates": list(view.candidates),
    }


class _InterruptPassthrough(BaseException):
    """interrupt() 가 던진 GraphInterrupt 를 mafia_core 의 safe_decide(except Exception)가 삼키지 못하게
    BaseException 으로 감싸서 노드 밖까지 통과시키는 용도. 노드 래퍼(_bubble)가 원래 예외로 되돌린다."""

    def __init__(self, original):
        super().__init__("interrupt")
        self.original = original


def _ask(payload: dict):
    try:
        return interrupt(payload)
    except GraphInterrupt as exc:
        raise _InterruptPassthrough(exc)


def _bubble(fn):
    """노드 함수를 감싸서 _InterruptPassthrough 를 원래 GraphInterrupt 로 복원한다."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _InterruptPassthrough as wrapped:
            raise wrapped.original
    wrapper.__name__ = fn.__name__
    return wrapper


class GraphHumanController:
    """사람 플레이어. input() 대신 interrupt() 로 그래프를 멈추고 입력을 기다린다.

    interrupt(payload) 를 호출하면 그래프 실행이 멈추고 payload 가 호출자에게 전달된다.
    호출자가 Command(resume=값) 으로 재개하면 interrupt() 가 그 값을 돌려준다.
    """

    def __init__(self, name: str):
        self.name = name

    def night_action(self, view) -> Optional[NightDecision]:
        kinds = {
            "mafia": ("kill", "처치할 대상을 고르세요."),
            "doctor": ("protect", "보호할 대상을 고르세요."),
            "police": ("investigate", "조사할 대상을 고르세요."),
        }
        if view.role not in kinds:
            return None
        kind, prompt = kinds[view.role]
        answer = _ask(_human_payload(view, kind, prompt))
        return NightDecision(str(answer), "사람 선택")

    def speak(self, view) -> Speech:
        answer = _ask(_human_payload(view, "speak", "발언을 입력하세요. (빈 입력은 패스)"))
        text = str(answer).strip()
        return Speech(text=text if text else "이번에는 패스하겠습니다.")

    def vote(self, view) -> VoteDecision:
        answer = _ask(_human_payload(view, "vote", "투표할 대상을 고르세요."))
        return VoteDecision(str(answer), "사람 선택")


def _alive_mafias(engine):
    return [p.name for p in engine.state.players if p.alive and p.role == "mafia"]

# =====================================
# 3. 그래프 만들기
# =====================================
def build_graph(controllers: dict, on_event: Callable = default_printer, checkpointer=None):
    """컨트롤러 딕셔너리(이름 -> Controller)를 받아 컴파일된 그래프를 돌려준다.

    controllers 에는 RandomController / LLMPlayer / GraphHumanController 를 섞어 쓸 수 있다.
    (컨트롤러는 직렬화할 수 없으므로 State 가 아니라 이 함수의 클로저에 둔다.)
    """
    if checkpointer is None:
        # GameState(dataclass, random.Random 포함)를 체크포인트에 저장하기 위해 pickle 허용
        checkpointer = MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))

    def engine_of(state: MafiaState) -> GameEngine:
        return GameEngine(state["game"], controllers, on_event=on_event)

    # ----- 3-1. 게임 시작 -----
    def setup(state: MafiaState) -> dict:
        engine_of(state).announce_start()
        return {"game": state["game"]}

    # ----- 3-2. 밤 (마피아 -> 의사 -> 경찰 순으로 노드를 나눈다) -----
    def night_start(state: MafiaState) -> dict:
        engine_of(state).begin_night()
        return {"game": state["game"], "kill": None, "proposal": None, "protect": None, "inv": None}

    # 마피아가 2명이면 첫 마피아(제안) -> 둘째 마피아(최종) 순서로 노드를 나눈다.
    # 사람이 둘째일 때 interrupt 로 재개해도 첫 마피아(AI)는 다시 호출되지 않고, 저장된 제안을 그대로 쓴다.
    def night_mafia_first(state: MafiaState) -> dict:
        engine = engine_of(state)
        mafias = _alive_mafias(engine)
        if not mafias:
            return {"game": state["game"], "kill": None, "proposal": None}
        target, reason = engine.collect_mafia_step(mafias[0], None)
        proposal = {"mafia": mafias[0], "target": target, "reason": reason} if target else None
        return {"game": state["game"], "proposal": proposal, "kill": target}

    def night_mafia_second(state: MafiaState) -> dict:
        engine = engine_of(state)
        mafias = _alive_mafias(engine)
        proposal = state.get("proposal")
        if len(mafias) < 2:
            return {"game": state["game"], "kill": state.get("kill")}
        partner = {"target": proposal["target"], "reason": proposal["reason"]} if proposal else None
        target, _reason = engine.collect_mafia_step(mafias[1], partner)
        return {"game": state["game"], "kill": target or state.get("kill")}
    def night_doctor(state: MafiaState) -> dict:
        protect = engine_of(state).collect_doctor_decision()
        return {"game": state["game"], "protect": protect}

    def night_police(state: MafiaState) -> dict:
        inv = engine_of(state).collect_police_decision()
        return {"game": state["game"], "inv": inv}

    # ----- 3-3. 아침: 밤 결과 판정 + 승리 판정 (둘 다 mafia_core 가 한다) -----
    def morning(state: MafiaState) -> dict:
        engine_of(state).apply_night(state.get("kill"), state.get("protect"), state.get("inv"))
        return {"game": state["game"]}

    # ----- 3-4. 낮 토론: 한 번 실행에 한 명씩 발언 -----
    def day_start(state: MafiaState) -> dict:
        engine = engine_of(state)
        engine.begin_day()
        order = []
        for _ in range(max(1, config.DISCUSSION_ROUNDS)):
            order.extend(engine.discussion_order())
        return {"game": state["game"], "order": order, "idx": 0}

    def speak(state: MafiaState) -> dict:
        name = state["order"][state["idx"]]
        engine_of(state).run_discussion_turn(name)
        return {"game": state["game"], "idx": state["idx"] + 1}

    # ----- 3-5. 투표: 한 번 실행에 한 명씩 투표 -----
    def vote_start(state: MafiaState) -> dict:
        game = state["game"]
        game.phase = "vote"
        engine_of(state)._emit("phase", "%d일차 투표" % game.day)
        return {"game": game, "order": alive_names(game), "idx": 0,
                "vote_kind": "vote", "tie": [], "votes": {}}

    def vote(state: MafiaState) -> dict:
        game = state["game"]
        engine = engine_of(state)
        name = state["order"][state["idx"]]
        # 정규화/재시도/무작위 대체와 속마음 기록은 core 의 collect_single_vote 가 한다
        target = engine.collect_single_vote(name, state["vote_kind"], state.get("tie") or None)
        votes = dict(state.get("votes") or {})
        if target is not None:
            votes[name] = target
        return {"game": game, "votes": votes, "idx": state["idx"] + 1}
    def tally(state: MafiaState) -> dict:
        """개표(공개) + 처형. 동점이면 동점자만 후보로 재투표 1회."""
        game = state["game"]
        engine = engine_of(state)
        kind = state["vote_kind"]
        res = engine.apply_votes(state.get("votes") or {}, kind)
        summary = {"executed": res["executed"], "tie": res["tie"], "counts": res["counts"]}
        if res["tie"] and kind == "vote":
            return {"game": game, "result": summary, "vote_kind": "revote", "tie": list(res["tie"]),
                    "votes": {}, "order": alive_names(game), "idx": 0}
        return {"game": game, "result": summary}

    # ----- 3-6. 처형 후 승리 판정 -----
    def execution(state: MafiaState) -> dict:
        game = state["game"]
        engine = engine_of(state)
        if engine.check_end() is None and game.day >= config.MAX_DAYS:
            game.winner = "draw"
            game.phase = "end"
            engine._emit("end", "무승부 (%d일 경과)" % config.MAX_DAYS, winner="draw")
        return {"game": game}

    # ----- 3-7. 종료 -----
    def ending(state: MafiaState) -> dict:
        game = state["game"]
        label = {"citizen": "시민팀", "mafia": "마피아팀", "draw": "무승부"}.get(game.winner, "알 수 없음")
        engine_of(state)._emit("phase", "게임 종료: %s (%d일차)" % (label, game.day))
        return {"game": game}

    # ----- 조건부 엣지(분기 함수) -----
    def after_morning(state: MafiaState) -> str:
        return "ending" if state["game"].winner else "day_start"

    def more_speakers(state: MafiaState) -> str:
        return "speak" if state["idx"] < len(state["order"]) else "vote_start"

    def more_voters(state: MafiaState) -> str:
        return "vote" if state["idx"] < len(state["order"]) else "tally"

    def after_tally(state: MafiaState) -> str:
        # tally 가 재투표를 준비했다면(idx 0, vote_kind == revote, 투표가 비어 있음) 다시 투표 노드로
        if state["vote_kind"] == "revote" and not state["votes"] and state["idx"] == 0:
            return "vote"
        return "execution"

    def after_execution(state: MafiaState) -> str:
        return "ending" if state["game"].winner else "night_start"

    # ----- 그래프 조립 -----
    builder = StateGraph(MafiaState)
    for name, fn in [
        ("setup", setup), ("night_start", night_start), ("night_mafia_first", night_mafia_first),
        ("night_mafia_second", night_mafia_second),
        ("night_doctor", night_doctor), ("night_police", night_police), ("morning", morning),
        ("day_start", day_start), ("speak", speak), ("vote_start", vote_start), ("vote", vote),
        ("tally", tally), ("execution", execution), ("ending", ending),
    ]:
        builder.add_node(name, _bubble(fn))

    builder.add_edge(START, "setup")
    builder.add_edge("setup", "night_start")
    builder.add_edge("night_start", "night_mafia_first")
    builder.add_edge("night_mafia_first", "night_mafia_second")
    builder.add_edge("night_mafia_second", "night_doctor")
    builder.add_edge("night_doctor", "night_police")
    builder.add_edge("night_police", "morning")
    builder.add_conditional_edges("morning", after_morning, {"ending": "ending", "day_start": "day_start"})
    builder.add_edge("day_start", "speak")
    builder.add_conditional_edges("speak", more_speakers, {"speak": "speak", "vote_start": "vote_start"})
    builder.add_edge("vote_start", "vote")
    builder.add_conditional_edges("vote", more_voters, {"vote": "vote", "tally": "tally"})
    builder.add_conditional_edges("tally", after_tally, {"vote": "vote", "execution": "execution"})
    builder.add_conditional_edges("execution", after_execution,
                                  {"ending": "ending", "night_start": "night_start"})
    builder.add_edge("ending", END)

    return builder.compile(checkpointer=checkpointer)


# =====================================
# 4. 실행 도우미
# =====================================
def make_config(thread_id: str = "mafia-1", recursion_limit: int = 2000) -> dict:
    """thread_id 는 체크포인터가 '이 게임'의 저장 기록을 구분하는 이름표다."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def get_game(graph, cfg: dict) -> core.GameState:
    """체크포인트에 저장된 최신 GameState 를 꺼낸다."""
    return graph.get_state(cfg).values["game"]


def run_spectator(graph, game: core.GameState, thread_id: str = "spectator-1") -> core.GameState:
    """관전 모드: 사람 입력 없이 끝까지 실행하고 최종 GameState 를 돌려준다."""
    cfg = make_config(thread_id)
    result = graph.invoke({"game": game}, cfg)
    if "__interrupt__" in result:
        raise RuntimeError("관전 모드인데 사람 입력 요청이 발생했습니다. run_with_human 을 사용하세요.")
    return get_game(graph, cfg)


def ask_human_from_input(payload: dict, ask_rule: Optional[Callable] = None) -> str:
    """interrupt payload 를 보여주고 input() 으로 답을 받는다. '?질문' 은 사회자(ask_rule)가 답한다."""
    print("-" * 40)
    print("[%d일차] 당신은 %s 입니다." % (payload["day"], payload["role_ko"]))
    print("생존자:", ", ".join(payload["alive"]))
    if payload["dead"]:
        print("사망자:", ", ".join("%s(%s)" % (d["name"], d["role_ko"]) for d in payload["dead"]))
    for info in payload["private_info"]:
        print("[비공개]", info)
    if payload.get("partner_proposal"):
        print("[동료 제안]", payload["partner_proposal"])
    print(payload["prompt"])
    cands = payload["candidates"]
    if payload["kind"] != "speak":
        for i, c in enumerate(cands, 1):
            print("  %d. %s" % (i, c))
    while True:
        text = input("> ").strip()
        if text.startswith("?"):
            if ask_rule is not None and text[1:].strip():
                try:   # 사회자(LLM) 오류가 나도 게임은 계속한다
                    print("[사회자]", ask_rule(text[1:].strip()))
                except Exception as exc:
                    print("[사회자] 답변 중 오류: %s" % exc)
            else:
                print("[사회자] 규칙 질문 기능이 연결되어 있지 않습니다.")
            continue
        if payload["kind"] == "speak":
            return text
        chosen = core.normalize_choice(text, cands)
        if chosen is not None:
            return chosen
        print("올바른 번호나 이름이 아닙니다. 다시 입력하세요.")


def run_with_human(graph, game: core.GameState, thread_id: str = "human-1",
                   answer_fn: Optional[Callable] = None, ask_rule: Optional[Callable] = None,
                   max_interrupts: int = 500) -> core.GameState:
    """참여 모드: interrupt 가 걸릴 때마다 answer_fn(payload) 의 답으로 Command(resume=...) 재개.

    answer_fn 을 주지 않으면 input() 으로 사람이 직접 입력한다. (테스트에서는 함수를 넘겨 자동 응답)
    """
    if answer_fn is None:
        def answer_fn(payload):
            return ask_human_from_input(payload, ask_rule)
    cfg = make_config(thread_id)
    result = graph.invoke({"game": game}, cfg)
    count = 0
    while "__interrupt__" in result:
        count += 1
        if count > max_interrupts:
            raise RuntimeError("사람 입력 요청이 너무 많습니다.")
        payload = result["__interrupt__"][0].value
        result = graph.invoke(Command(resume=answer_fn(payload)), cfg)
    return get_game(graph, cfg)


# =====================================
# 5. 그래프 시각화
# =====================================
def graph_mermaid(graph) -> str:
    """Mermaid 텍스트 (인터넷 불필요)."""
    return graph.get_graph().draw_mermaid()


def show_graph(graph, png_path=None) -> str:
    """가능하면 PNG 를 그리고(mermaid.ink 접속 필요), 안 되면 Mermaid 텍스트를 출력한다."""
    mermaid = graph_mermaid(graph)
    try:
        png = graph.get_graph().draw_mermaid_png()
        if png_path:
            with open(png_path, "wb") as f:
                f.write(png)
        try:
            from IPython.display import Image, display
            display(Image(png))
        except Exception:
            pass
        print("===== 그래프 PNG 생성 완료 =====")
    except Exception as exc:
        print("PNG 생성 실패(인터넷 필요): %s" % type(exc).__name__)
        print("아래 Mermaid 텍스트를 https://mermaid.live 에 붙여넣거나 마크다운에서 확인하세요.\n")
    print(mermaid)
    return mermaid


# =====================================
# 6. N판 자동 실행 통계
# =====================================
def run_batch(n: int, controller_factory: Callable, seed0: int = 0, verbose: bool = False) -> dict:
    """관전 모드로 n 판 실행해 승률/평균 일수/안전장치 횟수를 집계한다.

    controller_factory(game) -> {이름: Controller}  (예: lambda g: build_controllers(g, vs))
    """
    results = []
    quiet = (lambda ev: None) if not verbose else default_printer
    for i in range(n):
        game = core.new_game(list(config.CHARACTERS), None, seed=seed0 + i)
        graph = build_graph(controller_factory(game), on_event=quiet)
        final = run_spectator(graph, game, thread_id="batch-%d" % (seed0 + i))
        results.append(final)
    wins = {"citizen": 0, "mafia": 0, "draw": 0}
    for g in results:
        wins[g.winner] = wins.get(g.winner, 0) + 1
    total = {"retries": 0, "fallbacks": 0, "leak_blocked": 0, "llm_calls": 0}
    for g in results:
        for k in total:
            total[k] += g.stats.get(k, 0)
    return {
        "games": n,
        "wins": wins,
        "mafia_rate": wins["mafia"] / n if n else 0.0,
        "citizen_rate": wins["citizen"] / n if n else 0.0,
        "avg_days": sum(g.day for g in results) / n if n else 0.0,
        "stats": total,
        "states": results,
    }


def print_batch_summary(summary: dict) -> None:
    n = summary["games"]
    print("===== 관전 %d판 통계 =====" % n)
    print("마피아팀 승률 : %.1f%% (%d판)" % (summary["mafia_rate"] * 100, summary["wins"]["mafia"]))
    print("시민팀 승률   : %.1f%% (%d판)" % (summary["citizen_rate"] * 100, summary["wins"]["citizen"]))
    print("무승부        : %d판" % summary["wins"].get("draw", 0))
    print("평균 진행 일수: %.2f일" % summary["avg_days"])
    st = summary["stats"]
    print("안전장치 합계 : 재시도 %d, fallback %d, 정체노출 차단 %d, AI 결정 요청 %d" % (
        st["retries"], st["fallbacks"], st["leak_blocked"], st["llm_calls"]))
