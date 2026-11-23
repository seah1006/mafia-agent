# =====================================
# 회귀 테스트 (LLM 없이 실행)
# =====================================
# 1) 경찰 조사 결과 문장을 에이전트가 올바르게 해석하는지 (마피아 / 마피아 아님)
# 2) LangGraph 버전에서 두 번째 마피아가 동료의 제안(partner_proposal)을 실제로 받는지
# 실행: python tests/test_regression.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import mafia_core as core
from mafia_agents import private_facts
import mafia_graph as mg


def check(name, ok, detail=""):
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


# =====================================
# 1. 경찰 조사 결과 해석
# =====================================
def test_police_facts():
    ok = True
    for seed in range(30):
        state = core.new_game(list(config.CHARACTERS), None, seed)
        police = next(p.name for p in state.players if p.role == "police")
        mafia = next(p.name for p in state.players if p.role == "mafia")
        citizen = next(p.name for p in state.players if p.role == "citizen")
        state.day = 1
        core.resolve_night(state, None, None, (police, citizen))
        state.day = 2
        core.resolve_night(state, None, None, (police, mafia))
        view = core.build_view(state, police, "vote")
        _, cleared, found = private_facts(view)
        if cleared != [citizen] or found != [mafia]:
            ok = False
            print("   seed", seed, view.private_info, "cleared", cleared, "found", found)
    return check("경찰 조사 결과 해석 (30판)", ok)


# =====================================
# 2. 그래프: 두 번째 마피아가 동료 제안을 받는가
# =====================================
class RecordingRandom(core.RandomController):
    """RandomController 와 같지만 밤 행동 때 받은 partner_proposal 을 기록한다."""

    seen = []

    def night_action(self, view):
        if view.role == "mafia":
            RecordingRandom.seen.append((view.day, self.name, view.partner_proposal))
        return super().night_action(view)


def test_graph_partner_proposal():
    nights_with_two = 0
    received = 0
    for seed in range(20):
        RecordingRandom.seen = []
        game = core.new_game(list(config.CHARACTERS), None, seed)
        ctrls = {p.name: RecordingRandom(p.name, seed=seed) for p in game.players}
        graph = mg.build_graph(ctrls, on_event=lambda e: None)
        mg.run_spectator(graph, game, thread_id="reg-%d" % seed)
        by_day = {}
        for day, name, pp in RecordingRandom.seen:
            by_day.setdefault(day, []).append(pp)
        for day, pps in by_day.items():
            if len(pps) == 2:
                nights_with_two += 1
                if pps[0] is None and isinstance(pps[1], dict) and pps[1].get("target"):
                    received += 1
    return check("그래프 둘째 마피아 동료 제안 수신", nights_with_two > 0 and received == nights_with_two,
                 "(%d/%d 밤)" % (received, nights_with_two))


# =====================================
# 3. 일관성 검사: 고집부리는 LLM 이어도 확정 정보에 반하는 투표는 나오지 않는가
# =====================================
def test_consistency_guard():
    from mafia_agents import LLMPlayer
    ok = True
    for seed in range(10):
        state = core.new_game(list(config.CHARACTERS), None, seed)
        police = next(p.name for p in state.players if p.role == "police")
        citizen = next(p.name for p in state.players if p.role == "citizen")
        mafias = [p.name for p in state.players if p.role == "mafia"]
        state.day = 1
        core.resolve_night(state, None, None, (police, citizen))
        # 경찰: 항상 확인된 시민을 고르려는 가짜 LLM
        pl = LLMPlayer(police, "police", None, state=state)
        pl._decide = lambda v, a, warning="": ((citizen if citizen in v.candidates else v.candidates[0]), "고집")
        out = pl.vote(core.build_view(state, police, "vote"))
        ok = ok and out.target != citizen
        # 마피아: 항상 동료를 고르려는 가짜 LLM
        m0, m1 = mafias
        pm = LLMPlayer(m0, "mafia", None, state=state)
        pm._decide = lambda v, a, warning="": ((m1 if m1 in v.candidates else v.candidates[0]), "고집")
        out2 = pm.vote(core.build_view(state, m0, "vote"))
        ok = ok and out2.target != m1
    return check("일관성 검사(후보 제한 재질문) 10판", ok)


# =====================================
# 4. v1.3 발언 보정과 사고 과정 보기 (LLM 없이)
# =====================================
def test_speech_fixes_and_trace():
    import mafia_agents as ag
    ok = True
    # 자기 이름 3인칭: 문장 삭제, 경찰 주장 문장은 1인칭으로
    ok &= ag.fix_self_reference("박준수가 수상해요. 윤서진을 의심해야 합니다.", "윤서진") == "박준수가 수상해요."
    ok &= ag.fix_self_reference("저 윤서진은 경찰입니다.", "윤서진") == "나는 경찰입니다."
    # 경찰 주장 감지 ("경찰이 누구지"는 주장이 아님)
    ok &= bool(ag.CLAIM_RE.search("나는 경찰이야.")) and not ag.CLAIM_RE.search("경찰이 누구지 궁금해요")
    # 몰아가기 감지
    today = [{"who": w, "text": "강태호 수상"} for w in ("A", "B", "C")]
    ok &= ag.pile_on_target(today, ["강태호", "A", "B", "C"]) == ("강태호", 3)
    # 경찰 주장 목록이 추론 재료에 들어가는가
    state = core.new_game(list(config.CHARACTERS), None, 3)
    state.day = 1
    state.phase = "day"
    who = state.players[0].name
    core._log(state, "speech", "나는 경찰입니다. %s는 마피아가 아닙니다." % state.players[1].name, actor=who, phase="day")
    mat = ag.reasoning_material(core.build_view(state, state.players[2].name, "speak"))
    ok &= ("경찰이라고 주장한 사람: %s" % who) in mat
    # 사람이 참여하면 사고 과정 보기는 자동으로 꺼진다 (정보 분리)
    msgs = []
    h_state = core.new_game(list(config.CHARACTERS)[:7], config.HUMAN_NAME, 5)
    ctrls = ag.build_controllers(h_state, None, human_ask_rule=False, show_thinking=True, printer=msgs.append)
    ok &= all(getattr(c, "trace_out", None) is None for c in ctrls.values()) and len(msgs) == 1
    # 관전 모드에서는 켜지고, 상자가 출력된다
    s_ctrls = ag.build_controllers(state, None, human_ask_rule=False, show_thinking=True, printer=msgs.append)
    pl = s_ctrls[state.players[2].name]
    pl._steps = ["도구 호출: vote(target='x')"]
    pl._emit_trace(core.build_view(state, pl.name, "vote"), "vote", "투표: x", compact=True)
    ok &= "사고 과정" in msgs[-1] and "도구 호출" in msgs[-1]
    return check("v1.3 발언 보정(3인칭/경찰 주장/몰아가기) + 사고 과정 보기(관전 전용)", ok)


# =====================================
# 5. 해설 에이전트의 게임 기록 문서화와 코드 계산 사실 (LLM/임베딩 없이)
# =====================================
def test_game_qa_docs():
    import json
    import mafia_qa
    ok = True
    log = json.loads((config.DATA_DIR / "demo" / "demo_game.json").read_text(encoding="utf-8"))
    docs = mafia_qa.log_to_docs(log)
    kinds = {d.metadata["kind"] for d in docs}
    ok &= {"role", "speech", "vote", "night", "reason", "thought"} <= kinds
    # "왜 X를 노렸어?"를 찾을 수 있도록 선택 이유에 대상(target)이 붙는가
    kill = log["night_history"][0]["kill"]
    ok &= any(d.metadata["kind"] == "reason" and d.metadata["target"] == kill for d in docs)
    # 코드 계산 사실: 경찰 조사 결과와 득표 수가 실제 기록과 같은가
    inv = log["night_history"][0]["investigation"]
    facts = mafia_qa.player_facts(log, inv["police"])
    ok &= ("조사한 사람 = %s" % inv["target"]) in facts and ("마피아" if inv["is_mafia"] else "마피아 아님") in facts
    ex = log["vote_history"][0]["executed"]
    ok &= ("받은 표 %d" % log["vote_history"][0]["counts"][ex]) in mafia_qa.player_facts(log, ex)
    ok &= "그런 플레이어는 없습니다" in mafia_qa.player_facts(log, "홍길동")
    # 진행 중인 게임에는 해설 에이전트를 만들 수 없다 (비밀 보호)
    try:
        mafia_qa.GameQA.from_state(None, core.new_game(list(config.CHARACTERS), None, 1))
        ok = False
    except ValueError:
        pass
    return check("해설 에이전트: 기록 문서화/이유-대상 연결/코드 계산 사실/종료 전 차단", ok)


if __name__ == "__main__":
    results = [test_police_facts(), test_graph_partner_proposal(), test_consistency_guard(),
               test_speech_fixes_and_trace(), test_game_qa_docs()]
    print("\n%d/%d 통과" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
