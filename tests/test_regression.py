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


if __name__ == "__main__":
    results = [test_police_facts(), test_graph_partner_proposal(), test_consistency_guard()]
    print("\n%d/%d 통과" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
