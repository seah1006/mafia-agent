# =====================================
# tests/test_agents.py : RAG + LLM 플레이어 실제 호출 테스트
# 실행: python tests/test_agents.py   (Ollama 실행 중이어야 함)
# =====================================
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import rag
import mafia_core
import mafia_agents
from mafia_agents import LLMPlayer

RESULTS = []
LAT = {}


def check(label, ok, extra=""):
    RESULTS.append(bool(ok))
    print(("[PASS] " if ok else "[FAIL] ") + label + (" | " + str(extra) if extra else ""))


def timed(key, fn):
    t = time.time()
    r = fn()
    LAT.setdefault(key, []).append(round(time.time() - t, 1))
    return r


# =====================================
# 1. DB 생성 + 필터 검색
# =====================================
def test_rag():
    print("\n== 1. RAG ==")
    vs = timed("db_build", lambda: rag.build_db())
    check("청크 수 > 16", len(vs.get()["ids"]) > 16, len(vs.get()["ids"]))
    for role in config.ROLES:
        docs = rag.search_strategy(vs, role, "토론 전략")
        check("전략 필터 role=%s" % role, docs and all(
            d.metadata["type"] == "strategy" and d.metadata["role"] == role for d in docs))
    rules = rag.search_rules(vs, "투표 동점이면?")
    check("규칙 필터 type=rule", rules and all(d.metadata["type"] == "rule" for d in rules))
    for name in config.CHARACTERS:
        p = rag.get_persona(vs, name)
        check("페르소나 %s" % name, name in p.splitlines()[0], len(p))
    # 제목 검증 오류 메시지 확인(잘못된 PAGE_META 를 임시 주입)
    saved = config.PAGE_META[1]["title"]
    config.PAGE_META[1]["title"] = "엉뚱한 제목"
    try:
        rag.load_pdf_with_meta()
        check("제목 불일치 시 오류", False)
    except ValueError as e:
        check("제목 불일치 시 한국어 오류", "제목" in str(e), str(e)[:60])
    finally:
        config.PAGE_META[1]["title"] = saved
    return vs


# =====================================
# 2. 시나리오 (코어로 게임 상태 구성)
# =====================================
def make_scenario(vs):
    state = mafia_core.new_game(list(config.CHARACTERS), None, 3)
    by_role = {}
    for p in state.players:
        by_role.setdefault(p.role, []).append(p.name)
    state.day, state.phase = 1, "day"
    return state, by_role


def run_action(state, name, action, vs, llm=None, use_tools=True):
    role = next(p.role for p in state.players if p.name == name)
    view = mafia_core.build_view(state, name, action)
    pl = LLMPlayer(name, role, vs, llm=llm, use_tools=use_tools, state=state)
    fn = {"kill": pl.night_action, "protect": pl.night_action, "investigate": pl.night_action,
          "vote": pl.vote, "speak": pl.speak}[action]
    res = timed(action, lambda: fn(view))
    return pl, view, res


def test_llm(vs):
    print("\n== 2. 실제 LLM 호출 ==")
    state, br = make_scenario(vs)
    for action, role in [("kill", "mafia"), ("investigate", "police"), ("protect", "doctor")]:
        name = br[role][0]
        pl, view, res = run_action(state, name, action, vs)
        check("%s 행동(%s)" % (role, action), res is not None and res.target in view.candidates,
              "%s -> %s [%s] %ss" % (name, getattr(res, "target", None), pl.last_source, LAT[action][-1]))
    voter = br["citizen"][0]
    pl, view, res = run_action(state, voter, "vote", vs)
    check("투표", res is not None and res.target in view.candidates,
          "%s -> %s [%s] %ss" % (voter, getattr(res, "target", None), pl.last_source, LAT["vote"][-1]))

    print("\n-- 발언 + 정체 노출 검사 --")
    leaks = 0
    for role in ["mafia", "police", "citizen"]:
        name = br[role][0]
        pl, view, sp = run_action(state, name, "speak", vs)
        leak = mafia_core.contains_role_leak(sp.text, role) if role == "mafia" else False
        leaks += leak
        print("  [%s/%s] %s" % (role, name, sp.text))
        check("발언 %s 길이/한국어/노출없음" % role,
              0 < len(sp.text) <= config.SPEECH_MAX_CHARS and not leak,
              "%d자 %ss 의심도 %d명" % (len(sp.text), LAT["speak"][-1], len(sp.suspicion)))
    check("정체 노출 0건", leaks == 0)
    return state, br


# =====================================
# 3. 강제 실패 -> fallback
# =====================================
class BrokenLLM:
    """항상 예외를 내는 LLM. 에이전트/구조화 출력이 모두 실패해야 None 이 나온다."""

    def bind_tools(self, *a, **k):
        return self

    def with_structured_output(self, *a, **k):
        raise RuntimeError("강제 실패")

    def invoke(self, *a, **k):
        raise RuntimeError("강제 실패")


def test_fallback(vs, state, br):
    print("\n== 3. 강제 실패 폴백 ==")
    name = br["doctor"][0]
    view = mafia_core.build_view(state, name, "protect")
    pl = LLMPlayer(name, "doctor", vs, llm=BrokenLLM(), state=state)
    res = pl.night_action(view)
    check("LLM 전부 실패 시 None 반환", res is None, pl.last_source)
    value, used_fb = mafia_core.safe_decide(pl.night_action, view, view.candidates, state, "protect")
    check("코어 safe_decide 가 무작위 대체", used_fb and (getattr(value, "target", value) in view.candidates),
          "fallbacks=%s" % state.stats.get("fallbacks"))
    # 에이전트가 도구를 안 쓰는 경우(use_tools=False) -> 구조화 출력 경로
    pl2, view2, res2 = run_action(state, br["doctor"][0], "protect", vs, use_tools=False)
    check("도구 없이 구조화 출력 경로", res2 is not None and pl2.last_source == "structured", pl2.last_source)


# =====================================
# 4. 사회자
# =====================================
def test_moderator(vs):
    print("\n== 4. 사회자 ==")
    mod = mafia_agents.Moderator(vs)
    ans = timed("rule_qa", lambda: mod.explain_rule("투표가 동점이면 어떻게 되나요?"))
    print("  " + ans.replace("\n", "\n  "))
    check("룰 답변에 출처 표기", "(룰북 p." in ans, "%ss" % LAT["rule_qa"][-1])
    check("narrate 템플릿", "처형" in mod.narrate({"type": "execution", "target": "A", "text": ""}))


if __name__ == "__main__":
    vs = test_rag()
    state, br = test_llm(vs)
    test_fallback(vs, state, br)
    test_moderator(vs)
    print("\n== 요약 ==")
    print("통과 %d / %d" % (sum(RESULTS), len(RESULTS)))
    print("호출 지연(초):", LAT)
    print("LLM 호출 수:", mafia_agents.LLM_CALLS["count"], "| state.stats:", state.stats)
    sys.exit(0 if all(RESULTS) else 1)
