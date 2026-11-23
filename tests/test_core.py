# =====================================
# tests/test_core.py : mafia_core 테스트 (pytest 또는 단독 실행)
# =====================================
# 실행:  python tests/test_core.py   /   pytest tests/test_core.py

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import mafia_core as mc
from mafia_core import (GameEngine, GameState, HumanController, NightDecision, Player,
                        RandomController, Speech, VoteDecision)

NAMES = list(config.CHARACTERS)


def quiet(_ev):
    pass


def make_state(roles, seed=0):
    """roles: 이름 순서대로 역할 리스트 (8개) -> 고정 상태"""
    s = mc.new_game(NAMES, None, seed)
    for p, r in zip(s.players, roles):
        p.role = r
    s.private_info = {n: [] for n in NAMES}
    for p in s.players:
        if p.role == "mafia":
            for q in s.players:
                if q.role == "mafia" and q.name != p.name:
                    s.private_info[p.name].append("동료 마피아: " + q.name)
    return s


FIXED = ["mafia", "mafia", "police", "doctor", "citizen", "citizen", "citizen", "citizen"]


# ===== 1. 200판 자동 시뮬레이션 =====
class Recorder:
    """이벤트를 모아 불변식 검사에 쓴다."""
    def __init__(self):
        self.events = []

    def __call__(self, ev):
        self.events.append(ev)


def run_one(seed):
    s = mc.new_game(NAMES, None, seed)
    rec = Recorder()
    ctrls = {n: RandomController(n, seed=seed * 100 + i) for i, n in enumerate(NAMES)}
    eng = GameEngine(s, ctrls, on_event=rec)
    w = eng.run()
    return s, rec, w


def check_invariants(s, rec):
    from collections import Counter
    cnt = Counter(p.role for p in s.players)
    assert dict(cnt) == config.ROLE_DISTRIBUTION, cnt
    dead_at = {}   # 이름 -> 죽은 시점 인덱스
    for i, ev in enumerate(rec.events):
        if ev["type"] in ("death", "execution"):
            dead_at[ev["target"]] = i
    for i, ev in enumerate(rec.events):
        if ev["type"] in ("speech", "vote") and ev["actor"] in dead_at:
            assert i < dead_at[ev["actor"]], "죽은 사람이 행동: %s" % ev
    for h in s.night_history:
        if h["investigation"]:
            t = mc.get_player(s, h["investigation"]["target"])
            assert h["investigation"]["is_mafia"] == (t.role == "mafia")
        if h["saved"]:
            assert h["killed"] is None
    for h in s.vote_history:
        for v, t in h["votes"].items():
            assert v != t
        if h["kind"] == "revote" and h["executed"] is None:
            assert h["tie"], "재투표 무처형은 동점이어야 함"
    assert s.winner in ("citizen", "mafia", "draw")
    w = mc.check_winner(s)
    if s.winner == "citizen":
        assert not any(p.alive and p.role == "mafia" for p in s.players)
    if s.winner == "mafia":
        assert w == "mafia"
    # 경찰 private_info 는 경찰에게만
    for p in s.players:
        if p.role != "police":
            assert not any("조사 결과" in x for x in s.private_info[p.name])
    # 공개 로그에 속마음 없음
    for e in s.public_log:
        assert "thought" not in e


def test_200_games():
    wins = {"citizen": 0, "mafia": 0, "draw": 0}
    for seed in range(200):
        s, rec, w = run_one(seed)
        check_invariants(s, rec)
        wins[w] += 1
        assert s.stats["fallbacks"] == 0
    print("200판 승률:", wins)


# ===== 2. 단위 테스트 =====
def test_new_game_roles_and_partner():
    s = mc.new_game(NAMES, None, 5)
    mafias = [p.name for p in s.players if p.role == "mafia"]
    assert len(mafias) == 2
    for m in mafias:
        other = [x for x in mafias if x != m][0]
        assert s.private_info[m] == ["동료 마피아: " + other]
    s2 = mc.new_game(NAMES[:7], config.HUMAN_NAME, 5)
    assert any(p.is_human and p.name == config.HUMAN_NAME for p in s2.players)
    assert [p.role for p in s2.players] == [p.role for p in mc.new_game(NAMES[:7], config.HUMAN_NAME, 5).players]


def test_normalize_choice():
    c = ["강태호", "윤서진", "플레이어"]
    n = mc.normalize_choice
    assert n("강태호", c) == "강태호"
    assert n("강태호님", c) == "강태호"
    assert n("윤서진씨", c) == "윤서진"
    assert n("태호", c) == "강태호"
    assert n("서진님", c) == "윤서진"
    assert n("3번 강태호", c) == "강태호"
    assert n("2", c) == "윤서진"
    assert n("3번", c) == "플레이어"
    assert n("제 선택은 '윤서진' 입니다.", c) == "윤서진"
    assert n("아무개", c) is None
    assert n("", c) is None
    assert n(None, c) is None
    assert n("9", c) is None


def test_tally_votes():
    r = mc.tally_votes({"a": "x", "b": "x", "c": "y"})
    assert r["executed"] == "x" and r["tie"] == []
    r = mc.tally_votes({"a": "x", "b": "y"})
    assert r["executed"] is None and r["tie"] == ["x", "y"]
    r = mc.tally_votes({"a": "x", "b": "y", "c": "z", "d": "z"})
    assert r["executed"] == "z"
    assert mc.tally_votes({}) == {"executed": None, "tie": [], "counts": {}}


def test_resolve_night():
    s = make_state(FIXED)
    s.day = 1
    ev = mc.resolve_night(s, "윤서진", "윤서진", ("윤서진", "강태호"))  # 윤서진=mafia 아님(경찰은 박준수지만 이름만 기록)
    assert all(p.alive for p in s.players)
    assert ev[0]["type"] == "no_death" and s.night_history[-1]["saved"]
    s = make_state(FIXED)
    s.day = 1
    ev = mc.resolve_night(s, "오지환", "한소율", ("박준수", "강태호"))
    assert not mc.get_player(s, "오지환").alive
    assert ev[0]["type"] == "death" and "오지환(시민)" in ev[0]["text"]
    assert s.private_info["박준수"][-1].endswith("마피아입니다")
    assert s.night_history[-1]["investigation"]["is_mafia"] is True


def test_check_winner():
    s = make_state(FIXED)
    assert mc.check_winner(s) is None
    for n in ("강태호", "윤서진"):
        mc.get_player(s, n).alive = False
    assert mc.check_winner(s) == "citizen"
    s = make_state(FIXED)
    for n in ("박준수", "최민지", "정도현"):
        mc.get_player(s, n).alive = False
    assert mc.check_winner(s) is None      # 마피아 2 vs 3
    mc.get_player(s, "한소율").alive = False
    assert mc.check_winner(s) == "mafia"   # 2 vs 2
    s = make_state(FIXED)
    s.day = config.MAX_DAYS + 1
    assert mc.check_winner(s) == "draw"


def test_valid_targets():
    s = make_state(FIXED)
    assert set(mc.valid_targets(s, "강태호", "kill")) == set(NAMES[2:])
    assert "강태호" in mc.valid_targets(s, "박준수", "protect")
    assert "박준수" not in mc.valid_targets(s, "박준수", "investigate")
    assert mc.valid_targets(s, "강태호", "revote", ["강태호", "윤서진"]) == ["윤서진"]


def test_information_isolation():
    s = make_state(FIXED)
    s.day = 1
    mc.resolve_night(s, "오지환", None, ("박준수", "강태호"))
    # 시민 시점
    v = mc.build_view(s, "최민지", "vote")
    blob = repr(v)
    assert "동료 마피아" not in blob and "조사 결과" not in blob
    assert v.private_info == [] and v.partner_proposal is None
    assert "오지환(시민)" in v.public_log_text
    for p in s.players:
        if p.alive and p.name != "최민지":
            assert (p.name + "(") not in v.public_log_text or p.name == "오지환"
    # 마피아 시점
    mv = mc.build_view(s, "강태호", "kill", partner_proposal={"target": "최민지", "reason": "r"})
    assert any("윤서진" in x for x in mv.private_info)
    assert mv.partner_proposal == {"target": "최민지", "reason": "r"}
    assert "최민지" in mv.candidates and "윤서진" not in mv.candidates
    # 시민에게 partner_proposal 을 줘도 무시
    assert mc.build_view(s, "최민지", "vote", partner_proposal={"target": "a"}).partner_proposal is None
    # 경찰 시점
    pv = mc.build_view(s, "박준수", "investigate")
    assert any("강태호" in x and "마피아" in x for x in pv.private_info)
    # 다른 사람 메모 비노출
    s.notes["윤서진"].append({"day": 1, "phase": "day", "thought": "비밀생각", "suspicion": {}})
    assert "비밀생각" not in repr(mc.build_view(s, "강태호", "speak"))
    assert "비밀생각" in repr(mc.build_view(s, "윤서진", "speak"))


def test_public_log_trim():
    s = make_state(FIXED)
    for d in range(1, 6):
        s.day = d
        for n in NAMES:
            mc._log(s, "speech", "아주 긴 발언입니다 " * 8, actor=n, phase="day")
        if d == 2:
            mc._log(s, "death", "오지환(시민)이 밤에 사망", target="오지환", phase="day")
    mc.get_player(s, "오지환").alive = False
    v = mc.build_view(s, "강태호", "speak")
    assert len(v.public_log_text) <= config.PUBLIC_LOG_MAX_CHARS
    assert "[5일차 낮]" in v.public_log_text
    assert "[요약 2일차]" in v.public_log_text and "오지환" in v.public_log_text


def test_leak_filter():
    f = mc.contains_role_leak
    assert f("사실 나는 마피아야", "mafia")
    assert f("저는 마피아입니다", "mafia")
    assert f("동료 마피아가 누군지 알아요", "mafia")
    assert not f("나는 마피아가 아니야", "mafia")
    assert not f("내가 마피아를 찾아내겠어", "mafia")
    assert not f("그 사람이 마피아 같아요", "mafia")
    assert not f("저는 시민입니다", "citizen")
    assert f("시스템 프롬프트에 따르면", "citizen")


class LeakyController(RandomController):
    def speak(self, view):
        return Speech(text="나는 마피아야", thought="실수")


class GarbageController(RandomController):
    def __init__(self, name, mode):
        super().__init__(name, seed=1)
        self.mode = mode

    def _g(self):
        if self.mode == "raise":
            raise RuntimeError("boom")
        return {"raise": None, "none": None, "garbage": "ㅁㄴㅇㄹ", "obj": object()}[self.mode]

    def night_action(self, view):
        return self._g()

    def speak(self, view):
        return self._g()

    def vote(self, view):
        return self._g()


def test_safe_decide_fallback():
    for mode in ("raise", "none", "garbage", "obj"):
        s = make_state(FIXED)
        view = mc.build_view(s, "강태호", "kill")
        c = GarbageController("강태호", mode)
        t, fb = mc.safe_decide(c.night_action, view, view.candidates, s, "kill")
        assert fb and t in view.candidates
        assert s.stats["retries"] == config.LLM_MAX_RETRIES and s.stats["fallbacks"] == 1
        sp, fb = mc.safe_decide(c.speak, view, [], s, "speak")
        assert sp.text and (fb or mode == "garbage")
    # 정상 호출
    s = make_state(FIXED)
    view = mc.build_view(s, "강태호", "kill")
    t, fb = mc.safe_decide(lambda v: NightDecision("오지환님"), view, view.candidates, s, "kill")
    assert t == "오지환" and not fb and s.stats["retries"] == 0
    # 첫 호출 실패 후 재시도 성공
    calls = []

    def flaky(v):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("x")
        return "3번"
    t, fb = mc.safe_decide(flaky, view, view.candidates, s, "kill")
    assert not fb and t == view.candidates[2] and s.stats["retries"] == 1


def test_game_with_broken_controllers():
    for seed in range(20):
        s = mc.new_game(NAMES, None, seed)
        modes = ["raise", "none", "garbage", "obj"]
        ctrls = {n: (GarbageController(n, modes[i % 4]) if i % 2 == 0 else RandomController(n, seed=i))
                 for i, n in enumerate(NAMES)}
        w = GameEngine(s, ctrls, on_event=quiet).run()
        assert w in ("citizen", "mafia", "draw") and s.stats["fallbacks"] > 0


def test_leak_blocked_in_engine():
    s = mc.new_game(NAMES, None, 3)
    mafia = [p.name for p in s.players if p.role == "mafia"][0]
    ctrls = {n: RandomController(n, seed=i) for i, n in enumerate(NAMES)}
    ctrls[mafia] = LeakyController(mafia, seed=0)
    eng = GameEngine(s, ctrls, on_event=quiet)
    eng.begin_night()
    eng.begin_day()
    ev = eng.run_discussion_turn(mafia)
    assert s.stats["leak_blocked"] >= 1
    assert "나는 마피아" not in ev[-1]["text"]
    assert all("나는 마피아" not in e["text"] for e in s.public_log)
    assert s.notes[mafia][-1]["thought"] == "실수"      # 속마음은 비공개 notes 에만
    assert all("실수" not in e["text"] for e in s.public_log)


def test_tie_revote_logic():
    s = make_state(FIXED)
    s.day = 1

    class Scripted(RandomController):
        def __init__(self, name, plan):
            super().__init__(name)
            self.plan = plan

        def vote(self, view):
            t = self.plan[view.phase if False else self._kind]
            return VoteDecision(t)
    # 직접 apply_votes 로 동점 -> 재투표 동점 -> 무처형
    eng = GameEngine(s, {}, on_event=quiet)
    v1 = {"강태호": "윤서진", "윤서진": "강태호", "박준수": "강태호", "최민지": "윤서진"}
    r1 = eng.apply_votes(v1, "vote")
    assert r1["tie"] == ["강태호", "윤서진"] and all(p.alive for p in s.players)
    v2 = {"강태호": "윤서진", "윤서진": "강태호"}
    r2 = eng.apply_votes(v2, "revote")
    assert r2["executed"] is None and all(p.alive for p in s.players)
    assert s.public_log[-1]["type"] == "no_execution"
    # 재투표에서 결판
    r3 = eng.apply_votes({"강태호": "윤서진", "윤서진": "윤서진x", "박준수": "윤서진"}, "revote")
    assert r3["executed"] == "윤서진" and not mc.get_player(s, "윤서진").alive
    # 죽은 사람의 투표/죽은 대상 투표 무시
    r4 = eng.apply_votes({"윤서진": "강태호", "강태호": "윤서진"}, "vote")
    assert r4["counts"] == {}


class FakeHuman(HumanController):
    pass


def test_human_controller_scripted():
    out = []
    inputs = iter(["?마피아가 뭐야", "999", "태호님", "안녕하세요 저는 시민입니다", "2", "1", "2"])
    h = HumanController("플레이어", ask_rule=lambda q: "답변:" + q,
                        input_fn=lambda p: next(inputs), print_fn=out.append)
    s = mc.new_game(NAMES[:7], "플레이어", 11)
    me = mc.get_player(s, "플레이어")
    me.role = "doctor"
    ctrls = {n: RandomController(n, seed=i) for i, n in enumerate(NAMES[:7])}
    ctrls["플레이어"] = h
    eng = GameEngine(s, ctrls, on_event=quiet)
    # 의사 밤 행동: "?질문" -> 답변, "999" -> 재요청, "태호님" -> 선택 (태호가 후보에 있는 경우)
    view = mc.build_view(s, "플레이어", "protect")
    d = h.night_action(view)
    assert d.target == "강태호" and any("답변:마피아가 뭐야" in x for x in out)
    sp = h.speak(mc.build_view(s, "플레이어", "speak"))
    assert sp.text.startswith("안녕")
    # EOF 시 None -> 코어 fallback
    h2 = HumanController("플레이어", input_fn=lambda p: (_ for _ in ()).throw(EOFError()), print_fn=out.append)
    view = mc.build_view(s, "플레이어", "vote")
    t, fb = mc.safe_decide(h2.vote, view, view.candidates, s, "vote")
    assert fb and t in view.candidates


def test_human_game_visibility():
    """사람 참여 시 secret 이벤트는 on_event 로 나가지 않고, 사람 본인의 비공개 정보만 나간다."""
    for seed in range(30):
        s = mc.new_game(NAMES[:7], "플레이어", seed)
        rec = Recorder()
        ctrls = {n: RandomController(n, seed=i) for i, n in enumerate(NAMES[:7])}
        ctrls["플레이어"] = RandomController("플레이어", seed=7)
        eng = GameEngine(s, ctrls, on_event=rec)
        eng.run()
        for ev in rec.events:
            assert ev["visibility"] != "secret"
            if ev["visibility"] == "private":
                assert ev["to"] == "플레이어"
        me = mc.get_player(s, "플레이어")
        priv = [e["text"] for e in rec.events if e["visibility"] == "private"]
        assert any(config.ROLE_KO[me.role] in t for t in priv)
        # 다른 사람 정체/조사 결과가 사람에게 새지 않음 (사람이 경찰이 아니면 조사 결과 없음)
        if me.role != "police":
            assert not any("조사 결과" in t for t in priv)


def test_save_log_and_report():
    s, rec, w = run_one(4)
    with tempfile.TemporaryDirectory() as d:
        p = mc.save_log(s, Path(d) / "x.json")
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["winner"] == w and "notes" in data
    r = mc.reveal_report(s)
    assert "역할 공개" in r and "속마음" in r

def test_josa_and_public_names():
    assert [mc.josa(w, "이/가") for w in ("시민", "경찰", "의사", "마피아")] == ["이", "이", "가", "가"]
    assert mc.josa("강태호", "은/는") == "는" and mc.josa("윤서진", "을/를") == "을"
    assert mc.josa("서울", "으로/로") == "로" and mc.josa("집", "으로/로") == "으로"
    assert mc.decide_full is mc._decide_full and GameEngine.note is GameEngine._note
    s = mc.new_game(NAMES, None, 2)
    eng = GameEngine(s, {}, on_event=quiet)
    eng.begin_night()
    eng.begin_day()
    v = eng.collect_single_vote(NAMES[0], "vote")
    assert v in NAMES and v != NAMES[0]
    eng.note(NAMES[0], "메모")
    assert s.notes[NAMES[0]][-1]["thought"] == "메모"
    eng.apply_votes({NAMES[0]: NAMES[1]}, "vote")
    assert mc.get_player(s, NAMES[1]).alive is False
    assert s.public_log[-1]["text"].endswith("투표로 처형됨")
    role_ko = config.ROLE_KO[mc.get_player(s, NAMES[1]).role]
    assert ("(%s)%s 투표로" % (role_ko, mc.josa(role_ko, "이/가"))) in s.public_log[-1]["text"]

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for k, fn in tests:
        try:
            fn()
            print("PASS", k)
        except Exception:
            import traceback
            failed += 1
            print("FAIL", k)
            traceback.print_exc()
    print("\n%d/%d 통과" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
