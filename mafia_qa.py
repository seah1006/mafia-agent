# =====================================
# mafia_qa.py : 질문에 답하는 에이전트 (게임이 끝난 뒤의 해설자)
# =====================================
# 사용자가 질문하면 에이전트가 필요한 지식 저장소를 "스스로 골라" 검색하고, 근거와 출처를 붙여 답한다.
#   - search_rulebook : 룰북 PDF RAG (type=rule 필터)                -> 규칙 질문
#   - search_game_log : 이번 판 기록 RAG (발언/투표/밤 행동/속마음)    -> "왜 그랬어?" 같은 질문
#   - player_summary  : 코드가 계산한 플레이어별 사실(역할, 투표, 밤 행동) -> 정확한 사실 확인
# 게임 기록에는 비밀(밤 행동, 속마음)이 들어 있으므로 게임이 끝난 뒤에만 만든다(정보 분리 원칙).
import dataclasses
import re
import time

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain.agents import create_agent

import config
import rag
import mafia_core
import mafia_agents

GAME_COLLECTION = "mafia_game_log"
PHASE_KO = {"night": "밤", "day": "낮", "vote": "투표", "revote": "재투표"}
# 속마음 노트 중 "[밤 처치 선택] 박준수: 이유" / "[투표] 강태호: 이유" 형태는 누구를 왜 골랐는지에 대한 기록이다
REASON_RE = re.compile(r"^\[(밤 처치 선택|밤 보호 선택|밤 조사 선택|투표|재투표)\] (\S+?): (.*)$", re.S)
REASON_KO = {"밤 처치 선택": "밤에 처치 대상으로", "밤 보호 선택": "밤에 보호 대상으로", "밤 조사 선택": "밤에 조사 대상으로",
             "투표": "투표 대상으로", "재투표": "재투표 대상으로"}
ROLE_WORDS = {"마피아": "mafia", "경찰": "police", "의사": "doctor", "시민": "citizen"}
ALL_WORDS = ("전체", "모두", "모든", "전원", "다들", "all")
WINNER_KO = {"citizen": "시민팀 승리", "mafia": "마피아팀 승리", "draw": "무승부"}


# =====================================
# 1. 게임 기록 -> 문서(Document) + 메타데이터
# =====================================
def state_to_log(state):
    """GameState 를 save_log() 와 같은 형태의 dict 로 바꾼다(파일 저장 없이 바로 질문할 때 사용)."""
    return {
        "players": [dataclasses.asdict(p) for p in state.players],
        "day": state.day, "winner": state.winner, "seed": state.seed,
        "public_log": list(state.public_log), "private_info": dict(state.private_info),
        "notes": dict(state.notes), "night_history": list(state.night_history),
        "vote_history": list(state.vote_history), "stats": dict(state.stats),
    }


def log_to_docs(log):
    """한 판의 기록을 한 줄짜리 문서로 나누고 kind/day/actor/target 메타데이터를 붙인다."""
    role_ko = {p["name"]: config.ROLE_KO[p["role"]] for p in log["players"]}
    docs = []

    def add(text, kind, day, actor="-", target="-"):
        docs.append(Document(page_content=text, metadata={
            "kind": kind, "day": int(day or 0), "actor": actor or "-", "target": target or "-"}))

    for p in log["players"]:
        add("[역할] %s: %s (%s)" % (p["name"], role_ko[p["name"]], "생존" if p["alive"] else "사망"),
            "role", 0, p["name"])
    for e in log["public_log"]:
        d, t, a, tg = e.get("day", 0), e.get("type"), e.get("actor", ""), e.get("target", "")
        if t == "speech":
            add("[%d일차 낮 발언] %s: %s" % (d, a, e["text"]), "speech", d, a)
        elif t in ("vote", "revote"):
            add("[%d일차 %s] %s -> %s" % (d, "재투표" if t == "revote" else "투표", a, tg), "vote", d, a, tg)
        else:
            add("[%d일차 공개] %s" % (d, e.get("text", "")), "public", d, a, tg)
    for nh in log["night_history"]:
        d = nh.get("day", 0)
        parts = ["마피아가 %s 을(를) 노림" % nh.get("kill"), "의사는 %s 보호" % nh.get("protect"),
                 "결과: %s" % ("%s 사망" % nh["killed"] if nh.get("killed") else "아무도 죽지 않음")]
        inv = nh.get("investigation")
        if inv:
            parts.append("경찰 %s 이(가) %s 조사 -> %s" % (
                inv["police"], inv["target"], "마피아" if inv["is_mafia"] else "마피아 아님"))
        add("[%d일차 밤 (게임 후 공개)] " % d + ", ".join(parts), "night", d, nh.get("kill") or "-")
    for name, notes in log["notes"].items():
        for n in notes:
            d = n.get("day", 0)
            m = REASON_RE.match(n.get("thought", "") or "")
            if m:
                # 선택 이유: 대상(target)도 메타데이터로 남겨 "왜 X를 노렸어?"로 찾을 수 있게 한다
                add("[%d일차 %s 이유 (게임 후 공개)] %s(%s)가 %s %s 고른 이유(당시 판단, 대상의 직업은 모르는 상태): %s" % (
                    d, m.group(1), name, role_ko.get(name, "?"), m.group(2), REASON_KO[m.group(1)], m.group(3)),
                    "reason", d, name, m.group(2))
                continue
            sus = sorted((n.get("suspicion") or {}).items(), key=lambda kv: -kv[1])[:3]
            text = "[%d일차 %s 속마음 (게임 후 공개)] %s(%s): %s" % (
                d, PHASE_KO.get(n.get("phase"), n.get("phase", "")), name, role_ko.get(name, "?"),
                n.get("thought", ""))
            if sus:
                text += " | 의심도: " + ", ".join("%s %d" % kv for kv in sus)
            add(text, "thought", d, name)
    return docs


def build_game_db(log):
    """게임 기록을 메모리 Chroma(bge-m3)에 넣는다. 판마다 새 컬렉션을 쓴다."""
    vs = Chroma(collection_name="%s_%s_%d" % (GAME_COLLECTION, log.get("seed"), int(time.time() * 1000)),
                embedding_function=rag._embeddings())
    vs.add_documents(log_to_docs(log))
    return vs


# =====================================
# 2. 코드가 계산하는 사실 (LLM 없이 정확한 사실)
# =====================================
def player_fates(log):
    """이름 -> '1일차 밤에 사망' / '2일차 투표로 처형' (살아남은 사람은 없음)"""
    fate = {}
    for e in log["public_log"]:
        if e.get("type") in ("death", "execution") and e.get("target"):
            fate[e["target"]] = "%d일차 %s" % (e["day"], "밤에 사망" if e["type"] == "death" else "투표로 처형")
    return fate


def game_summary(log):
    """판 전체의 확정 사실: 결과, 팀별 명단과 인원, 날짜별 사망/처형. '마피아는 몇 명/누구' 같은 질문의 근거."""
    role = {p["name"]: p["role"] for p in log["players"]}
    fate = player_fates(log)

    def names(r):
        return [n for n in role if role[n] == r]

    def fmt(ns):
        return ", ".join("%s(%s)" % (n, fate.get(n, "끝까지 생존")) for n in ns) or "없음"

    town = [n for n in role if role[n] != "mafia"]
    lines = ["결과: %s (%s일차에 종료)" % (WINNER_KO.get(log.get("winner"), "진행 중"), log.get("day")),
             "마피아팀 %d명: %s" % (len(names("mafia")), fmt(names("mafia"))),
             "시민팀 %d명: 경찰 %s / 의사 %s / 시민 %d명 %s" % (
                 len(town), fmt(names("police")), fmt(names("doctor")), len(names("citizen")), fmt(names("citizen")))]
    executed = {e["day"]: e["target"] for e in log["public_log"] if e.get("type") == "execution"}
    voted_days = {vh["day"] for vh in log["vote_history"]}
    for nh in log["night_history"]:
        d = nh["day"]
        if nh.get("saved"):
            result = "의사가 살려서 사망자 없음"
        elif nh.get("killed"):
            result = "%s(%s) 사망" % (nh["killed"], config.ROLE_KO[role[nh["killed"]]])
        else:
            result = "사망자 없음"
        acts = ["마피아 처치 대상 %s" % (nh.get("kill") or "없음")]
        if nh.get("protect"):
            acts.append("의사 보호 대상 %s" % nh["protect"])
        inv = nh.get("investigation")
        if inv:
            acts.append("경찰 %s의 조사: %s -> %s" % (inv["police"], inv["target"], "마피아" if inv["is_mafia"] else "마피아 아님"))
        lines.append("%d일차 밤: %s (%s)" % (d, result, " / ".join(acts)))
        if d in executed:
            lines.append("%d일차 투표: %s(%s) 처형" % (d, executed[d], config.ROLE_KO[role[executed[d]]]))
        elif d in voted_days:
            lines.append("%d일차 투표: 처형 없음(동점)" % d)
    # 8B 모델이 기록을 보고 세는 것을 자주 틀리므로 횟수와 승리 이유도 코드가 계산한다
    saves = [nh["day"] for nh in log["night_history"] if nh.get("saved")]
    lines.append("의사가 처치를 막아 살린 횟수: %d번%s" % (len(saves), " (%s)" % ", ".join("%d일차 밤" % d for d in saves) if saves else ""))
    alive = [n for n in role if n not in fate]
    am = [n for n in alive if role[n] == "mafia"]
    at = [n for n in alive if role[n] != "mafia"]
    lines.append("최종 생존자: 마피아 %d명(%s) / 시민팀 %d명(%s)" % (len(am), ", ".join(am) or "없음", len(at), ", ".join(at) or "없음"))
    reason = {"citizen": "마피아가 모두 제거되어 시민팀 승리",
              "mafia": "살아있는 마피아 수(%d)가 시민팀 수(%d) 이상이 되어 마피아팀 승리" % (len(am), len(at)),
              "draw": "제한 일수를 넘겨 무승부"}.get(log.get("winner"))
    if reason:
        lines.append("승부가 난 이유: " + reason)
    return "\n".join(lines)


def player_facts(log, name):
    role = {p["name"]: p["role"] for p in log["players"]}
    if name not in role:
        return "그런 플레이어는 없습니다. 참가자: " + ", ".join(role)
    lines = ["%s: %s" % (name, config.ROLE_KO[role[name]]), player_fates(log).get(name, "끝까지 생존")]
    for vh in log["vote_history"]:
        mine = vh["votes"].get(name)
        got = sorted(v for v, t in vh["votes"].items() if t == name)
        lines.append("%d일차 %s: %s 에게 투표 / 받은 표 %d (%s)" % (
            vh["day"], "재투표" if vh.get("kind") == "revote" else "투표", mine or "-", len(got), ", ".join(got) or "없음"))
    for nh in log["night_history"]:
        if role[name] == "mafia" and nh.get("kill"):
            lines.append("%d일차 밤: 마피아팀(%s 포함)이 처치하려고 고른 사람 = %s" % (nh["day"], name, nh["kill"]))
        if role[name] == "doctor" and nh.get("protect"):
            p = nh["protect"]
            lines.append("%d일차 밤: 의사 %s 이(가) 보호한 사람 = %s%s%s" % (
                nh["day"], name, p, "(자기 자신)" if p == name else "", " -> 처치를 막아 살림" if nh.get("saved") else ""))
        inv = nh.get("investigation")
        if inv and inv.get("police") == name:
            lines.append("%d일차 밤: 경찰 %s 이(가) 조사한 사람 = %s -> 결과: %s" % (
                nh["day"], name, inv["target"], "마피아" if inv["is_mafia"] else "마피아 아님"))
    peak = {}
    for n in log["notes"].get(name, []):
        for k, v in (n.get("suspicion") or {}).items():
            peak[k] = max(peak.get(k, 0), v)
    if peak:
        lines.append("가장 높았던 의심도: " + ", ".join("%s %d" % kv for kv in sorted(peak.items(), key=lambda kv: -kv[1])[:3]))
    return "\n".join(lines)


# =====================================
# 3. 해설 에이전트
# =====================================
SYSTEM = """당신은 방금 끝난 마피아 게임의 해설자입니다. 한국어로만 답하세요.
- 아래 [이번 판 확정 사실]은 코드가 계산한 정답입니다. '누가 마피아였어', '마피아는 몇 명', '누가 이겼어', '언제 죽었어', '요약' 같은
  이번 판 전체 질문은 이 사실대로 답하고, 인원과 명단을 빠짐없이 적으세요(마피아는 여러 명일 수 있습니다).
  룰북의 일반 규칙으로 이번 판 사실을 대신하지 마세요. 질문의 전제가 이 사실과 다르면(예: 이긴 팀에게 '어떻게 졌어?') 먼저 바로잡으세요.
- 그 밖의 내용은 답하기 전에 반드시 도구로 근거를 찾으세요. 규칙 질문은 search_rulebook, 이번 판에서 일어난 일은 search_game_log,
  특정 플레이어의 역할/투표/밤 행동 같은 정확한 사실은 player_summary 를 쓰세요.
- 질문이 여러 부분(예: 이번 판 사실 + 규칙)으로 되어 있으면 부분마다 맞는 도구를 각각 호출하고, 모든 부분에 답하세요.
- 규칙 질문이라도 기억으로 답하지 말고 먼저 search_rulebook 을 호출하세요.
- "왜 ~했어?" 질문은 그 사람의 '고른 이유'나 '속마음' 기록을 근거로 답하세요. 그 사람이 당시 몰랐던 사실(다른 사람의 역할 등)을 이유로 만들지 마세요.
- "왜 처형됐어/왜 들켰어"처럼 다른 사람들의 판단을 묻는 질문은 search_game_log 로 그 사람을 지목한 발언과 투표 이유를 찾으세요.
- 조사 결과, 득표 수, 사망 시점 같은 핵심 사실은 빠뜨리지 마세요. 도구 결과의 이름을 그대로 옮기세요.
- 도구 결과에 없는 내용은 지어내지 말고 "기록에 없습니다"라고 하세요.
- 3~4문장으로 답하고, 마지막 줄에 '근거:' 뒤에 사용한 출처(예: 룰북 p.3, 1일차 투표, 1일차 속마음 정도현)를 적으세요."""


class GameQA:
    """끝난 게임 하나에 대해 질문을 받는 에이전트. ask() 는 답과 도구 사용 기록을 돌려준다."""

    def __init__(self, rule_vs, log, llm=None, printer=print):
        self.rule_vs = rule_vs
        self.log = log
        self.names = [p["name"] for p in log["players"]]
        self.summary = game_summary(log)
        # 판 전체 확정 사실은 항상 보이도록 시스템 프롬프트에 넣는다 (검색에 한 명만 걸려 '마피아 1명'이라고 답하는 문제 방지)
        self.system = SYSTEM + "\n\n[이번 판 확정 사실 (코드 계산)]\n" + self.summary
        self.game_vs = build_game_db(log)
        self.llm = llm or mafia_agents.make_llm(0.0)   # 해설은 사실 위주: 온도 0
        self.printer = printer
        self.calls = []        # 이번 질문에서 에이전트가 호출한 도구
        self.tools = self._make_tools()
        self.agent = create_agent(self.llm, tools=self.tools, system_prompt=self.system)

    @classmethod
    def from_file(cls, rule_vs, path, **kw):
        import json
        from pathlib import Path
        return cls(rule_vs, json.loads(Path(path).read_text(encoding="utf-8")), **kw)

    @classmethod
    def from_state(cls, rule_vs, state, **kw):
        if state.winner is None:
            raise ValueError("게임이 끝난 뒤에만 해설 에이전트를 만들 수 있습니다(비밀 정보 보호).")
        return cls(rule_vs, state_to_log(state), **kw)

    # ----- 3-1. 도구 -----
    def _player(self, text):
        return mafia_core.normalize_choice(text, self.names) if text else None

    def _players(self, text):
        """이름 또는 역할 이름('경찰', '마피아')을 플레이어 목록으로 바꾼다(게임이 끝났으므로 역할 사용 가능)."""
        if not text:
            return []
        t = str(text).strip()
        role = ROLE_WORDS.get(t)
        if role:
            return [p["name"] for p in self.log["players"] if p["role"] == role]
        named = [n for n in self.names if n in t]          # "강태호, 윤서진" 처럼 여러 이름
        if named:
            return named
        compact = t.replace(" ", "")
        if any(w in compact for w in ALL_WORDS):
            return list(self.names)
        if "시민팀" in compact:                            # 시민팀 = 경찰 + 의사 + 시민
            return [p["name"] for p in self.log["players"] if p["role"] != "mafia"]
        for word, r in ROLE_WORDS.items():                  # "마피아들", "마피아팀", "경찰은"
            if word in compact:
                return [p["name"] for p in self.log["players"] if p["role"] == r]
        who = self._player(t)
        return [who] if who else []

    def _search_rulebook(self, question):
        docs = rag.search_rules(self.rule_vs, question, k=3)
        return "\n\n".join("(룰북 p.%d) %s" % (d.metadata["page"] + 1, d.page_content[:400]) for d in docs)

    def _search_game_log(self, query, player=""):
        who = self._players(player)
        flt = {"$or": [{"actor": {"$in": who}}, {"target": {"$in": who}}]} if who else None
        docs = self.game_vs.similarity_search(query, k=6, filter=flt)
        docs.sort(key=lambda d: d.metadata.get("day", 0))   # 날짜순으로 보여 줘야 1일차/2일차를 섞지 않는다
        return "\n".join(d.page_content for d in docs) or "검색 결과 없음"

    def _make_tools(self):
        qa = self

        @tool
        def search_rulebook(question: str) -> str:
            """마피아 게임 룰북(규칙 페이지)을 검색한다. 규칙에 대한 질문일 때 쓴다."""
            qa.calls.append(("search_rulebook", {"question": question}))
            return qa._search_rulebook(question)

        @tool
        def search_game_log(query: str, player: str = "") -> str:
            """이번 판의 기록(발언, 투표, 밤 행동, 속마음)을 검색한다. player 에 이름을 넣으면 그 사람 관련 기록만 찾는다."""
            qa.calls.append(("search_game_log", {"query": query, "player": player}))
            return qa._search_game_log(query, player)

        @tool
        def player_summary(player: str) -> str:
            """플레이어의 역할, 사망 시점, 날짜별 투표, 밤 행동(경찰 조사 결과 포함), 의심도를 코드가 계산해 돌려준다. player 에는 이름 또는 '경찰' 같은 역할 이름."""
            qa.calls.append(("player_summary", {"player": player}))
            who = qa._players(player)
            if not who:
                return player_facts(qa.log, player)
            body = "\n\n".join(player_facts(qa.log, w) for w in who)
            if len(who) > 1:   # 역할/팀으로 물으면 인원과 명단을 먼저 알려 준다
                body = "'%s' 에 해당하는 사람 %d명: %s\n\n%s" % (player, len(who), ", ".join(who), body)
            return body

        @tool
        def game_overview() -> str:
            """이번 판 전체 요약(승자, 팀별 명단과 인원, 날짜별 사망/처형)을 코드가 계산해 돌려준다. '누가 마피아였어', '몇 명', '요약', '누가 이겼어' 질문에 쓴다."""
            qa.calls.append(("game_overview", {}))
            return qa.summary

        return [search_rulebook, search_game_log, player_summary, game_overview]

    # ----- 3-2. 질문 -----
    def _fallback(self, question):
        """에이전트가 도구를 안 쓰고 답하면, 코드가 두 저장소를 모두 검색해 근거를 주고 다시 답하게 한다."""
        ctx = ("[룰북]\n" + self._search_rulebook(question) +
               "\n\n[이번 판 기록]\n" + self._search_game_log(question))
        self.calls.append(("(코드) 룰북 + 게임 기록 모두 검색", {"question": question}))
        msg = [("system", self.system), ("user", "[검색 결과]\n" + ctx + "\n\n[질문] " + question)]
        return self.llm.invoke(msg).content.strip()

    def ask(self, question, show=True):
        """질문 -> 에이전트가 도구 선택/검색 -> 근거와 출처가 붙은 답. {"answer","calls","seconds"} 를 돌려준다."""
        self.calls = []
        t0 = time.time()
        answer = ""
        try:
            res = self.agent.invoke({"messages": [("user", question)]})
            answer = str(res["messages"][-1].content or "").strip()
        except Exception as ex:
            answer = ""
            self.calls.append(("(오류) " + type(ex).__name__, {}))
        if not [c for c in self.calls if not c[0].startswith("(")] or not answer:
            answer = self._fallback(question)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
        out = {"answer": answer, "calls": list(self.calls), "seconds": time.time() - t0}
        if show and self.printer:
            self.printer(format_answer(question, out))
        return out


def format_answer(question, out):
    lines = ["[질문] " + question, "    ┌─ 에이전트가 고른 도구"]
    for name, args in out["calls"]:
        arg = ", ".join("%s=%r" % (k, v) for k, v in args.items() if v)
        lines.append("    │ %s(%s)" % (name, arg))
    lines.append("    └─ %.1f초" % out["seconds"])
    lines.append("[답변] " + out["answer"])
    return "\n".join(lines)
