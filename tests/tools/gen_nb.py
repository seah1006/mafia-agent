import json
import sys
from pathlib import Path

# 사용법: python tests/tools/gen_nb.py [출력 폴더]  (기본값: 프로젝트 폴더)
# 노트북 4개(01~04)를 다시 생성한다. 노트북 내용을 고칠 때는 이 파일을 수정한다.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def save(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "venv (3.11.9)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = "cell%03d" % i
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


CWD_CELL = '''
# ===== 0. 작업 폴더 맞추기 =====
# 이 노트북은 08_Mafia_Agent 폴더에서 실행한다고 가정한다.
# (VS Code에서 상위 폴더를 열었을 때도 동작하도록 폴더를 찾아 이동한다.)
import os
import sys
from pathlib import Path

if not Path("config.py").exists():
    if Path("08_Mafia_Agent/config.py").exists():
        os.chdir("08_Mafia_Agent")
    else:
        raise FileNotFoundError("08_Mafia_Agent 폴더에서 노트북을 열어주세요.")

sys.path.insert(0, os.getcwd())

import config

print("===== 작업 폴더 =====")
print(config.BASE_DIR)
print("LLM:", config.LLM_MODEL, "/ 임베딩:", config.EMBED_MODEL)
'''

# =====================================================
# 01_create_db
# =====================================================
nb1 = [
    md('''
# 01. 룰북 PDF -> 벡터 DB 만들기 (RAG)

**목적**: 마피아 게임 AI 플레이어가 참고할 지식을 PDF 한 권으로 만들고, 검색 가능한 벡터 DB(Chroma)로 구축한다.

PDF 16쪽의 구성은 다음과 같다.

| 쪽 | 내용 | type | role | name |
|---|---|---|---|---|
| 1~4 | 룰북 (규칙, 직업 능력, 진행 순서) | rule | all | - |
| 5~8 | 직업별 전략 가이드 | strategy | mafia / police / doctor / citizen | - |
| 9~16 | 캐릭터 프로필 8명 | character | all | 캐릭터 이름 |

**핵심 아이디어: 페이지마다 메타데이터(type, role, name)를 붙인다.**
그러면 "마피아 AI는 마피아 전략 페이지만", "강태호 AI는 강태호 프로필만" 처럼 검색 범위를 **필터로 제한**할 수 있다.
유사도 검색만 쓰면 시민 AI가 마피아 전략 문서를 읽어버리는 정보 유출이 생길 수 있다.

진행 순서: PDF 로딩 -> 메타데이터 확인 -> 청킹 -> 벡터 DB 구축 -> 필터 검색 시연
'''),
    code(CWD_CELL),
    md('''
## 1. PDF 로딩

`PyPDFLoader`는 PDF를 **페이지 한 장 = Document 하나**로 읽는다. 각 Document의 `metadata["page"]`는 0부터 시작한다.
'''),
    code('''
# ===== 1. PDF 로딩 =====
from langchain_community.document_loaders import PyPDFLoader

raw_docs = PyPDFLoader(str(config.PDF_PATH)).load()

print("===== PDF 로딩 완료 =====")
print("PDF 경로:", config.PDF_PATH)
print("PDF 페이지 수:", len(raw_docs))
print("1쪽 일부:")
print(raw_docs[0].page_content[:150])
print("원래 metadata:", raw_docs[0].metadata)
'''),
    md('''
## 2. 페이지 메타데이터 부착

`rag.load_pdf_with_meta()`는 `config.PAGE_META`의 type / role / name / title을 각 페이지에 붙인다.
이때 **PDF 각 페이지의 첫 줄이 약속된 제목과 같은지 검증**해서, PDF가 바뀌어 메타데이터가 어긋나는 사고를 막는다.
'''),
    code('''
# ===== 2. 메타데이터 부착 + 표로 확인 =====
import rag

docs = rag.load_pdf_with_meta()

print("%-4s %-10s %-8s %-8s %s" % ("쪽", "type", "role", "name", "제목"))
print("-" * 60)
for d in docs:
    m = d.metadata
    print("%-4d %-10s %-8s %-8s %s" % (m["page"] + 1, m["type"], m["role"], m["name"], m["title"]))
'''),
    md('''
## 3. 청킹 (Chunking)

페이지가 길면 한 덩어리로 검색하기 부정확하므로 `RecursiveCharacterTextSplitter`로 **500자 단위(겹침 50자)**로 자른다.
쪼개진 chunk도 원래 페이지의 메타데이터를 그대로 물려받는다. (필터가 chunk에서도 동작해야 하므로 중요하다.)
'''),
    code('''
# ===== 3. 청킹 =====
from collections import Counter

chunks = rag.split_docs(docs, chunk_size=500, chunk_overlap=50)

print("===== 청킹 결과 =====")
print("페이지 수:", len(docs), "-> chunk 수:", len(chunks))

# type 별 chunk 수
by_type = Counter(c.metadata["type"] for c in chunks)
for t, n in by_type.items():
    print("  %-10s %d개" % (t, n))

# 쪽별 chunk 수 표
per_page = Counter(c.metadata["page"] + 1 for c in chunks)
print("\\n쪽별 chunk 수:", dict(sorted(per_page.items())))

# 미리보기: 마피아 전략 가이드의 첫 chunk
preview = [c for c in chunks if c.metadata["type"] == "strategy" and c.metadata["role"] == "mafia"][0]
print("\\n[미리보기] metadata:", preview.metadata)
print(preview.page_content[:200])
'''),
    md('''
## 4. 벡터 DB 구축 (Chroma + bge-m3 임베딩)

chunk 텍스트를 `bge-m3`(Ollama) 임베딩으로 벡터로 바꿔 Chroma에 저장한다. **Ollama 앱이 켜져 있고 `bge-m3`가 설치되어 있어야 한다.**
`rag.build_db()`는 기존 DB를 지우고 새로 만든다 (`reset=True`).
'''),
    code('''
# ===== 4. 벡터 DB 구축 =====
vs = rag.build_db()

print("===== 벡터 DB 구축 완료 =====")
print("저장 위치:", config.DB_DIR)
print("저장된 Chunk 수:", vs._collection.count())
'''),
    md('''
## 5. 메타데이터 필터 검색 시연

같은 DB에서 **필터 조건만 바꿔** 3가지 용도로 검색한다.

| 용도 | 함수 | 필터 |
|---|---|---|
| 사회자가 규칙 질문에 답할 때 | `search_rules` | type = rule |
| AI가 자기 역할의 전략 조언을 볼 때 | `search_strategy` | type = strategy AND role = 내 역할 |
| AI가 자기 캐릭터 설정을 읽을 때 | `get_persona` | type = character AND name = 내 이름 |
'''),
    code('''
# ===== 5-1. 규칙 검색 (type = rule) =====
question = "의사가 자기 자신을 살릴 수 있나요?"

print("질문:", question)
for i, d in enumerate(rag.search_rules(vs, question, k=3), 1):
    print("\\n[%d] 쪽 %d | %s | type=%s" % (i, d.metadata["page"] + 1, d.metadata["title"], d.metadata["type"]))
    print(d.page_content[:150].replace("\\n", " "))
'''),
    code('''
# ===== 5-2. 전략 검색: 같은 질문, 역할만 다르게 (type = strategy AND role) =====
question = "초반 토론에서 어떻게 행동해야 하나요?"

for role in ["mafia", "citizen"]:
    print("\\n##### role=%s (%s) #####" % (role, config.ROLE_KO[role]))
    for d in rag.search_strategy(vs, role, question, k=2):
        print("- 쪽 %d | %s | role=%s" % (d.metadata["page"] + 1, d.metadata["title"], d.metadata["role"]))
        print("  ", d.page_content[:100].replace("\\n", " "))
'''),
    code('''
# ===== 5-3. 필터가 없으면 어떻게 될까? (비교) =====
question = "초반 토론에서 어떻게 행동해야 하나요?"

print("[필터 없음] 전체 문서에서 유사도 검색")
for d in vs.similarity_search(question, k=4):
    print("- 쪽 %d | %s | type=%s role=%s" % (d.metadata["page"] + 1, d.metadata["title"], d.metadata["type"], d.metadata["role"]))
print("\\n-> 필터가 없으면 다른 역할의 전략/캐릭터 문서가 섞여 들어올 수 있다.")
'''),
    code('''
# ===== 5-4. 캐릭터 페르소나 (type = character AND name) =====
for name in ["강태호", "윤서진"]:
    print("\\n##### %s #####" % name)
    print(rag.get_persona(vs, name)[:250])
'''),
    code('''
print("===== 01 완료: 이제 02_agent_test.ipynb 로 이동하세요 =====")
'''),
]

# =====================================================
# 02_agent_test
# =====================================================
nb2 = [
    md('''
# 02. AI 플레이어 1명 단위 테스트

**목적**: 전체 게임을 돌리기 전에 AI 플레이어 한 명씩 제대로 동작하는지 확인한다.

확인 항목
1. 역할별로 `LLMPlayer`를 하나씩 만든다 (마피아 / 경찰 / 의사 / 시민)
2. **정보 분리**: AI에게 전달되는 `PlayerView`에는 자기가 알아도 되는 정보만 들어있다
3. 발언 1회, 투표 1회, 밤 행동(처치/보호/조사)
4. **정체 노출 테스트**: 마피아가 "나 마피아야"라고 말하면 코드가 막는다
5. **fallback**: LLM이 엉뚱한 출력을 해도 게임이 멈추지 않는다

설계 원칙: *LLM은 연기를 하고, 진실(규칙/정보 접근권한)은 코드가 지킨다.*
'''),
    code(CWD_CELL),
    code('''
# ===== 1. 모듈 불러오기 =====
import rag
import mafia_core as core
import mafia_agents as agents

vs = rag.load_db()
print("===== 벡터 DB 로드 완료 =====")
print("Chunk 수:", vs._collection.count())
'''),
    md('''
## 1. 게임 하나 만들고 역할별 AI 만들기

`new_game()`이 역할을 무작위로 배정한다 (seed를 주면 같은 결과). 이 상태(`GameState`)는 **진실의 원천**이며 AI에게 직접 전달하지 않는다.
'''),
    code('''
# ===== 2. 게임 생성 =====
state = core.new_game(list(config.CHARACTERS), None, seed=7)

print("===== 역할 배정 (진실, AI에게는 보이지 않음) =====")
for p in state.players:
    print("  %s : %s" % (p.name, config.ROLE_KO[p.role]))

# 역할별 대표 플레이어 한 명씩 고르기
rep = {}
for p in state.players:
    rep.setdefault(p.role, p.name)
print("\\n대표 플레이어:", {config.ROLE_KO[r]: n for r, n in rep.items()})
'''),
    code('''
# ===== 3. 역할별 LLMPlayer 생성 =====
llm = agents.make_llm()

players = {
    role: agents.LLMPlayer(name, role, vs, llm=llm, state=state)
    for role, name in rep.items()
}

print("===== LLMPlayer 생성 완료 =====")
for role, ai in players.items():
    print("%-6s %-5s 공통 도구: %s" % (config.ROLE_KO[role], ai.name, list(ai.tools.keys())))
print("(역할별 행동 도구 - 마피아 choose_kill_target / 의사 protect / 경찰 investigate - 는 행동할 때마다 후보 목록과 함께 만들어진다)")
'''),
    md('''
## 2. 정보 분리 확인: PlayerView

AI에게는 `build_view()`가 만든 `PlayerView`만 전달된다. 같은 시점에 **마피아의 시야**와 **시민의 시야**를 비교해보자.
마피아의 `private_info`에는 동료 마피아가 있지만, 시민에게는 아무것도 없어야 한다.
'''),
    code('''
# ===== 4. PlayerView 비교 =====
mafia_name = rep["mafia"]
citizen_name = rep["citizen"]

view_m = core.build_view(state, mafia_name, "speak")
view_c = core.build_view(state, citizen_name, "speak")

for v in [view_m, view_c]:
    print("----- %s의 PlayerView -----" % v.name)
    print("자기 역할:", v.role_ko)
    print("비공개 정보(private_info):", v.private_info)
    print("투표/발언 대상 후보:", v.candidates)
print()
print("마피아 view 에 동료 정보가 있는가?", len(view_m.private_info) > 0)
print("시민 view 에 비공개 정보가 있는가?", len(view_c.private_info) > 0)
print("PlayerView 필드 목록:", list(view_m.__dataclass_fields__.keys()))
print("-> 다른 사람의 역할(role) 필드는 PlayerView 어디에도 없다.")
'''),
    md('''
## 3. 밤 행동 (역할별 도구)

- 마피아: `choose_kill_target` 으로 처치 대상 선택
- 의사: `protect` 로 보호 대상 선택
- 경찰: `investigate` 로 조사 대상 선택
- 시민: 밤 행동 없음 (`None`)

AI는 도구로 선택을 **기록**할 뿐이고, 선택이 유효한지는 코드가 다시 검사한다. `last_source`는 어떤 경로로 답이 나왔는지 알려준다 (agent / structured / none).
'''),
    code('''
# ===== 5. 밤 행동 =====
actions = {"mafia": "kill", "doctor": "protect", "police": "investigate"}

for role, action in actions.items():
    ai = players[role]
    view = core.build_view(state, ai.name, action)
    decision = ai.night_action(view)
    print("[%s %s] 후보: %s" % (config.ROLE_KO[role], ai.name, view.candidates))
    print("   선택:", decision.target if decision else None, "| 이유:", getattr(decision, "reason", ""), "| 경로:", ai.last_source)

print("\\n[시민]", players["citizen"].night_action(core.build_view(state, rep["citizen"], "speak")))
'''),
    md('''
## 4. 발언 1회와 투표 1회

발언은 한 번의 LLM 호출로 `{속마음(thought), 의심도(suspicion), 발언(text)}`을 구조화해서 받는다.
**속마음은 게임 중에는 공개되지 않고** `state.notes`에만 저장된다 (게임이 끝나면 `reveal_report`로 공개).
'''),
    code('''
# ===== 6. 발언 1회 =====
import time

ai = players["citizen"]
view = core.build_view(state, ai.name, "speak")

t0 = time.time()
speech = ai.speak(view)
print("[%s] 발언: %s" % (ai.name, speech.text))
print("   속마음(비공개):", speech.thought)
print("   의심도(비공개):", speech.suspicion)
print("   소요 시간: %.1f초" % (time.time() - t0))
'''),
    code('''
# ===== 7. 투표 1회 =====
ai = players["police"]
view = core.build_view(state, ai.name, "vote")

vote = ai.vote(view)
print("[%s] 투표 후보: %s" % (ai.name, view.candidates))
print("   투표 -> %s (이유: %s)" % (vote.target, vote.reason))
print("   후보 안에 있는가?", core.normalize_choice(vote.target, view.candidates) is not None)
'''),
    md('''
## 5. 정체 노출 테스트 (leak test)

마피아가 공개 토론에서 "나는 마피아야" 같은 말을 하면 게임이 망가진다. 두 겹으로 방어한다.
1. 프롬프트에서 금지 (LLM이 지켜주길 기대)
2. **코드 검사** `contains_role_leak()` - LLM이 어겨도 코드가 걸러낸다 (이 노트북에서 확인)
'''),
    code('''
# ===== 8-1. 규칙 검사 단위 테스트 =====
samples = [
    ("mafia", "나는 마피아야, 믿어줘"),
    ("mafia", "저는 마피아가 아닙니다. 시민이에요."),
    ("mafia", "우리 마피아끼리 힘내자"),
    ("mafia", "저는 윤서진님이 의심스럽습니다."),
    ("citizen", "저는 시민입니다."),
]
for role, text in samples:
    print("%-4s | 노출=%-5s | %s" % (role, core.contains_role_leak(text, role), text))
'''),
    code('''
# ===== 8-2. 노출 발언을 하는 가짜 마피아로 safe_speak 확인 =====
class LeakyMafia:
    """항상 정체를 말해버리는 나쁜 AI (테스트용)"""

    def speak(self, view):
        return core.Speech(text="나는 마피아야! 우리 마피아끼리 이기자!", thought="일부러 노출")


test_state = core.new_game(list(config.CHARACTERS), None, seed=1)
test_mafia = [p.name for p in test_state.players if p.role == "mafia"][0]
test_view = core.build_view(test_state, test_mafia, "speak")

speech, used_fallback = core.safe_speak(LeakyMafia().speak, test_view, test_state)
print("최종 발언:", speech.text)
print("fallback 사용:", used_fallback)
print("leak_blocked 통계:", test_state.stats["leak_blocked"])
'''),
    code('''
# ===== 8-3. 실제 LLM 마피아가 정체를 말하는지 3회 확인 =====
leaks = 0
mafia_ai = players["mafia"]
for i in range(3):
    sp = mafia_ai.speak(core.build_view(state, mafia_ai.name, "speak"))
    leaked = core.contains_role_leak(sp.text, "mafia")
    leaks += int(leaked)
    print("%d회: 노출=%s | %s" % (i + 1, leaked, sp.text))
print("정체 노출 횟수: %d / 3" % leaks)
'''),
    md('''
## 6. 안전장치(fallback) 강제 시연

LLM이 없는 이름을 말하거나, 예외를 내거나, 엉뚱한 형식을 돌려줘도 `safe_decide()`가
**정규화 -> 재시도 -> 무작위 대체** 순서로 처리해서 게임은 절대 멈추지 않는다.
'''),
    code('''
# ===== 9. fallback 시연 =====
class BrokenAI:
    """항상 이상한 출력을 내는 AI (테스트용)"""

    def __init__(self, mode):
        self.mode = mode

    def vote(self, view):
        if self.mode == "garbage":
            return core.VoteDecision("몰라요 아무나")      # 후보에 없는 이름
        if self.mode == "error":
            raise RuntimeError("LLM 호출 실패")           # 예외
        return core.VoteDecision("3번 " + view.candidates[2] + "님")   # 퍼지 매칭으로 구제 가능


demo_state = core.new_game(list(config.CHARACTERS), None, seed=3)
demo_view = core.build_view(demo_state, demo_state.players[0].name, "vote")

for mode in ["fuzzy", "garbage", "error"]:
    before = dict(demo_state.stats)
    target, used_fallback = core.safe_decide(BrokenAI(mode).vote, demo_view, demo_view.candidates, demo_state, "vote")
    print("%-8s -> 최종 선택: %s | fallback: %s | 재시도 +%d" % (
        mode, target, used_fallback, demo_state.stats["retries"] - before["retries"]))
print("\\n통계:", demo_state.stats)
'''),
    code('''
print("===== 02 완료: 이제 03_mafia_game.ipynb 로 이동하세요 =====")
print("이 노트북에서 사용한 LLM 호출 수(프로세스 전체):", agents.LLM_CALLS["count"])
'''),
]

# =====================================================
# 03_mafia_game
# =====================================================
nb3 = [
    md('''
# 03. 마피아 게임 실행 (GameEngine 버전)

**목적**: 순수 파이썬 `GameEngine`으로 마피아 한 판을 진행한다. 두 가지 모드를 지원한다.

| 모드 | 구성 | 설명 |
|---|---|---|
| 관전 모드 | AI 8명 | 사람은 구경만 한다. 마피아의 밤 대화/의사/경찰 선택과 **AI 마다의 사고 과정**도 보인다 |
| 참여 모드 | 사람 1 + AI 7 | 사람 정체 정보만 보인다. 발언, 투표를 직접 입력한다. `?질문`으로 사회자에게 규칙 문의 |

`GameEngine`의 역할 분담 (**LLM은 연기, 진실은 코드가 지킨다**)

- 엔진(코드): 밤 처리, 사망 발표, 투표 집계, 동점 재투표, 승리 판정, 정보 분리
- 컨트롤러(LLM / 사람): 발언과 선택만 담당
- 게임이 끝나면 `reveal_report()`로 전원의 역할과 속마음을 공개하고, `save_log()`로 JSON 저장

진행: 밤 1 -> 낮 1(토론) -> 투표 -> 밤 2 -> ... (마피아 전멸 시 시민 승, 마피아 수 >= 시민 수 시 마피아 승, 10일 초과 무승부)
'''),
    code(CWD_CELL),
    code('''
# ===== 1. 준비 =====
import random
import time

import rag
import mafia_core as core
import mafia_agents as agents

vs = rag.load_db()
print("===== 벡터 DB 로드 완료 (Chunk %d개) =====" % vs._collection.count())
'''),
    md('''
## 1. 관전 모드: AI 8명 + 사고 과정 보기

`build_controllers()`는 플레이어마다 역할에 맞는 도구와 캐릭터 페르소나를 가진 `LLMPlayer`를 만들어 준다.
**한 판에 수 분이 걸린다.** `SEED`를 바꾸면 다른 역할 배정이 나온다.

`SHOW_THINKING = True`이면 AI가 결정할 때마다 **사고 과정** 상자가 함께 나온다. 챗봇(질문 -> 답변)과 달리,
각 AI가 *자기 정보만 받고 -> 룰북을 검색하고 -> 도구를 호출해 -> 결정하는* 에이전트라는 것을 눈으로 확인할 수 있다.

| 상자 안의 줄 | 의미 |
|---|---|
| 받은 정보(PlayerView) / 나만 아는 정보 | 코드가 골라 준 그 AI 전용 정보 (다른 사람의 직업은 없음) |
| RAG 검색 | 메타데이터 필터로 읽은 룰북 페이지 (내 역할 전략, 내 캐릭터 프로필) |
| 도구 호출 / 도구 응답 | `create_agent`가 고른 도구(`choose_kill_target`, `investigate`, `protect`, `vote`)와 인자, 도구의 후보 검사 결과 |
| 전략 판단(코드) / 몰아가기 감지(코드) | 코드가 상황을 계산해 내린 지시 (예: 경찰이 정체를 밝힐 시점) |
| 안전장치 작동 / 후처리(코드) | 반복, 자기 이름 3인칭, 없는 밤 행동 언급 등을 감지해 재생성하거나 고친 기록 |
| 속마음 / 의심도 | 공개 발언과 따로 남는 비공개 추리 (게임이 끝나면 공개) |

> 사고 과정은 다른 사람의 비밀을 보여 주므로 **관전 모드에서만** 켜진다. 사람이 참여하면 자동으로 꺼진다.
'''),
    code('''
# ===== 2. 관전 모드 실행 =====
SEED = 11
SHOW_THINKING = True   # AI 사고 과정(받은 정보 -> RAG -> 도구 호출 -> 안전장치 -> 결정) 출력. 결과만 보려면 False

state = core.new_game(list(config.CHARACTERS), None, seed=SEED)
controllers = agents.build_controllers(state, vs, show_thinking=SHOW_THINKING)
engine = core.GameEngine(state, controllers)   # on_event 기본값: 한국어로 출력

t0 = time.time()
winner = engine.run()
elapsed = time.time() - t0

print("\\n===== 게임 종료 =====")
print("승자:", winner, "| 진행 일수:", state.day, "| 소요 시간: %.0f초" % elapsed)
'''),
    md('''
## 2. 게임 종료 후: 속마음 공개, 로그 저장, 통계
'''),
    code('''
# ===== 3. 결과 공개 =====
print(core.reveal_report(state))
'''),
    code('''
# ===== 4. 로그 저장 + 통계 =====
log_path = core.save_log(state)
print("===== 로그 저장 완료 =====")
print("저장 위치:", log_path)

print("\\n===== 안전장치 통계 =====")
print("AI 결정 요청 수 :", state.stats["llm_calls"])   # 발언/투표/밤 행동 요청 수 (실제 LLM 호출은 더 많음)
print("재시도 횟수      :", state.stats["retries"])
print("fallback 횟수    :", state.stats["fallbacks"])
print("정체 노출 차단   :", state.stats["leak_blocked"])
print("일차당 평균 소요 : %.0f초" % (elapsed / max(state.day, 1)))
'''),
    md('''
## 3. 시연용: 녹화된 판 다시 보기

실제 LLM 한 판은 수 분이 걸리고 결과도 매번 다르다. 발표 시간에 맞추기 위해,
같은 코드(사고 과정 보기 켜짐)로 **미리 진행한 실제 한 판의 출력**을 `data/demo/` 에 저장해 두었다.
아래 셀은 그 기록을 다시 출력할 뿐이며 LLM을 호출하지 않는다. (실시간 진행은 위의 1번 셀)
'''),
    code('''
# ===== 5. 녹화된 판 재생 (LLM 호출 없음) =====
import json

DEMO_TXT = config.DATA_DIR / "demo" / "demo_game.txt"     # 진행 화면 그대로
DEMO_LOG = config.DATA_DIR / "demo" / "demo_game.json"    # save_log() 결과 (역할, 속마음 포함)
DELAY = 0.05   # 줄 사이 간격(초). 0 이면 한 번에 출력

meta = json.loads(DEMO_LOG.read_text(encoding="utf-8"))
print("===== 녹화된 판: seed %s, 승자 %s, %s일차에 종료 =====\\n" % (meta["seed"], meta["winner"], meta["day"]))
for line in DEMO_TXT.read_text(encoding="utf-8").splitlines():
    print(line, flush=True)
    if DELAY:
        time.sleep(DELAY)
'''),
    md('''
## 4. 참여 모드: 사람 1 + AI 7

- 사람의 이름은 `config.HUMAN_NAME`(기본 "서지우", 본인 이름으로 바꿔도 됨), AI는 캐릭터 8명 중 7명이 무작위로 참가한다.
- 입력 방법: **번호 또는 이름**을 입력한다. 발언은 자유롭게 문장을 입력한다 (빈 입력은 패스).
- `?질문`: 사회자(룰북 RAG, `type=rule` 필터)에게 규칙을 물어본다. 예: `?경찰은 누구를 조사할 수 있나요`
- 사람이 처치/보호/조사 대상을 정하는 밤에는 본인의 비공개 정보만 화면에 나온다.

> 실행하면 입력창이 뜬다. 입력을 하지 않고 넘어가려면 이 셀은 건너뛰어도 된다.
'''),
    code('''
# ===== 6. 참여 모드 실행 =====
HUMAN_SEED = None   # 숫자를 넣으면 같은 배정으로 재현된다

ai_names = random.Random(HUMAN_SEED).sample(config.CHARACTERS, 7)
h_state = core.new_game(ai_names, config.HUMAN_NAME, seed=HUMAN_SEED)
h_controllers = agents.build_controllers(h_state, vs)   # 사람은 HumanController (input 사용)
h_engine = core.GameEngine(h_state, h_controllers)

h_winner = h_engine.run()
print("\\n승자:", h_winner)
'''),
    code('''
# ===== 7. 참여 모드 결과 공개 =====
print(core.reveal_report(h_state))
print("로그:", core.save_log(h_state))
'''),
    code('''
print("===== 03 완료: 이제 04_langgraph_mafia.ipynb 로 이동하세요 =====")
'''),
]

# =====================================================
# 04_langgraph_mafia
# =====================================================
nb4 = [
    md('''
# 04. LangGraph 로 만든 마피아 게임

**목적**: 03에서 `GameEngine`의 while 문으로 짠 게임 진행을 **LangGraph(상태 그래프)**로 다시 표현한다.
규칙 판정은 `mafia_core`의 함수를 그대로 재사용하고, 그래프는 "다음에 무엇을 실행할지"만 맡는다.

## LangChain `create_agent` vs LangGraph

| | create_agent (06_Agent) | LangGraph |
|---|---|---|
| 관점 | 한 명의 AI가 도구를 골라 쓰는 **개별 판단** | 게임 전체의 **진행 순서와 상태 관리** |
| 흐름 | LLM이 정함 (도구 호출 반복) | 개발자가 노드와 엣지로 정함 |
| 분기 | 없음 | 조건부 엣지 (승리 여부 등) |
| 사람 개입 | 어려움 | `interrupt` / `Command(resume=...)` |
| 상태 저장 | 없음 | checkpointer |

이 프로젝트에서는 **AI 한 명의 두뇌 = create_agent, 게임 심판/진행 = LangGraph**로 역할을 나눴다.

## LangGraph 핵심 개념 (이 노트북에서 순서대로 사용)

1. **State**: 노드들이 함께 읽고 쓰는 공유 데이터 (`TypedDict`). 우리는 `game`(GameState), 밤 선택, 발언/투표 순서 등을 담는다.
2. **Node**: State를 받아 일을 하고 **바뀐 부분만 딕셔너리로 돌려주는 함수**. (`night_mafia`, `morning`, `speak` ...)
3. **Edge**: 노드 사이의 고정된 다음 길. (`night_start -> night_mafia`)
4. **Conditional edge**: 함수의 반환값에 따라 다음 노드를 고른다. (승부가 났으면 `ending`, 아니면 다음 단계)
5. **interrupt / Command(resume)**: 노드 실행 중 그래프를 멈추고 사람의 입력을 기다린다. 입력을 받아 재개한다.
6. **Checkpointer(MemorySaver) + thread_id**: 노드가 끝날 때마다 State를 저장한다. `thread_id`는 "이 게임"의 이름표이다. interrupt는 체크포인터가 있어야 동작한다.
'''),
    code(CWD_CELL),
    code('''
# ===== 1. 모듈 불러오기 =====
import random
import time

from langgraph.types import Command

import rag
import mafia_core as core
import mafia_agents as agents
import mafia_graph as mg

vs = rag.load_db()
print("===== 준비 완료 =====")
'''),
    md('''
## 1. 그래프 구조

게임 한 판의 흐름이다. 밤은 마피아(제안, 최종) -> 의사 -> 경찰 순서로 노드를 나눴고, 발언과 투표는 **한 명씩 한 번의 노드 실행**으로 처리한다.
(`speak`, `vote` 노드는 자기 자신으로 돌아오는 **반복 엣지**로 모든 사람을 처리한다.)

노드 이름과 역할

| 노드 | 하는 일 (mafia_core 재사용) |
|---|---|
| setup | 게임 시작 안내 |
| night_start / night_mafia_first / night_mafia_second / night_doctor / night_police | 밤 시작, 첫 마피아(제안) -> 둘째 마피아(최종) -> 의사 -> 경찰 선택 수집 |
| morning | 밤 결과 판정(`apply_night`) + 승리 판정 |
| day_start / speak | 낮 토론 순서 정하기 / 한 명 발언 |
| vote_start / vote / tally | 투표 시작 / 한 명 투표 / 개표 (동점이면 재투표) |
| execution | 처형 후 승리 판정 |
| ending | 종료 처리 |

조건부 엣지: `morning` 뒤 (승부 -> ending / 계속 -> day_start), `speak` 반복, `vote` 반복, `tally` 뒤 (동점 -> vote 재투표 / execution), `execution` 뒤 (승부 -> ending / 계속 -> night_start)
'''),
    code('''
# ===== 2. 그래프 시각화 =====
# 먼저 LLM 없이 구조만 보기 위해 RandomController 로 그래프를 만든다.
game0 = core.new_game(list(config.CHARACTERS), None, seed=0)
random_ctrls = {p.name: core.RandomController(p.name, seed=i) for i, p in enumerate(game0.players)}

graph = mg.build_graph(random_ctrls)

# PNG 는 인터넷(mermaid.ink)이 필요하므로, 실패하면 Mermaid 텍스트를 출력한다.
mermaid_text = mg.show_graph(graph)
'''),
    md('''
## 2. 빠른 동작 확인 (LLM 없이, RandomController)

LLM을 쓰기 전에 그래프 흐름이 맞는지 확인한다. `run_spectator`는 사람 입력 없이 끝까지 실행한다.
`thread_id`는 체크포인트 저장소에서 이 게임을 구분하는 이름이다.
'''),
    code('''
# ===== 3. RandomController 로 그래프 한 판 =====
final = mg.run_spectator(graph, game0, thread_id="quick-test")

print("\\n===== 그래프 실행 완료 =====")
print("승자:", final.winner, "| 진행 일수:", final.day)
print("체크포인트 개수:", len(list(graph.get_state_history(mg.make_config("quick-test")))))
'''),
    md('''
### 체크포인트 들여다보기

노드가 하나 끝날 때마다 State가 저장된다. `get_state_history`로 지나온 노드 순서를 볼 수 있다.
'''),
    code('''
# ===== 4. 실행 경로 확인 =====
cfg = mg.make_config("quick-test")
history = list(graph.get_state_history(cfg))

path = [snap.next[0] for snap in reversed(history) if snap.next]

print("실행된 노드 순서(앞 25개):", path[:25])
print("총 노드 실행 횟수:", len(path))
'''),
    md('''
## 3. 관전 모드: LLM 8명을 LangGraph 로 실행

03과 같은 `LLMPlayer`를 컨트롤러로 넣기만 하면 된다. **진행(그래프)과 판단(LLM)이 분리**되어 있어서 컨트롤러만 바꿔 끼울 수 있다.
수 분이 걸린다.
'''),
    code('''
# ===== 5. LLM 관전 모드 (LangGraph) =====
SEED = 21

game = core.new_game(list(config.CHARACTERS), None, seed=SEED)
ctrls = agents.build_controllers(game, vs)
llm_graph = mg.build_graph(ctrls)

t0 = time.time()
final = mg.run_spectator(llm_graph, game, thread_id="spectator-%d" % SEED)
print("\\n승자:", final.winner, "| 일수:", final.day, "| 소요 %.0f초" % (time.time() - t0))
print(core.reveal_report(final))
'''),
    md('''
## 4. Human-in-the-loop: 사람 입력 대기와 재개

참여 모드에서 사람 차례가 오면 노드 안의 `interrupt(payload)`가 실행되어 **그래프가 멈춘다**.
호출자는 `result["__interrupt__"]`로 질문 내용(payload)을 받고, `Command(resume=답)`으로 다시 `invoke`하면 멈춘 곳에서 이어진다.

```
graph.invoke(입력, cfg)                 -> 사람 차례에서 멈춤, result["__interrupt__"] 에 질문
graph.invoke(Command(resume="답"), cfg) -> 같은 thread_id 의 체크포인트에서 재개
```

> 주의: 재개하면 멈췄던 **노드를 처음부터 다시 실행**한다. 그래서 `speak`/`vote`를 "한 명 = 한 노드"로 쪼갰다.
> (한 노드에서 AI 7명을 모두 처리하면 사람 차례마다 AI 호출이 반복된다.)
>
> 같은 이유로 마피아도 `night_mafia_first`(제안) / `night_mafia_second`(최종)로 나눠 첫 제안을 State(`proposal`)에 저장한다. 사람이 둘째 마피아일 때 재개해도 첫 마피아 AI는 다시 호출되지 않는다.\n>\n> 또한 `mafia_core`의 안전장치가 예외를 삼키기 때문에, `GraphHumanController`는 interrupt가 던지는 예외를 `BaseException`으로 감싸 노드 밖으로 통과시킨다 (`mafia_graph.py` 참고).

먼저 **스크립트로 자동 응답**하는 시연으로 흐름을 눈으로 확인한다. (AI는 RandomController라서 빠르다)
'''),
    code('''
# ===== 6. interrupt / resume 단계별 시연 =====
ai7 = random.Random(5).sample(config.CHARACTERS, 7)
hgame = core.new_game(ai7, config.HUMAN_NAME, seed=5)
hctrls = {p.name: core.RandomController(p.name, seed=i) for i, p in enumerate(hgame.players)}
hctrls[config.HUMAN_NAME] = mg.GraphHumanController(config.HUMAN_NAME)   # 사람 자리만 interrupt 컨트롤러

hgraph = mg.build_graph(hctrls, on_event=lambda ev: None)   # 이벤트 출력 생략
hcfg = mg.make_config("hitl-demo")

result = hgraph.invoke({"game": hgame}, hcfg)

for step in range(4):
    if "__interrupt__" not in result:
        break
    payload = result["__interrupt__"][0].value
    print("[멈춤 %d] 종류=%s | 다음 실행 노드=%s" % (step + 1, payload["kind"], hgraph.get_state(hcfg).next))
    print("   질문:", payload["prompt"], "| 후보:", payload["candidates"][:3], "...")
    answer = "시연용 발언입니다." if payload["kind"] == "speak" else payload["candidates"][0]
    print("   재개 값:", answer)
    result = hgraph.invoke(Command(resume=answer), hcfg)
'''),
    code('''
# ===== 7. 나머지는 자동 응답으로 끝까지 =====
def auto_answer(payload):
    if payload["kind"] == "speak":
        return "저는 시민이라고 생각합니다."
    return random.choice(payload["candidates"])


while "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    result = hgraph.invoke(Command(resume=auto_answer(payload)), hcfg)

hfinal = mg.get_game(hgraph, hcfg)
print("===== 참여 모드(스크립트) 종료 =====")
print("승자:", hfinal.winner, "| 일수:", hfinal.day)
'''),
    md('''
### 직접 참여하기 (사람 1 + AI 7)

아래 셀은 `input()`으로 직접 입력한다. `?질문`을 입력하면 사회자(룰북 RAG)가 답한다.
`run_with_human`이 interrupt 처리 루프(`invoke -> payload 확인 -> 입력 받기 -> Command(resume)`)를 대신해준다.
'''),
    code('''
# ===== 8. 직접 참여 (input) =====
moderator = agents.Moderator(vs)

pgame = core.new_game(random.sample(config.CHARACTERS, 7), config.HUMAN_NAME, seed=None)
pctrls = agents.build_controllers(pgame, vs)
pctrls[config.HUMAN_NAME] = mg.GraphHumanController(config.HUMAN_NAME)   # input 대신 interrupt 사용

pgraph = mg.build_graph(pctrls)
pfinal = mg.run_with_human(pgraph, pgame, thread_id="human-play", ask_rule=moderator.explain_rule)

print(core.reveal_report(pfinal))
'''),
    md('''
## 5. 관전 N판 통계

여러 판을 자동으로 돌려 **마피아/시민 승률, 평균 진행 일수, 안전장치 발동 횟수**를 모은다.
LLM으로는 한 판이 수 분 걸리므로 기본 `N_LLM`은 작게 둔다. 먼저 RandomController로 규칙/그래프의 균형을 빠르게 본다.
'''),
    code('''
# ===== 9. 빠른 통계: RandomController 50판 =====
N_RANDOM = 50


def random_factory(game):
    return {p.name: core.RandomController(p.name, seed=i + (game.seed or 0) * 10) for i, p in enumerate(game.players)}


summary_random = mg.run_batch(N_RANDOM, random_factory, seed0=0)
mg.print_batch_summary(summary_random)
'''),
    code('''
# ===== 10. LLM 통계 (시간이 오래 걸림: N_LLM 을 늘리면 더 정확하지만 느리다) =====
N_LLM = 2   # 한 판 수 분. 발표 전 여유가 있을 때만 늘리세요.

t0 = time.time()
summary_llm = mg.run_batch(N_LLM, lambda g: agents.build_controllers(g, vs), seed0=100)
mg.print_batch_summary(summary_llm)
print("총 소요: %.0f초 (판당 %.0f초)" % (time.time() - t0, (time.time() - t0) / N_LLM))
'''),
    md('''
## 6. 정리

- **State / Node / Edge**: 게임 한 판을 "노드(한 단계) + 엣지(순서)"로 선언해 흐름이 눈에 보인다.
- **조건부 엣지**: 승리 판정, 반복(발언/투표), 동점 재투표를 분기 함수로 표현했다.
- **interrupt + Command(resume)**: 사람 입력을 기다리는 일을 `input()` 루프 없이 그래프 기능으로 처리했다.
- **checkpointer + thread_id**: 노드마다 상태가 저장되어 멈춤/재개가 가능하고, 지나온 경로도 확인할 수 있다.
- 규칙 판정은 모두 `mafia_core`가 하므로 03(GameEngine)과 04(LangGraph)는 **같은 규칙, 다른 진행 방식**이다.
'''),
]

# =====================================================
# 05_ask_agent
# =====================================================
nb5 = [
    md("""
# 05. 질문에 답하는 에이전트 (RAG 질의응답)

**목적**: 사용자가 질문하면 에이전트가 **RAG 로 근거를 찾아 출처와 함께 답한다.**

| 부분 | 무엇을 검색하나 | 어떻게 답하나 |
|---|---|---|
| 1. 사회자 (`Moderator`) | 룰북 PDF 의 규칙 페이지 (`type=rule` 필터) | 검색한 발췌만 근거로 답하고 출처 페이지를 붙인다. 룰북에 없으면 모른다고 한다 |
| 2. 해설 에이전트 (`mafia_qa.GameQA`) | 룰북 + **끝난 게임의 기록**(발언, 투표, 밤 행동, 속마음) | `create_agent` 가 질문에 맞는 **도구를 스스로 골라** 검색한 뒤 근거를 붙여 답한다 |

03 노트북의 AI 8명은 같은 RAG 를 "행동"에 쓰고, 이 노트북의 에이전트는 "질문에 답하는 데" 쓴다.
"""),
    code(CWD_CELL),
    code('''
# ===== 1. 준비 =====
import json

import rag
import mafia_agents as agents
import mafia_qa

vs = rag.load_db()
print("===== 룰북 벡터 DB 로드 완료 (Chunk %d개) =====" % vs._collection.count())
'''),
    md("""
## 1. 사회자: 룰북 RAG 질의응답

질문 -> `rag.search_rules()` (메타데이터 필터 `type=rule`, 상위 3개) -> LLM 이 발췌만 근거로 3문장 이내 답변 -> 출처 페이지 표시.
마지막 질문처럼 룰북에 없는 내용은 지어내지 않고 모른다고 답하는지 확인한다.
"""),
    code('''
# ===== 2. 사회자에게 규칙 질문 =====
moderator = agents.Moderator(vs)

questions = [
    "의사는 자기 자신을 보호할 수 있나요?",
    "투표에서 동점이 나오면 어떻게 되나요?",
    "마피아는 언제 이기나요?",
    "이 게임에서 우승하면 상금이 있나요?",   # 룰북에 없는 질문
]
for q in questions:
    docs = rag.search_rules(vs, q, k=3)
    print("[질문]", q)
    print("   검색된 페이지:", ", ".join("p.%d %s" % (d.metadata["page"] + 1, d.metadata["title"]) for d in docs))
    print("[사회자]", moderator.explain_rule(q))
    print()
'''),
    md("""
## 2. 해설 에이전트: 질문에 맞는 지식을 스스로 골라 검색

게임이 끝나면 그 판의 기록을 문서로 나눠 **두 번째 벡터 DB**(메모리 Chroma, bge-m3)에 넣는다.
에이전트(`create_agent`)는 질문을 보고 아래 도구 중 필요한 것을 골라 호출한다.

| 도구 | 검색 대상 | 쓰이는 질문 예 |
|---|---|---|
| `search_rulebook` | 룰북 RAG (`type=rule`) | "의사는 자기 자신을 보호해도 돼?" |
| `search_game_log` | 이번 판 기록 RAG (메타데이터 `kind`/`day`/`actor`/`target` 로 사람별 필터) | "마피아는 왜 박준수를 노렸어?" |
| `player_summary` | 코드가 계산한 사실 (역할, 득표, 밤 행동, 조사 결과) | "경찰은 누구를 조사했고 결과가 뭐였어?" |

- 게임 기록에는 밤 행동과 속마음 같은 **비밀이 들어 있어서, 게임이 끝난 뒤에만** 만든다 (`from_state()` 는 진행 중인 게임이면 거부한다).
- 에이전트가 도구를 쓰지 않고 답하려 하면, 코드가 두 저장소를 모두 검색해 근거를 준 뒤 다시 답하게 한다.
- 여기서는 `data/demo/` 에 저장된 실제 한 판(시민팀 2일차 승리)에 대해 질문한다.
"""),
    code('''
# ===== 3. 끝난 게임으로 해설 에이전트 만들기 =====
qa = mafia_qa.GameQA.from_file(vs, config.DATA_DIR / "demo" / "demo_game.json")

kinds = {}
for m in qa.game_vs.get()["metadatas"]:
    kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
print("===== 게임 기록 벡터 DB: 문서 %d개 =====" % sum(kinds.values()))
print("종류별:", kinds)   # role / speech / vote / public / night / reason / thought
'''),
    code('''
# ===== 4. 질문하기 (에이전트가 고른 도구가 함께 출력된다) =====
for q in [
    "경찰은 첫날 밤에 누구를 조사했고 결과가 뭐였어?",
    "마피아는 왜 첫날 밤에 박준수를 노렸어?",
    "정도현은 왜 1일차에 강태호에게 투표했어?",
    "의사는 자기 자신을 보호해도 돼?",
    "이번 판에서 오지환은 피자를 좋아했어?",   # 기록에 없는 질문
]:
    qa.ask(q)
    print()
'''),
    md("""
### 직접 질문하기

`MY_QUESTION` 을 바꿔서 실행한다. 03 노트북에서 방금 끝난 게임에 질문하려면 같은 커널에서
`qa_live = mafia_qa.GameQA.from_state(vs, state)` 로 만든 뒤 `qa_live.ask("...")` 를 호출한다.
"""),
    code('''
# ===== 5. 직접 질문 =====
MY_QUESTION = "의사는 누구였고 누구를 보호했어?"
result = qa.ask(MY_QUESTION)
'''),
    md("""
## 3. 정리

| | 일반 챗봇 | 사회자 | 해설 에이전트 |
|---|---|---|---|
| 지식 | 모델이 아는 것 | 룰북 RAG | 룰북 RAG + 게임 기록 RAG + 코드 계산 |
| 흐름 | 질문 -> 답 | 질문 -> 검색 -> 답 (고정) | 질문 -> **도구 선택** -> 검색 -> 답 |
| 근거 | 없음 | 룰북 페이지 | 룰북 페이지, 날짜별 기록 |
| 모를 때 | 지어낼 수 있음 | "룰북에 없다" | "기록에 없다" |

한계: 8B 모델이라 "이번 판 사실 + 규칙"처럼 두 가지를 한 번에 묻으면 한쪽만 답할 때가 있다. 질문을 나눠서 하면 정확하다.
"""),
]

save("01_create_db.ipynb", nb1)
save("02_agent_test.ipynb", nb2)
save("03_mafia_game.ipynb", nb3)
save("04_langgraph_mafia.ipynb", nb4)
save("05_ask_agent.ipynb", nb5)
print("saved")
