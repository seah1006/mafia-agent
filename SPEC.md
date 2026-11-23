# 08_Mafia_Agent 공통 설계서 (SPEC)

딥러닝실습 2학기 과제 "나만의 에이전트 만들기". 마감 2026-11-24(화).
평가: 1) 구현 완성도 2) 구현 컨셉 설명.
모든 작업자는 이 문서를 계약(contract)으로 삼는다. 인터페이스를 바꿔야 하면 먼저 메인 에이전트에게 메시지로 알리고, 승인 후 이 문서의 "변경 기록"에 한 줄 추가한다.

## 0. 환경 / 공통 규칙
- 제출 환경: 실습실 PC, Windows, **Python 3.11.9**, VS Code + Jupyter 노트북, Ollama `qwen3:8b`(LLM), `bge-m3`(임베딩).
- 이 PC 테스트 환경: `C:\AI_Lab\venv\Scripts\python.exe` (Python 3.12.10, langchain 1.4.8, langchain-core 1.7.4, langchain-community 0.4.2, langchain-ollama 1.1.0, langchain-chroma 1.1.0, langchain-text-splitters 1.1.3, langgraph 1.2.19, pypdf 6.21.4, pydantic 2.14.2, chromadb 1.5.9). Ollama는 http://127.0.0.1:11434 에서 이미 실행 중 (직접 serve/pull 하지 말 것).
- 이 venv 로 python 을 실행할 때는 PowerShell 에서 반드시 먼저:
  `$env:PIP_PREFIX=$null; $env:PYTHONHOME=$null; $env:PYTHONPATH=$null; $env:PYTHONNOUSERSITE=$null; $env:PYTHONIOENCODING='utf-8'`
  (Aside 셸이 이 값들을 설정해 둬서, 지우지 않으면 엉뚱한 런타임을 쓴다.) **pip install 금지.** 패키지가 필요하면 메인에게 요청.
- **Python 3.11 호환 필수**: f-string 안에서 바깥과 같은 따옴표 재사용 금지(3.12 전용), `type X = ...` 문법 금지, PEP 695 제네릭 금지, `itertools.batched` 금지. 검증: `python -m py_compile` + 위 규칙 grep.
- 경로: `.py` 모듈은 `config.BASE_DIR = Path(__file__).resolve().parent` 기준. 노트북은 08_Mafia_Agent 폴더에서 열리므로 `import config` 후 config 경로 사용. 절대경로 하드코딩 금지.
- 주석/출력 문자열은 한국어. 교수님 샘플 스타일(`# ===== 1. LLM 설정 =====` 같은 구획 주석)을 따른다.
- 다른 작업자 소유 파일은 수정하지 않는다. 필요하면 해당 작업자나 메인에게 메시지.

## 1. 파일과 소유자
| 파일 | 소유 | 내용 |
|---|---|---|
| `SPEC.md`, `config.py` | 메인 | 계약, 공통 상수 |
| `data/마피아_룰북.pdf` (+ `data/rulebook_src/` 원고) | A (룰북) | 16페이지 룰북 + 캐릭터 설정집 |
| `mafia_core.py`, `tests/test_core.py` | B (코어) | 규칙 엔진, 상태, 정보 분리(PlayerView), 안전장치, 게임 루프, Random/Human 컨트롤러 |
| `rag.py`, `mafia_agents.py`, `tests/test_agents.py` | C (에이전트) | RAG(DB 생성/검색), LLM 플레이어 컨트롤러, 사회자 |
| `mafia_graph.py`, `01~04 *.ipynb`, `requirements.txt`, `README.md` | D (그래프/노트북) | LangGraph 버전, 노트북, 설명 문서 |

## 2. 게임 규칙 (8인 클래식 마피아)
- 인원 8: 마피아 2, 경찰 1, 의사 1, 시민 4. 역할 배정은 무작위(seed 지정 가능).
- 모드: **참여 모드** = 사람 1 + AI 7 (AI 캐릭터 8명 중 7명 무작위), **관전 모드** = AI 8.
- 사람 플레이어 이름 기본값 `"서지우"` (v1.2: "플레이어"는 마피아 AI가 첫날 밤 먼저 노리는 편향이 있어 변경).
- 진행: **밤 1 → 낮 1 → 밤 2 → ...** (첫 단계는 밤).
- 밤:
  1. 마피아: 살아있는 마피아가 순서대로 결정. 첫 마피아가 제안(대상+이유) → 두 번째 마피아는 동료 제안을 보고 최종 결정(동의 또는 변경). 한 명만 살아있으면 그 선택이 최종. 대상은 살아있는 비마피아.
  2. 의사: 살아있는 아무나 1명 보호(자기 자신 가능, 연속 보호 제한 없음).
  3. 경찰: 자신을 제외한 살아있는 1명 조사 → "마피아다/마피아가 아니다" 결과를 경찰 본인만 안다.
  4. 처치 대상 == 보호 대상이면 아무도 죽지 않음.
- 아침: 사망자 발표. **죽은 사람의 직업은 공개**한다(밤 사망, 처형 모두).
- 낮 토론: 살아있는 플레이어가 무작위 순서로 1인 1발언(라운드 수 `config.DISCUSSION_ROUNDS`, 기본 1). 발언은 2~3문장.
- 투표: 살아있는 모두가 자신 외 살아있는 1명에게 투표(기권 없음). 최다 득표자 처형. 동점이면 동점자만 후보로 **재투표 1회**, 그래도 동점이면 처형 없음.
- 승리 판정(밤 결과 후, 처형 후 매번): 마피아 생존 0 → **시민팀 승**. 마피아 생존 수 >= 비마피아 생존 수 → **마피아팀 승**. `config.MAX_DAYS`(10)일 초과 → 무승부.

## 3. config.py (메인 작성, 이미 존재)
상수: `BASE_DIR, DATA_DIR, PDF_PATH, DB_DIR, LOG_DIR, LLM_MODEL="qwen3:8b", EMBED_MODEL="bge-m3", OLLAMA_BASE_URL, ROLES, ROLE_KO, ROLE_DISTRIBUTION, CHARACTERS(8명), HUMAN_NAME, DISCUSSION_ROUNDS, MAX_DAYS, PAGE_META, SPEECH_MAX_CHARS, LLM_TEMPERATURE, LLM_MAX_RETRIES`.
역할 키: `"mafia" | "police" | "doctor" | "citizen"`. 한국어 표기는 `ROLE_KO`.

## 4. PDF 페이지 구성 (A 작성, C가 메타데이터로 사용) - 정확히 16페이지
PyPDFLoader 의 `metadata["page"]`(0부터) 기준. 각 페이지 첫 줄은 아래 제목을 그대로 쓴다(DB 생성 시 제목으로 검증).
| page(0-index) | 제목 | type | role | name |
|---|---|---|---|---|
| 0 | 마피아 게임 룰북 | rule | all | - |
| 1 | 기본 규칙과 승리 조건 | rule | all | - |
| 2 | 직업별 능력 | rule | all | - |
| 3 | 진행 순서와 특수 규칙 | rule | all | - |
| 4 | 마피아 전략 가이드 | strategy | mafia | - |
| 5 | 경찰 전략 가이드 | strategy | police | - |
| 6 | 의사 전략 가이드 | strategy | doctor | - |
| 7 | 시민 전략 가이드 | strategy | citizen | - |
| 8~15 | 캐릭터 프로필: {이름} | character | all | 이름 |
캐릭터 페이지 순서 = `config.CHARACTERS` 순서. 메타데이터 매핑은 `config.PAGE_META`.

## 5. mafia_core.py 인터페이스 (B 구현) - LLM 의존성 0
```python
@dataclass
class Player: name: str; role: str; alive: bool = True; is_human: bool = False

@dataclass
class GameState:
    players: list[Player]; day: int = 0; phase: str = "setup"   # setup|night|day|vote|end
    public_log: list[dict]        # {"day","phase","type","actor","target","text"} 공개 사건/발언/투표
    private_info: dict[str, list[str]]   # 이름 -> 본인만 아는 정보(경찰 조사결과, 마피아 동료 등)
    notes: dict[str, list[dict]]  # 이름 -> AI 속마음 기록 {"day","phase","thought","suspicion"}
    night_history: list[dict]; vote_history: list[dict]
    winner: str | None = None     # "citizen"|"mafia"|"draw"
    seed: int | None = None
    stats: dict                   # {"retries":0,"fallbacks":0,"leak_blocked":0,"llm_calls":0}

@dataclass
class PlayerView:          # 정보 분리의 핵심: AI 에게는 이것만 전달된다
    name: str; role: str; role_ko: str; day: int; phase: str
    alive: list[str]; dead: list[dict]          # [{"name","role_ko","cause"}]
    public_log_text: str                        # 최근 공개 기록(요약 포함), 길이 제한
    private_info: list[str]                     # 본인 비공개 정보만
    my_notes: list[dict]                        # 본인 과거 속마음만
    candidates: list[str]                       # 이번 행동의 유효 대상
    partner_proposal: dict | None = None        # 두 번째 마피아에게만 {"target","reason"}

@dataclass
class NightDecision: target: str; reason: str = ""
@dataclass
class Speech: text: str; thought: str = ""; suspicion: dict[str, int] = field(default_factory=dict)  # 0~100
@dataclass
class VoteDecision: target: str; reason: str = ""

class Controller(Protocol):
    name: str
    def night_action(self, view: PlayerView) -> NightDecision | None: ...   # 시민이면 호출 안 됨
    def speak(self, view: PlayerView) -> Speech: ...
    def vote(self, view: PlayerView) -> VoteDecision: ...
```
필수 함수:
- `new_game(ai_names: list[str], human_name: str | None, seed: int | None) -> GameState` (역할 배정, 마피아 private_info 에 동료 이름 기록)
- `build_view(state, name, action: str, partner_proposal=None) -> PlayerView` action: "kill"|"protect"|"investigate"|"speak"|"vote"|"revote"
- `valid_targets(state, name, action, tie_candidates=None) -> list[str]`
- `normalize_choice(text, candidates) -> str | None` ("강태호님", "태호", "3번 강태호" 등 퍼지 매칭)
- `resolve_night(state, kill_target, protect_target, investigation: tuple[str,str] | None) -> list[dict]`
- `tally_votes(votes: dict[str,str]) -> dict` {"executed": name|None, "tie": [names], "counts": {...}}
- `check_winner(state) -> str | None`
- `contains_role_leak(text, role) -> bool` (마피아가 "나 마피아", 공개 발언에 비공개 정보 직접 노출 등 단순 규칙)
- `safe_decide(fn, view, candidates, state, kind) -> (value, used_fallback)`: 컨트롤러 호출 → normalize → 실패 시 재시도(`LLM_MAX_RETRIES`) → 그래도 실패면 무작위 유효 대상, stats 갱신. 예외도 잡는다. 게임은 절대 멈추지 않는다.
- `GameEngine(state, controllers: dict[str, Controller], on_event=print)`: `run_night()`, `run_day()`, `run_vote()`, `run(max_days)` → winner. `on_event(event_dict)` 콜백으로 출력(노트북/LangGraph 재사용).
- `save_log(state, path=None) -> Path` (logs/ 에 JSON, 속마음 포함), `reveal_report(state) -> str` (게임 종료 후 역할 + 속마음 요약)
- 컨트롤러: `RandomController`(테스트용, LLM 없음), `HumanController`(input() 기반, 번호/이름 입력, 사회자 규칙 질문 `?질문` 지원은 hook 으로: `ask_rule: Callable[[str], str] | None`).

## 6. rag.py / mafia_agents.py 인터페이스 (C 구현)
```python
# rag.py
def load_pdf_with_meta(pdf_path=config.PDF_PATH) -> list[Document]   # 페이지마다 PAGE_META 부착 + 제목 검증
def split_docs(docs, chunk_size=500, chunk_overlap=50) -> list[Document]
def build_db(pdf_path=..., db_dir=config.DB_DIR, reset=True) -> Chroma
def load_db(db_dir=config.DB_DIR) -> Chroma
def search_rules(vs, query, k=3) -> list[Document]          # filter type=rule
def search_strategy(vs, role, query, k=2) -> list[Document] # filter type=strategy AND role
def get_persona(vs, name) -> str                            # filter type=character AND name (get, 질의 불필요)

# mafia_agents.py
def make_llm(temperature=config.LLM_TEMPERATURE) -> ChatOllama   # model qwen3:8b, reasoning=False
class LLMPlayer:  # mafia_core.Controller 구현
    def __init__(self, name, role, vs, llm=None, use_tools=True): ...
    # 역할별 도구: 공통 search_strategy, read_my_notes / 마피아 choose_kill_target / 경찰 investigate / 의사 protect / vote
    # 밤 행동·투표: create_agent + 역할 도구 1개(도구가 선택을 기록) → 실패 시 with_structured_output → 실패 시 None 반환(코어가 fallback)
    # speak: 페르소나(PDF) + 전략(RAG, 역할 필터) + PlayerView → 구조화 출력 {thought, suspicion, speech} 한 번의 호출
class Moderator:
    def __init__(self, vs, llm=None): ...
    def explain_rule(self, question: str) -> str      # 룰북 RAG + 출처 페이지
    def narrate(self, event: dict) -> str             # 기본은 템플릿(LLM 미사용, 속도)
def build_controllers(state, vs, human_ask_rule=True) -> dict[str, Controller]
```
속도 원칙: think 끔(`reasoning=False`), 발언 `SPEECH_MAX_CHARS` 이하, public_log_text 는 코어가 길이 제한. 한 낮/밤 사이클 LLM 호출 수를 최소화.

## 7. mafia_graph.py / 노트북 (D 구현)
- `mafia_graph.py`: LangGraph `StateGraph`. 노드: setup → night → morning(resolve+win check) → discussion → vote → execution(win check) → night ... ; conditional edge 로 엔딩 분기; 사람 입력은 `interrupt()`; `MemorySaver` checkpointer + thread_id. 게임 규칙 판정은 mafia_core 함수 재사용(중복 구현 금지).
- 노트북: `01_create_db.ipynb`(PDF→메타데이터→청킹 결과 표→Chroma, 필터 검색 시연), `02_agent_test.ipynb`(AI 1명 발언/투표/밤행동, 정체 노출 테스트, 정보 분리 확인), `03_mafia_game.ipynb`(GameEngine 버전: 참여/관전 모드, 종료 후 속마음 공개), `04_langgraph_mafia.ipynb`(그래프 시각화 + 실행 + 관전 N판 통계).
- `requirements.txt` 버전 고정(위 테스트 환경 버전) + `ipykernel`. `README.md`: 컨셉(정보 분리, 역할별 도구, 코드가 진실을 지킴, 안전장치), 구조도, 실행 방법, 실습실 PC 세팅 순서.

## 8. 테스트 기준 (완성도 증거)
- 코어: RandomController 8명으로 seed 0~199 자동 200판 → 예외 0, 모든 판 종료, 규칙 불변식(죽은 사람 행동/투표 없음, 마피아 2 경찰 1 의사 1 시민 4) 위반 0.
- 에이전트: 각 역할 1회씩 실제 qwen3:8b 호출 성공, 정체 노출 0, 반환 형식 오류 시 fallback 작동.
- 통합: 관전 모드 실제 LLM 1판 끝까지(시간 측정), 사이클당 평균 소요 시간 기록.

## 변경 기록
- 2026-11-17 v1 작성 (메인)
- 2026-11-17 v1.1 (B, 메인 승인): GameState.rng 추가. build_view(..., tie_candidates=None). safe_decide kind="speak" -> (Speech, fb), 그 외 (대상, fb). stats["llm_calls"]는 코어가 컨트롤러 호출마다 +1 (LLMPlayer는 증가 금지). GameEngine 단계 함수: announce_start, begin_night, collect_mafia_decision, collect_doctor_decision, collect_police_decision, apply_night, begin_day, discussion_order, run_discussion_turn, collect_votes(kind, tie_candidates), apply_votes, check_end. 이벤트 dict에 visibility(public/private/secret). 사람 발언은 정체노출 필터 미적용.
- 2026-11-18 v1.2 (메인): HUMAN_NAME "플레이어" → "서지우". 에이전트 안전장치 추가(확정 정보 모순 투표 시 후보 제한 재질문, 자기 이름 3인칭/없던 '어제'/날을 넘긴 반복 발언 감지, 두 시도 중 나은 발언 선택, 문장 단위 자르기, '자기 자신' 응답 처리). 그래프: State proposal 필드 복구, 사회자 오류 시 계속 진행. 통계 표시 "LLM 호출" → "AI 결정 요청". 노트북 01~04 전체 실행 출력 저장. tests/test_regression.py 추가.
- 2026-11-24 v1.4 (메인): mafia_qa.py 추가(GameQA: 끝난 게임 기록 -> 메모리 Chroma, create_agent + 도구 search_rulebook/search_game_log/player_summary, 도구 미사용 시 코드가 두 저장소 검색 후 재답변, 진행 중 게임 거부). 05_ask_agent.ipynb(사회자 Q&A + 해설 Q&A). 회귀 테스트 5번 추가.
- 2026-11-24 v1.3 (메인): build_controllers(..., show_thinking=False, printer=print) 추가(관전 모드 전용 사고 과정 출력, 사람 참여 시 자동 해제). 경찰 정체 공개 시점 판단, 경찰 주장 목록과 처형된 시민팀 투표자 계산, 몰아가기 감지, 첫날 '어제' 교정, 밤 행동 환각 감지, 자기 이름 3인칭 후처리, 속마음 복사 시 코드 요약. 03 노트북에 사고 과정 보기와 녹화된 판 재생(data/demo/) 추가.
