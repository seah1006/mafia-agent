# =====================================
# 08_Mafia_Agent 공통 설정
# =====================================
# 모든 모듈과 노트북은 이 파일의 경로/상수를 사용한다.
# 경로는 이 파일 위치 기준이므로 실습실 PC 어디에 복사해도 그대로 동작한다.

from pathlib import Path

# =====================================
# 1. 경로
# =====================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "마피아_룰북.pdf"
DB_DIR = BASE_DIR / "vector_db"
LOG_DIR = BASE_DIR / "logs"

# =====================================
# 2. 모델 (Ollama)
# =====================================
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "bge-m3"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLM_TEMPERATURE = 0.7      # 캐릭터마다 말투 차이를 주기 위해 0보다 높게
LLM_MAX_RETRIES = 2        # 형식 오류 시 재시도 횟수 (그 뒤에는 코드가 무작위 대체)

# =====================================
# 3. 게임 규칙
# =====================================
ROLES = ["mafia", "police", "doctor", "citizen"]
ROLE_KO = {"mafia": "마피아", "police": "경찰", "doctor": "의사", "citizen": "시민"}
ROLE_DISTRIBUTION = {"mafia": 2, "police": 1, "doctor": 1, "citizen": 4}   # 총 8명
TOTAL_PLAYERS = sum(ROLE_DISTRIBUTION.values())

# AI 캐릭터 8명 (PDF 캐릭터 프로필 페이지 순서와 같다)
CHARACTERS = [
    "강태호",  # 공격형: 직설적, 반말 섞인 말투
    "윤서진",  # 논리형: 차분한 존댓말, 근거 정리
    "박준수",  # 분위기 메이커: 농담, 가벼운 말투
    "최민지",  # 의심형: 질문 공세
    "정도현",  # 과묵형: 짧은 단답
    "한소율",  # 중재형: 부드러운 존댓말, 갈등 조정
    "오지환",  # 직감형: 감정적, 느낌으로 판단
    "임채원",  # 관찰형: 발언 기록과 투표 패턴 분석
]
# 사람 플레이어 이름. "플레이어"처럼 눈에 띄는 이름이면 마피아 AI 가 첫날 밤에 먼저 노리는 경향이 있어
# 평범한 이름을 쓴다. 본인 이름으로 바꿔도 된다(캐릭터 8명과 겹치지만 않으면 됨).
HUMAN_NAME = "서지우"

DISCUSSION_ROUNDS = 1      # 낮 토론 라운드 수
MAX_DAYS = 10              # 무한 루프 방지
SPEECH_MAX_CHARS = 150     # AI 발언 최대 길이 (속도와 가독성)
PUBLIC_LOG_MAX_CHARS = 2500  # PlayerView 에 넣는 공개 기록 최대 길이

# =====================================
# 4. PDF 페이지 메타데이터 (PyPDFLoader page, 0부터)
# =====================================
PAGE_META = {
    0: {"type": "rule", "role": "all", "name": "-", "title": "마피아 게임 룰북"},
    1: {"type": "rule", "role": "all", "name": "-", "title": "기본 규칙과 승리 조건"},
    2: {"type": "rule", "role": "all", "name": "-", "title": "직업별 능력"},
    3: {"type": "rule", "role": "all", "name": "-", "title": "진행 순서와 특수 규칙"},
    4: {"type": "strategy", "role": "mafia", "name": "-", "title": "마피아 전략 가이드"},
    5: {"type": "strategy", "role": "police", "name": "-", "title": "경찰 전략 가이드"},
    6: {"type": "strategy", "role": "doctor", "name": "-", "title": "의사 전략 가이드"},
    7: {"type": "strategy", "role": "citizen", "name": "-", "title": "시민 전략 가이드"},
}
for _i, _name in enumerate(CHARACTERS):
    PAGE_META[8 + _i] = {
        "type": "character",
        "role": "all",
        "name": _name,
        "title": f"캐릭터 프로필: {_name}",
    }
