# =====================================
# play.py : 터미널에서 마피아 에이전트를 한 번에 실행
# =====================================
# 사용법 (08_Mafia_Agent 폴더에서)
#   python play.py                 메뉴에서 고르기
#   python play.py watch           관전 모드: AI 8명 + 사고 과정 보기
#   python play.py watch --quiet   관전 모드: 사고 과정 없이 진행만
#   python play.py play            참여 모드: 나 1명 + AI 7명
#   python play.py replay          녹화된 실제 한 판 다시 보기 (LLM 호출 없음, 즉시)
#   python play.py ask             질문하기: 사회자(룰북) + 녹화된 판 해설 에이전트
#   옵션: --seed 11 (같은 배정 재현), --delay 0.05 (재생 속도)
import argparse
import json
import os
import random
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, os.getcwd())

import config  # noqa: E402

DEMO_TXT = config.DATA_DIR / "demo" / "demo_game.txt"
DEMO_LOG = config.DATA_DIR / "demo" / "demo_game.json"


# =====================================
# 1. 실행 전 점검: Ollama, 모델, 벡터 DB
# =====================================
def check_ollama():
    url = config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            names = [m["name"] for m in json.load(r)["models"]]
    except Exception:
        sys.exit("[오류] Ollama 에 연결할 수 없습니다. 시작 메뉴에서 Ollama 앱을 켠 뒤 다시 실행하세요.")
    missing = [m for m in (config.LLM_MODEL, config.EMBED_MODEL)
               if not any(n == m or n.split(":")[0] == m for n in names)]
    if missing:
        sys.exit("[오류] Ollama 에 모델이 없습니다: %s\n        ollama pull %s" % (", ".join(missing), missing[0]))


def load_vs():
    import rag
    try:
        vs = rag.load_db()
    except Exception:
        print("벡터 DB 가 없어서 룰북 PDF 로 새로 만듭니다 (처음 한 번, 약 15초)...")
        vs = rag.build_db()
    print("===== 룰북 벡터 DB 준비 완료 (Chunk %d개) =====" % vs._collection.count())
    return vs


# =====================================
# 2. 모드별 실행
# =====================================
def finish(state, vs, elapsed):
    import mafia_core as core
    print()
    print(core.reveal_report(state))
    path = core.save_log(state)
    st = state.stats
    print("\n로그 저장: %s" % path)
    print("소요 시간 %.0f초 | AI 결정 요청 %d | 재시도 %d | fallback %d | 정체 노출 차단 %d"
          % (elapsed, st["llm_calls"], st["retries"], st["fallbacks"], st["leak_blocked"]))
    if ask_yes("\n방금 끝난 판에 대해 해설 에이전트에게 질문할까요?"):
        import mafia_qa
        qa_loop(mafia_qa.GameQA.from_state(vs, state), vs)


def run_watch(seed, thinking):
    import mafia_core as core
    import mafia_agents as agents
    check_ollama()
    vs = load_vs()
    state = core.new_game(list(config.CHARACTERS), None, seed=seed)
    controllers = agents.build_controllers(state, vs, show_thinking=thinking)
    t0 = time.time()
    core.GameEngine(state, controllers).run()
    finish(state, vs, time.time() - t0)


def run_play(seed):
    import mafia_core as core
    import mafia_agents as agents
    check_ollama()
    vs = load_vs()
    print("\n당신의 이름은 '%s' 입니다. 발언은 그대로 입력, 투표/밤 행동은 번호로 고르세요." % config.HUMAN_NAME)
    print("'?질문' 을 입력하면 사회자가 룰북에서 찾아 답합니다. (예: ?의사는 자기 자신을 보호할 수 있나요?)\n")
    ai_names = random.Random(seed).sample(config.CHARACTERS, 7)
    state = core.new_game(ai_names, config.HUMAN_NAME, seed=seed)
    controllers = agents.build_controllers(state, vs)   # 사람은 HumanController (input 사용)
    t0 = time.time()
    core.GameEngine(state, controllers).run()
    finish(state, vs, time.time() - t0)


def run_replay(delay):
    meta = json.loads(DEMO_LOG.read_text(encoding="utf-8"))
    print("===== 녹화된 판: seed %s, 승자 %s, %s일차에 종료 =====\n" % (meta["seed"], meta["winner"], meta["day"]))
    for line in DEMO_TXT.read_text(encoding="utf-8").splitlines():
        print(line, flush=True)
        if delay:
            time.sleep(delay)


def run_ask():
    import mafia_qa
    check_ollama()
    vs = load_vs()
    print("녹화된 판(data/demo)을 해설 에이전트에 넣는 중...")
    qa_loop(mafia_qa.GameQA.from_file(vs, DEMO_LOG), vs)


def qa_loop(qa, vs):
    import mafia_agents as agents
    moderator = agents.Moderator(vs)
    print("\n===== 질문하기 =====")
    print("  이번 판 질문 : 그냥 입력      (예: 경찰은 첫날 밤에 누구를 조사했어?)")
    print("  규칙 질문    : ?로 시작       (예: ?투표에서 동점이 나오면 어떻게 돼?)")
    print("  끝내기       : 빈 줄에서 Enter\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            if q.startswith("?"):
                print("[사회자]", moderator.explain_rule(q[1:].strip()))
            else:
                qa.ask(q)
        except Exception as exc:   # LLM 오류가 나도 다음 질문은 계속 받는다
            print("[오류] 답변 중 문제가 생겼습니다: %s" % exc)
        print()


# =====================================
# 3. 메뉴
# =====================================
def ask_yes(msg):
    try:
        return input(msg + " (y/n) ").strip().lower() in ("y", "yes", "ㅛ", "네", "예")
    except (EOFError, KeyboardInterrupt):
        return False


MENU = [
    ("watch", "관전 모드   : AI 8명이 게임 + 사고 과정 보기 (한 판 4~10분)"),
    ("play", "참여 모드   : 나 1명 + AI 7명"),
    ("replay", "녹화 재생   : 미리 진행한 실제 한 판 다시 보기 (즉시)"),
    ("ask", "질문하기    : 사회자(룰북) + 해설 에이전트(녹화된 판)"),
]


def choose_mode():
    print("===== 마피아 에이전트 (qwen3:8b · LangChain · LangGraph) =====")
    for i, (_, desc) in enumerate(MENU, 1):
        print("  %d. %s" % (i, desc))
    print("  0. 종료")
    while True:
        try:
            c = input("번호를 고르세요> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if c == "0":
            return None
        if c.isdigit() and 1 <= int(c) <= len(MENU):
            return MENU[int(c) - 1][0]
        print("1~%d 중에서 골라 주세요." % len(MENU))


def main():
    p = argparse.ArgumentParser(description="터미널에서 마피아 에이전트 실행")
    p.add_argument("mode", nargs="?", choices=[m for m, _ in MENU], help="생략하면 메뉴가 나온다")
    p.add_argument("--seed", type=int, default=None, help="역할 배정 재현용 seed (관전 기본값: 무작위)")
    p.add_argument("--quiet", action="store_true", help="관전 모드에서 사고 과정 상자를 숨긴다")
    p.add_argument("--delay", type=float, default=0.03, help="녹화 재생 줄 간격(초)")
    args = p.parse_args()

    mode = args.mode or choose_mode()
    if mode == "watch":
        run_watch(args.seed, not args.quiet)
    elif mode == "play":
        run_play(args.seed)
    elif mode == "replay":
        run_replay(args.delay)
    elif mode == "ask":
        run_ask()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단했습니다.")
