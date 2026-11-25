외부 자료 (External data)
=========================

해설 에이전트(`mafia_qa.GameQA`)의 `search_background` 도구가 검색하는 외부 지식이다.
룰북(우리 게임의 규칙)과 컬렉션을 분리해 `vector_db` 의 `mafia_background` 에 저장한다 (`rag.load_background_db()`).

| 파일 | 원문 | 판(revision) | 라이선스 |
|---|---|---|---|
| `wikipedia_ko_mafia.md` | 한국어 위키백과 「마피아 (게임)」 | 41513099 (2026-03-18) | CC BY-SA 4.0 |
| `wikipedia_en_mafia.md` | English Wikipedia "Mafia (party game)" | 1387169334 (2026-11-22) | CC BY-SA 4.0 |

- 저작자: 위키백과 기여자들 (Wikipedia contributors). 각 파일 머리에 원문 판 URL 이 있다.
- 변경 사항: 위키 문법, 표, 각주, 목차를 지우고 본문만 남겼다. 영어 문서는 역사, 진행 방식, 게임 이론,
  표준 조사/보호 역할, 승리 조건·동점 변형, 인공지능 절만 발췌했다.
- 이 폴더의 텍스트는 원문과 같은 CC BY-SA 4.0 으로 배포된다: https://creativecommons.org/licenses/by-sa/4.0/
- 수집일 2026-11-24. 한국어 문서의 최신 판(42753033)은 창안자와 연도가 사실과 다르게 훼손되어 있어
  (예: "2023년 학생 최진우가 창안") 훼손 직전의 정상 판을 사용했다. 영어 문서와 교차 확인함
  (1986년 모스크바 대학교 드미트리 다비도프 창안, 1997년 앤드루 플로트킨의 늑대인간 규칙).
