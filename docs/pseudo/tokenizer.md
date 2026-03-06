# 한국어 BBPE 토크나이저 스도코드 (자연어 상세 버전)

## 0) 목적
`docs/plans/tokenizer.md`의 확정 지침을 그대로 따르며,  
`data/korean_raw` + `novel24`에서 텍스트를 추출/정제한 뒤  
`ByteLevelBPETokenizer`로 `korean_bbpe_v1` 토크나이저를 생성한다.

핵심 원칙:
- holdout은 매 실행 랜덤 분할로 만들지 않는다.
- 지정된 고정 경로 holdout 파일(`holdout_fixed`)을 모든 실험의 공통 검증셋으로 사용한다.

---

## 1) 사용자 확인 질문(구현 잠금 전 필수)
아래 항목은 구현 전에 최종 잠금한다.

1. holdout 지표 임계값(pass/fail) 수치 고정 여부  
- 현재 기본값: 수치 임계값 미고정, 비교 지표 수집 중심

2. `novel24` 처리 단위  
- 현재 기본값: 라인 단위 유지

---

## 2) 고정 상수와 경로

사용자 설정값은 `src/tokenizer.py` 파일 상단의 User Settings 블록에서 직접 수정한다(터미널 인자 미사용).

1. 입력 루트
- `RAW_ROOT = data/korean_raw`

2. 출력 루트
- `OUT_DIR = data/tokenizers/korean_bbpe_v1`
- `OUT_TOKENIZER_JSON = data/tokenizers/korean_bbpe_v1/tokenizer.json`
- `OUT_VOCAB_JSON = data/tokenizers/korean_bbpe_v1/vocab.json`
- `OUT_MERGES_TXT = data/tokenizers/korean_bbpe_v1/merges.txt`
- `OUT_META_JSON = data/tokenizers/korean_bbpe_v1/meta.json`
- `OUT_HOLDOUT_METRICS_JSON = data/tokenizers/korean_bbpe_v1/holdout_metrics.json`

3. 고정 holdout 경로
- `HOLDOUT_FIXED_PATH = data/tokenizers/_benchmark/holdout_fixed.jsonl`
- `HOLDOUT_FIXED_META_PATH = data/tokenizers/_benchmark/holdout_fixed.meta.json`

4. 데이터 범위
- 포함 소스(12개):
  - `009.전문분야_기술과학_한국어 멀티세션 데이터`
  - `010.전문분야_사회과학_한국어 멀티세션 데이터`
  - `011.일상대화 한국어 멀티세션 데이터`
  - `019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터`
  - `020.주제별 텍스트 일상 대화 데이터`
  - `021.용도별 목적대화 데이터`
  - `023.국회 회의록 기반 지식검색 데이터`
  - `030.웹데이터 기반 한국어 말뭉치 데이터`
  - `045.지식검색 대화`
  - `046.공감형 대화`
  - `141.한국어 멀티세션 대화`
  - `novel24`
- 제외 소스: `_unzip_logs`, `gsm8k`

5. 파일 포맷/샘플링
- 허용 확장자: `json`, `txt`
- 학습 샘플링 상한: 소스별 `2GB`
- 학습 샘플링 seed: `42`

6. holdout 생성 상수(고정 파일 초기 1회 생성용)
- 목표 계산식: `target_bytes = min(int(filtered_text_bytes * 0.10), 100MB)`
- `filtered_text_bytes`: 화이트리스트 추출 + 길이필터 통과 텍스트 바이트 합
- 전체 목표량: 소스별 `target_bytes` 합(고정 1.2GB 아님)
- holdout 생성 seed: `42` (고정)

7. 토크나이저 하이퍼파라미터
- `model = ByteLevelBPETokenizer`
- `add_prefix_space=True`
- `trim_offsets=True`
- `vocab_size=32000`
- `min_frequency=10`
- 특수 토큰 7개:
  - `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|eot_id|>`, `<|pad|>`
- `UNK` 추가 금지

8. 정제 상수
- 길이 필터: `5 <= len(text) <= 20000`

9. 병렬 실행 상수
- `NUM_WORKERS = 7` (사용자 변수로 변경 가능)
- `DUCKDB_THREADS_PER_WORKER = 1`
- `PARALLEL_CHUNK_ROWS = 5000`
- `TEMP_SHARD_ROOT = data/tokenizers/_tmp`
- `CLEANUP_TEMP_ON_SUCCESS = true`
- `CLEANUP_TEMP_ON_FAILURE = false`
- `FAIL_FAST = true`

---

## 3) 전체 실행 오케스트레이션
`토크나이저_빌드_실행()` 절차는 아래와 같다.

1. 실행 시작 메타 기록
- run_id, 시작 시각, 입력/출력 경로, 파라미터 기록

2. 파일 인벤토리 수집
- 포함 소스에서 `json/txt` 파일 목록 수집
- 파일 메타(`source`, `path`, `ext`, `size_bytes`) 생성

3. 소스별 2GB 샘플링(학습용)
- `hash(path || seed)` 기반 안정 셔플
- 소스별 누적 2GB까지 선택

4. 병렬 텍스트 추출(결정적 샤딩)
- 선택 파일을 `hash(path || seed) % NUM_WORKERS`로 분배한다.
- 워커별로 소스 추출 규칙을 적용해 문자열을 추출/길이필터한다.
- 워커별 JSONL shard(`{\"text\": ...}`)를 생성한다.

5. DuckDB 병합 + 중복 제거
- 워커 shard를 DuckDB `filtered_texts`로 병합 적재한다.
- `SELECT DISTINCT text` 적용
- 결과를 `dedup_texts`로 저장

6. 고정 holdout 로드 및 검증
- `HOLDOUT_FIXED_PATH` 존재 확인
- `HOLDOUT_FIXED_META_PATH` 존재 확인
- `HOLDOUT_FIXED_PATH`의 SHA256 계산
- meta의 SHA256과 일치하지 않으면 즉시 중단

7. 학습 데이터 누수 제거
- `dedup_texts`에서 holdout 문자열 완전 일치 항목 제거
- 결과를 `train_texts`로 저장

8. BBPE 학습
- `train_texts`로 학습 수행
- 특수 토큰 7개 포함

9. 고정 holdout 평가
- `HOLDOUT_FIXED_PATH` 텍스트 encode/decode
- 토큰 길이/round-trip 지표 계산

10. 산출물 저장
- tokenizer/vocab/merges 저장
- `meta.json`, `holdout_metrics.json` 저장

11. 종료 검증
- 산출물 존재, 스펙 일치, holdout 해시 일치 확인

---

## 4) 파일 인벤토리 스도코드
`파일_인벤토리_수집()` 절차:

1. 포함 소스 목록 순회
2. 각 소스에서 `json/txt` 파일 수집
3. 각 파일의 `source/path/ext/size_bytes` 기록
4. DuckDB 테이블 `file_inventory` 등록

---

## 5) 소스별 2GB 샘플링 스도코드(학습용)
`소스별_샘플링(file_inventory)` 절차:

1. 소스 단위 그룹핑
2. `stable_rank = hash(path || '42')` 계산
3. `stable_rank` 정렬 후 누적 바이트 계산
4. 소스 합계가 2GB 이하면 전체 선택
5. 소스 합계가 2GB 초과면 상한까지 선택
6. 결과를 `sampled_files`로 저장

---

## 6) 고정 holdout 파일 초기 생성(1회성) 스도코드
`고정_holdout_초기생성()` 절차:

1. 목적
- `holdout_fixed.jsonl`를 최초 1회 생성해 공용 비교 기준으로 고정한다.

2. 생성 대상 소스
- 학습과 동일한 12개 소스를 사용한다(`_unzip_logs`, `gsm8k` 제외).

3. 병렬 추출 + 소스별 비율 기반 샘플링 규칙
- 각 소스 파일을 결정적으로 워커에 분배해 병렬 추출/길이필터를 수행한다.
- 워커가 생성한 소스별 shard를 DuckDB에 병합 적재한다.
- 소스별로 `SELECT DISTINCT text`를 수행한 뒤
- 소스별 목표치 `min(filtered_text_bytes*10%, 100MB)`를 계산한다.
- 총합은 소스별 목표치 합으로 정해진다.
- 추출 결과가 0건인 파일은 자동 스킵하고 source별 0건 파일 수를 기록한다.

4. 샘플링 방식
- 소스별 텍스트에 `stable_rank = hash(text || '42')`를 계산한다.
- `stable_rank` 오름차순으로 누적하며 소스 목표 용량(`target_bytes`)까지 선택한다.
- 선택 단위는 문자열 row 단위로만 처리한다.
- 목표 바이트를 넘기게 만드는 마지막 row는 포함한다.

5. 파일 저장 형식
- `holdout_fixed.jsonl`를 UTF-8 JSONL로 저장한다.
- 각 줄은 `{"source": "...", "text": "..."}` 형식으로 기록한다.
- tokenizer 평가는 JSONL의 `text` 필드만 토크나이즈한다.

6. 메타 저장
- `holdout_fixed.meta.json`에 아래를 저장한다.
  - 생성 시각
  - 소스 목록(12개)
  - 소스별 `filtered_text_bytes`, `target_bytes`, `actual_bytes`
  - 소스별 `with_text_files`, `zero_text_files`, `underfill`
  - 소스별 실제 용량/row 수
  - 전체 실제 용량/row 수
  - 파일 SHA256

7. 고정화 규칙
- 파일 생성 후 내용 수정 금지
- 일반 학습 실행에서는 자동 재생성 금지
- 파일 누락 시 학습 실행을 실패 처리하고 초기 생성 절차를 먼저 수행한다.

8. 임시 shard 정리 규칙
- 성공 시 temp shard 디렉토리를 자동 삭제한다.
- 실패 시 기본값은 shard를 남겨 원인 분석에 사용한다.

---

## 7) 텍스트 추출 오케스트레이션 스도코드
`텍스트_추출(file_meta)` 절차:

1. 확장자 분기
- `json`이면 JSON 추출기
- `txt`이면 TXT 추출기

2. 소스 분기
- `009/010/011/141`: 멀티세션 추출기
- `019`: 법률 조항 추출기
- `020/021`: 목적/일상 대화 추출기
- `023`: 국회 QnA 추출기
- `030`: 웹 말뭉치 추출기
- `045/046`: utterances 추출기
- `novel24`: TXT 추출기

3. 반환 형식
- `List[str]`
- 빈 문자열/None 제외

---

## 8) 소스별 JSON 추출 스도코드

1. `009/010/011/141`
- `sessionInfo[*].dialog[*].utterance`

2. `019`
- `clauseArticle[*]`
- `ftcCnclsns` 내 문자열
- `illdcssBasiss` 내 문자열

3. `020/021`
- 우선 `info[*].annotations.lines[*].norm_text`
- 없으면 `info[*].annotations.text`

4. `023`
- `LAB_*` 파일에서만 추출:
  - `context`
  - `question.comment`
  - `answer.comment`
- `SLAB_*` 파일은 추출 대상에서 제외

5. `030`
- `named_entity[*].title[*].sentence`
- `named_entity[*].subtitle[*].sentence`
- `named_entity[*].content[*].sentence`

6. `045/046`
- `utterances[*].text`

7. 메타 문자열 제외
- `id`, `url`, `filename`, `tag` 등은 추출 금지

---

## 9) TXT 추출 스도코드
`TXT_텍스트_추출(path, source)` 절차:

1. UTF-8로 파일 읽기
2. 라인 단위 분할
3. 양끝 공백 정리
4. 빈 줄 제거
5. `novel24`는 라인 단위 유지

---

## 10) 정제/중복 제거 스도코드
`텍스트_정제_및_중복제거(raw_texts)` 절차:

1. DuckDB 적재
- `filtered_texts(text)`

2. 중복 제거
- `SELECT DISTINCT text FROM filtered_texts`
- 완전 일치 문자열만 제거

3. 결과 저장
- `dedup_texts`

---

## 11) 학습 데이터 누수 제거 스도코드
`학습셋_생성(dedup_texts, holdout_fixed_texts)` 절차:

1. holdout 텍스트를 DuckDB 테이블로 적재
- `holdout_fixed(text)`

2. 누수 제거 SQL 실행
- `SELECT d.text FROM dedup_texts d LEFT ANTI JOIN holdout_fixed h ON d.text = h.text`

3. 결과를 `train_texts`로 저장
4. 제거 건수/잔여 건수 집계 기록

---

## 12) BBPE 학습 스도코드
`토크나이저_학습(train_texts)` 절차:

1. `ByteLevelBPETokenizer(add_prefix_space=True, trim_offsets=True)` 초기화
2. 학습 인자 적용
- `vocab_size=32000`
- `min_frequency=10`
- `special_tokens=[7개 고정 토큰]`
3. `train_texts`로 학습 수행
4. 무결성 확인
- vocab size 32000
- 특수 토큰 7개 존재
- `UNK` 미포함

---

## 13) 고정 holdout 평가 스도코드
`holdout_평가(tokenizer, holdout_fixed_rows)` 절차:

1. `holdout_fixed_rows`에서 `source`, `text`를 분리한다.
2. `text`를 encode한다.
3. 토큰 길이 분포를 수집한다.
4. decode 후 prefix-space aware round-trip 성공률 계산 (`decoded == text` 또는 `decoded == " " + text`)
5. 지표 계산
- `mean_tokens_per_text`
- `mean_tokens_per_char`
- `token_length_p50`
- `token_length_p95`
- `token_length_p99`
- `round_trip_success_rate`
6. 전체/source별 집계 생성
7. 결과를 `holdout_metrics.json`으로 반환

---

## 14) 산출물 저장 스도코드
`산출물_저장(tokenizer, stats, metrics, holdout_sha256)` 절차:

1. `OUT_DIR` 생성
2. `tokenizer.json`, `vocab.json`, `merges.txt` 저장
3. `meta.json` 저장
- 실행 메타
- 샘플링/정제/중복/누수제거 통계
- 사용 holdout 경로
- 사용 holdout SHA256
4. `holdout_metrics.json` 저장
- 계산 지표
- 평가 holdout SHA256

---

## 15) 완료 검증 체크리스트

1. 산출물 5개 존재
2. vocab size 32000 일치
3. 특수 토큰 7개 존재
4. `UNK` 미포함
5. holdout SHA256 일치
6. 학습셋-검증셋 완전 일치 누수 0건
7. holdout 지표가 전체/source별로 기록됨
8. holdout 소스별 `target_bytes/actual_bytes/underfill`이 메타에 기록됨
9. source별 `with_text_files/zero_text_files`가 메타에 기록됨

---

## 16) 실패 처리 정책

1. 고정 holdout 파일 누락
- 즉시 중단
- “고정 holdout 초기 생성 절차 필요” 메시지 기록

2. holdout 해시 불일치
- 즉시 중단
- 예상 해시/실제 해시를 로그에 남김

3. 추출/파싱 실패
- 워커 1개라도 실패하면 즉시 전체 중단(`FAIL_FAST`)
- 실패 shard 경로/파일 경로를 로그로 남김
- 단, JSON 파싱은 성공했지만 추출 0건인 파일은 실패가 아니라 스킵 대상으로 집계한다.

4. 학습 실패
- 파라미터/입력 건수/예외 로그 저장 후 중단

5. 저장 실패
- 부분 산출물 상태 기록 후 중단
