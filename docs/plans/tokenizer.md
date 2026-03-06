# Korean BBPE 토크나이저 생성 계획

## 0. 문서 목적
`data/korean_raw`의 한국어 데이터와 `novel24`를 사용해 한국어 토크나이저를 생성하기 위한 실행 계획을 정의한다.  
본 문서는 구현 기준(입력 범위, 추출 규칙, 샘플링, 중복 제거, 학습 파라미터, 고정 holdout 검증, 수용 기준)을 고정하며, 코드 변경 범위는 포함하지 않는다.

---

## 1. 확정 결정사항

1. 학습 대상 데이터 범위
- 포함: `009/010/011/019/020/021/023/030/045/046/141` + `novel24` (총 12개 소스)
- 제외: `_unzip_logs`, `gsm8k`

2. 입력 포맷
- `json`, `txt` 사용
- `novel24`는 `txt`만 사용

3. 학습용 샘플링 정책
- 소스 폴더별 최대 `2GB` 랜덤 샘플
- 폴더 크기가 2GB 미만이면 전체 사용
- 재현성 seed: `42`

4. 토크나이저 사양
- 라이브러리: `tokenizers`
- 알고리즘: `ByteLevelBPETokenizer`
- 옵션: `add_prefix_space=True`, `trim_offsets=True`
- `vocab_size=32000`, `min_frequency=10`

5. 특수 토큰 정책
- `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|eot_id|>`, `<|pad|>`
- `UNK` 토큰은 추가하지 않음

6. 중복 제거 정책
- DuckDB `DISTINCT` 사용
- 문자열이 100% 동일한 경우에만 중복 제거

7. 고정 holdout 검증 정책
- holdout은 랜덤 분할로 생성하지 않고 고정 경로 파일을 사용한다.
- 고정 경로:
  - `HOLDOUT_FIXED_PATH = data/tokenizers/_benchmark/holdout_fixed.jsonl`
  - `HOLDOUT_FIXED_META_PATH = data/tokenizers/_benchmark/holdout_fixed.meta.json`
- holdout 구성 규칙(초기 1회 생성 후 고정):
  - 대상 소스: 학습과 동일한 12개 소스(`_unzip_logs`, `gsm8k` 제외)
  - 소스별 목표 텍스트량 계산식:
    - `target_bytes = min(int(filtered_text_bytes * 0.10), 100MB)`
    - `filtered_text_bytes`는 화이트리스트 추출 + 길이필터 통과 텍스트 바이트 합
  - 전체 목표 텍스트량은 소스별 `target_bytes` 합으로 결정된다(고정 1.2GB 아님).
- 어떤 데이터/어떤 방식으로 학습한 토크나이저든 검증 시 동일 `holdout_fixed`를 사용한다.
- 학습 데이터에 holdout 문장이 포함되면(문자열 완전 일치 기준) 해당 문장을 학습 데이터에서 제거한다.
- 추출 0건 파일은 자동 스킵하며, 소스별 `zero_text_files` 집계를 메타에 기록한다.

8. 설정 파일 정책
- `configs` 폴더는 수정하지 않음
- 실행 제어는 터미널 인자 대신 `src/tokenizer.py` 상단 사용자 설정 변수(`RUN_MODE`, 경로, 용량, seed 등)로 수행한다.

9. 병렬 실행 정책(확정)
- 기본 워커 수: `NUM_WORKERS=7` (사용자 변수로 변경 가능, 상한 없음)
- 적용 범위: `build`, `init_holdout` 모두 멀티프로세싱 적용
- 샤딩 방식: `hash(path || seed)` 기반 결정적 샤딩
- 실패 정책: `FAIL_FAST=True` (워커 1개 실패 시 전체 중단)
- DuckDB 스레드: `DUCKDB_THREADS_PER_WORKER=1`
- 중간산출물: 임시 shard 파일 사용 후 성공 시 자동 정리(`CLEANUP_TEMP_ON_SUCCESS=True`)

---

## 2. 산출물 경로

출력 루트:
- `data/tokenizers/korean_bbpe_v1/`

필수 산출물:
1. `tokenizer.json`
2. `vocab.json`
3. `merges.txt`
4. `meta.json`
5. `holdout_metrics.json`

공용 고정 검증셋 자산:
1. `data/tokenizers/_benchmark/holdout_fixed.jsonl`
2. `data/tokenizers/_benchmark/holdout_fixed.meta.json`

`meta.json` 포함 항목:
- 실행 시각
- 사용한 파라미터(vocab_size, min_frequency, seed)
- 소스별 샘플 파일 수/바이트 수
- 추출 건수, 필터 후 건수, distinct 후 건수
- 학습 데이터 최종 건수(holdout 중복 제거 후)
- 사용한 holdout 경로와 holdout 파일 SHA256
- 워커 추출 통계(`with_text_files`, `zero_text_files`, `filtered_text_bytes`)

`holdout_metrics.json` 포함 항목:
- 전체/source별 문장 수
- 평균 문자 길이
- 평균 토큰 길이
- 문자당 토큰 수(`tokens_per_char`)
- 문장당 토큰 길이 p50/p95/p99
- encode/decode round-trip 성공률 (prefix-space aware: `decoded == text` 또는 `decoded == " " + text`)
- 평가에 사용한 holdout 파일 SHA256

---

## 3. 전체 파이프라인

1. 파일 인벤토리 수집
- 대상 소스별 `json/txt` 파일 목록 수집
- 파일 크기 기준 메타 생성

2. 소스별 2GB 샘플링(학습용)
- 파일 목록을 seed 42로 안정 셔플
- 누적 바이트가 2GB에 도달할 때까지 선택

3. 병렬 텍스트 추출
- 파일을 결정적 샤딩으로 워커에 분배한다.
- 각 워커가 데이터셋 구조별 화이트리스트 키에서 문자열을 추출/필터링한다.
- 워커별 JSONL shard(`text` 필드)를 생성한다.

4. DuckDB 병합 + 품질 필터 반영
- 워커 shard를 DuckDB에 적재한다.
- 길이 기준: `5 <= length(text) <= 20000` 적용 결과를 기준으로 진행한다.

5. 중복 제거
- `SELECT DISTINCT text ...`로 완전 일치 문자열만 제거

6. 고정 holdout 로드
- `HOLDOUT_FIXED_PATH`를 로드한다.
- 파일이 없으면 중단한다(비교 가능성 보장을 위해 임의 holdout 생성 금지).
- `HOLDOUT_FIXED_META_PATH`의 해시와 실제 파일 해시를 검증한다.

7. 학습 데이터 누수 제거
- dedup 텍스트에서 holdout 완전일치 문자열을 제거해 `train_texts`를 만든다.

8. BBPE 학습
- `train_texts`로 토크나이저 학습
- 특수 토큰 7개 고정 추가

9. holdout 평가
- `HOLDOUT_FIXED_PATH` 텍스트를 encode/decode해 안정성 지표 계산
- 지표를 `holdout_metrics.json`으로 저장

10. 저장 및 메타 기록
- tokenizer/vocab/merges 저장
- 실행 통계 및 품질 지표 저장
- 병렬 실행 설정/워커 shard 통계 저장

---

## 4. 데이터셋별 텍스트 추출 규칙

1. `009/010/011/141`
- `sessionInfo[*].dialog[*].utterance`

2. `019`
- `clauseArticle[*]`
- `ftcCnclsns` 내 문자열 필드
- `illdcssBasiss` 내 문자열 필드

3. `020/021`
- 우선: `info[*].annotations.lines[*].norm_text`
- 대체: `info[*].annotations.text`

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

7. `novel24`
- `txt` 파일 라인 단위 문자열

메타데이터 문자열(`id`, `url`, `filename`, `tag`)은 학습 텍스트에 포함하지 않는다.

---

## 5. 고정 holdout 평가 정의

1. 목적
- 학습 미사용 데이터에서 토크나이저 일반화 품질을 점검한다.
- 실험/버전 간 비교에서 평가 데이터 변동을 제거한다.

2. 고정 holdout 생성/관리 원칙
- `holdout_fixed.jsonl`는 초기 1회 생성 후 고정한다.
- 소스별 목표치는 `min(filtered_text_bytes * 10%, 100MB)`로 계산한다.
- 전체 holdout 목표치는 소스별 목표치 합으로 결정된다.
- dedup 이후 가용 텍스트 부족 시 `underfill`을 허용하고 메타에 기록한다.
- 파일 포맷은 JSONL이며, 각 줄은 `{"source": "...", "text": "..."}` 구조를 사용한다.
- 파일 내용 변경 금지. 변경이 필요한 경우 파일명을 바꾸지 않고 내용을 바꾸는 대신, 별도 합의 후 메타 해시를 갱신하고 변경 이력을 남긴다.

3. 산출 지표
- `mean_tokens_per_text`
- `mean_tokens_per_char`
- `token_length_p50`
- `token_length_p95`
- `token_length_p99`
- `round_trip_success_rate`

4. 해석 기준
- `tokens_per_char`가 낮을수록 압축 효율이 좋다.
- `p95/p99`가 지나치게 크면 장문 처리 비효율 가능성이 높다.
- round-trip 성공률(`decoded == text` 또는 `decoded == " " + text` 기준)은 `100%`를 목표로 한다.
- 지표 임계값은 현재 고정하지 않고 버전 간 비교 지표로 수집한다.

5. 리포트 단위
- 전체 집계 1개
- source별 집계 N개

---

## 6. 검증 및 수용 기준

1. 산출물 검증
- `tokenizer.json`, `vocab.json`, `merges.txt`, `meta.json`, `holdout_metrics.json` 존재

2. 토크나이저 스펙 검증
- vocab 크기 32000 일치
- 특수 토큰 7개 모두 존재
- `UNK` 토큰 미포함

3. 데이터 처리 검증
- 소스별 샘플링/필터/중복제거 건수가 `meta.json`에 기록됨
- `HOLDOUT_FIXED_PATH` 해시가 `meta.json`/`holdout_metrics.json`과 일치
- 학습 데이터에서 holdout 완전일치 문자열이 제거됨
- 병렬 설정(`num_workers`, `duckdb_threads_per_worker`, `parallel_chunk_rows`)이 메타에 기록됨
- 워커 수가 달라도 동일 설정/입력에서는 holdout SHA와 주요 카운트가 일관됨
- holdout 메타에 소스별 `filtered_text_bytes`, `target_bytes`, `actual_bytes`, `with_text_files`, `zero_text_files`, `underfill`이 기록됨

4. 동작 검증
- 한국어/영문/숫자/기호 혼합 문자열 인코딩/디코딩이 예외 없이 동작
- 고정 holdout에서 round-trip 성공률 100%

---

## 7. 범위 밖(Out of Scope)

1. `configs/*` 경로 및 내용 수정
2. 모델 학습 파이프라인 변경
3. 전처리 데이터셋(`korean_processed`) 구조 변경

---

## 8. 리스크 및 대응

1. 리스크: `json`/`txt` 동시 사용 시 중복 텍스트 다량 유입  
대응: 완전 일치 중복 제거(`DISTINCT`)를 필수 적용

2. 리스크: 데이터셋별 구조 차이로 잘못된 필드 추출 가능  
대응: 화이트리스트 키 기반 추출 + 소스별 샘플 검증 로그 저장

3. 리스크: 긴 법률/회의록 텍스트로 학습 편향 발생 가능  
대응: 길이 필터 적용 + 소스별 샘플 바이트 상한(2GB)

4. 리스크: 평가셋이 실행마다 달라져 지표 비교가 흔들릴 수 있음  
대응: 고정 경로 holdout(`holdout_fixed.jsonl`) 강제 + SHA256 검증 필수화

5. 리스크: 일부 소스에서 추출 텍스트가 작아 목표치 미달(underfill) 가능  
대응: `10% + 100MB cap` 정책으로 목표를 현실화하고 source별 underfill을 메타에 기록

6. 리스크: 라벨링 구조 내 메타성 JSON 다수로 추출 0건 파일이 많아질 수 있음  
대응: 0건 파일 자동 스킵 + source별 `zero_text_files`/`with_text_files` 집계 기록
