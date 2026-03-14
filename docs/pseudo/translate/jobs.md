# 번역 작업 등록 의사코드

## 대상 파일
이 문서는 `src/translate/jobs.py`를 작성하기 위한 의사코드다.

이 파일의 책임:
1. 사용자 설정 상수 정의
2. `TranslationJob` 타입 정의
3. 활성 job 목록 반환

---

## 상단 사용자 설정 블록

파일 맨 위에 사용자가 직접 수정할 수 있는 상수를 둔다.

1. 모델 설정
- `MODEL_ID = "google/translategemma-4b-it"` 고정
- `DTYPE = "bfloat16"` 고정
- `DEVICE = "cuda"` 고정
- `ATTN_IMPLEMENTATION = "sdpa"` 고정
- `FLASH_BACKEND_NAME = "torch_builtin_sdpa_flash"` 고정
- `FALLBACK_ATTENTION_BACKEND = "sdpa_auto"` 고정
- `PREFER_TORCH_FLASH_ATTN = True`
- `ALLOW_FLASH_FALLBACK = True`
- `REQUIRE_HF_AUTH = True`

2. generation 설정
- `MAX_INPUT_TOKENS = 2048`
- `MIN_RESERVED_OUTPUT_TOKENS = 256`
- `MAX_OUTPUT_TOKENS_PROBLEM`
- `MAX_OUTPUT_TOKENS_THINKING`
- `MAX_OUTPUT_TOKENS_SOLUTION`
- `TEMPERATURE = 0.0`
- `DO_SAMPLE = False`
- `RETRY_LIMIT`

3. 실행 설정
- `ROW_BATCH_SIZE`
- `TARGET_FIELDS_PER_ROW = 3`
- `MAX_REQUESTS_PER_BATCH = ROW_BATCH_SIZE * TARGET_FIELDS_PER_ROW`
- `WRITE_META_EVERY_ROW = True`
- `LOG_EVERY_N_ROWS`
- `REPORT_SUFFIX = ".report.json"`
- `META_SUFFIX = ".meta.json"`

4. 언어 코드 설정
- 현재 기본 job은 `source_lang_code = "en"`, `target_lang_code = "ko"`로 고정한다.
- future dataset이 다른 언어를 쓰면 job 단위로만 바꾼다.

---

## TranslationJob 타입

`TranslationJob`은 dataclass로 정의한다.

필드:
1. `job_name: str`
2. `input_path: Path`
3. `output_path: Path`
4. `target_fields: tuple[str, ...]`
5. `source_lang_code: str`
6. `target_lang_code: str`
7. `enabled: bool`

추가 메서드:
1. `meta_path()`:
- `output_path`와 같은 위치에 `.meta.json` 확장자를 붙인 경로 반환

2. `report_path()`:
- `output_path`와 같은 위치에 `.report.json` 확장자를 붙인 경로 반환

3. `validate()`:
- `target_fields`가 비어 있지 않은지 확인
- 현재 계획에서는 `problem`, `thinking`, `solution` 외 값이 들어오면 예외 발생
- `input_path`와 `output_path`가 같은 파일이 아니어야 함
- `source_lang_code`, `target_lang_code`가 비어 있으면 안 됨

---

## job 목록 구성

`build_translation_jobs()` 함수는 아래 순서로 동작한다.

1. 빈 list 생성
2. 현재 1차 대상 job 생성
3. job validate 수행
4. list에 추가
5. list 반환

현재 기본 job 값:
1. `job_name = "nohurry_opus_reasoning_ko"`
2. `input_path = data/english_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/distilled_corpus_400k_with_cot-filtered.jsonl`
3. `output_path = data/korean_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/distilled_corpus_400k_with_cot-filtered.jsonl`
4. `target_fields = ("problem", "thinking", "solution")`
5. `source_lang_code = "en"`
6. `target_lang_code = "ko"`
7. `enabled = True`

---

## 활성 job 조회

`get_enabled_jobs()` 함수는 아래처럼 동작한다.

1. `build_translation_jobs()` 호출
2. `enabled is True`인 job만 필터링
3. `job_name` 기준으로 안정 정렬
4. 결과 반환

---

## future dataset 추가 규칙

새 데이터셋이 추가될 때는 아래만 수행한다.

1. `TranslationJob(...)` entry를 하나 더 만든다.
2. `target_fields`를 해당 데이터셋 구조에 맞게 지정한다.
3. `source_lang_code`, `target_lang_code`를 해당 데이터셋 언어 방향에 맞게 지정한다.
4. 공통 엔진과 runner는 수정하지 않는다.
