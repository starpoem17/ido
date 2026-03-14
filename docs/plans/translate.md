# Korean Raw 번역 파이프라인 계획

## 0. 문서 목적
`docs/personal/translate.md`를 기준으로 영어 reasoning JSONL 데이터를 한국어 JSONL로 번역하는 실행 계획을 고정한다.  
현재 1차 대상은 `nohurry-Opus-4.6-Reasoning-3000x-filtered/distilled_corpus_400k_with_cot-filtered.jsonl` 1개이며, 이후 번역 대상 데이터가 늘어나더라도 공통 엔진은 유지하고 작업 등록만 추가하는 구조를 목표로 한다.

핵심 원칙:
1. 입력과 출력은 모두 JSONL 스트리밍 처리로 구현한다.
2. 원본 row 순서와 비번역 필드는 그대로 유지한다.
3. 번역 대상 필드는 작업별로 명시하고, 현재 작업은 `problem`, `thinking`, `solution`만 번역한다.
4. 모델 추론은 단일 프로세스, 단일 GPU, 단일 모델 인스턴스로만 수행한다.
5. 모델은 `bf16`으로 로드한다.
6. TranslateGemma 공식 사용법에 맞춰 `AutoProcessor` + `AutoModelForImageTextToText` + `apply_chat_template()`를 사용한다.
7. attention backend는 외부 `flash_attn` 패키지가 아니라 PyTorch 내장 SDPA flash backend를 우선 사용한다.
8. 각 row마다 `problem`, `thinking`, `solution` 원문을 독립 입력 3개로 만들고, special token과 turn 구조는 `apply_chat_template()`와 tokenizer에 맡긴 채 여러 row와 함께 배치 추론한다.
9. 행 간 KV cache를 절대 재사용하지 않는다.
10. 장시간 작업을 고려해 기본 재시작 방식은 `이어쓰기/스킵`으로 한다.
11. gated repo 접근은 실행 시작 전에 인증과 접근 가능 여부를 먼저 확인한다.

---

## 1. 현재 확정 대상

### 1.1 입력/출력 경로
- 입력:
  - `data/english_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/distilled_corpus_400k_with_cot-filtered.jsonl`
- 출력:
  - `data/korean_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/distilled_corpus_400k_with_cot-filtered.jsonl`

### 1.2 번역 대상 필드
- `problem`
- `thinking`
- `solution`

### 1.3 언어 코드
- `source_lang_code = "en"`
- `target_lang_code = "ko"`

강제 규칙:
- 위 3개 필드만 영어에서 한국어로 번역한다.
- `id`, `difficulty`, `category`, `timestamp`, `hash` 등 나머지 필드는 원본 그대로 유지한다.
- JSONL 형식은 유지하고, 각 줄은 원본과 동일한 row 구조를 유지해야 한다.
- 출력 row 순서는 입력 row 순서와 완전히 동일해야 한다.

---

## 2. 공통 구조 계획

### 2.1 진입점
- 실행 진입점은 `uv run python -m src.translate.runner`로 고정한다.
- 번역 로직은 `src/translate` 아래 공통 엔진 중심으로 구성한다.

### 2.2 작업 등록 방식
- 각 번역 대상은 공통 타입 `TranslationJob`으로 정의한다.
- 새 데이터셋이 추가되면 별도 스크립트를 복제하지 않고 작업 목록에 `TranslationJob` entry만 추가한다.
- `TranslationJob`은 최소 아래 정보를 가진다.
  - `job_name`
  - `input_path`
  - `output_path`
  - `target_fields`
  - `source_lang_code`
  - `target_lang_code`
  - `enabled`

### 2.3 설정 방식
- 모델명, row 배치 크기, 최대 입력 토큰 수, 최대 출력 토큰 수, 재시도 횟수, 로그 주기, 재시작 정책, 인증 요구 여부는 CLI 인자가 아니라 코드 상단 상수로 둔다.
- 사용자 규칙에 따라 추후 수정 가능성이 있는 값은 파일 상단에 모아 배치한다.
- row 배치 크기는 `ROW_BATCH_SIZE` 상수로 직접 수정한다.

---

## 3. 모델 로드 및 추론 정책

### 3.1 모델 로드
- 번역 모델은 `google/translategemma-4b-it`를 사용한다.
- 모델은 단일 GPU에 `bf16`으로 1개만 올린다.
- 모델은 job 시작 시 1회만 로드하고, enabled job을 순차 처리한다.
- `attn_implementation="sdpa"`를 사용한다.
- `torch.compile`은 사용하지 않는다.
- `torch.cuda.empty_cache()`를 반복 호출하지 않고 PyTorch allocator가 메모리를 관리하도록 둔다.

### 3.2 공식 입력 형식
- TranslateGemma 공식 문서에 맞춰 입력은 수동 시스템 프롬프트가 아니라 chat template 메시지로 생성한다.
- 각 호출은 아래 형태의 단일 `user` 메시지를 사용한다.
  - `role = "user"`
  - `content = [{"type": "text", "source_lang_code": "en", "target_lang_code": "ko", "text": ...}]`
- `content[].text`에는 번역할 원문만 넣고, `<|bos|>`, `<|eos|>` 같은 문자열을 직접 붙이지 않는다.
- 입력 토큰 길이 계산도 raw prompt 문자열 기준이 아니라 `processor.apply_chat_template(..., tokenize=True)` 결과의 `input_ids` 길이 기준으로 판단한다.

### 3.3 인증 정책
- gated repo 사용 전 Hugging Face 토큰 존재 여부를 먼저 확인한다.
- 토큰이 없으면 `hf auth login` 또는 `HF_TOKEN` 설정이 필요하다는 명확한 에러를 낸다.
- 토큰이 있어도 모델 접근 승인이 없으면 실행 시작 단계에서 중단한다.

### 3.4 병렬화 정책
- 모델 추론 시 멀티프로세싱을 사용하지 않는다.
- GPU 메모리 압박을 고려하여 다중 프로세스 추론, 다중 모델 인스턴스, 동시 다중 배치 추론은 기본 계획에서 제외한다.
- CPU는 파일 읽기와 JSON 직렬화 정도만 담당하며, 추론 파이프라인을 분산하지 않는다.

### 3.5 attention backend 정책
- 외부 `flash_attn` 패키지는 사용하지 않는다.
- 모델 로드시 PyTorch built-in flash attention 사용 가능 여부를 1회 확인한다.
- flash가 실제 runtime 입력에서 가능하면 generation 호출은 PyTorch SDPA flash-only 컨텍스트 안에서 실행한다.
- flash가 불가능하면 같은 job 전체를 일반 SDPA 경로로 고정하고 math, mem-efficient, cuDNN 중 가능한 커널로 진행한다.
- 번역 파이프라인에서는 flash 실패 시 job을 중단하지 않고 일반 SDPA 연산으로 폴백한다.

### 3.6 디코딩 정책
- 디코딩은 결정론적 설정을 기본으로 한다.
- generation 결과는 입력 길이 이후 토큰만 잘라 decode한다.
- reasoning 데이터이므로 요약하지 않고 의미를 최대한 유지하는 번역을 원칙으로 한다.

---

## 4. 필드별 배치 번역 정책

### 4.1 호출 단위
- 한 row에서 `problem`, `thinking`, `solution`을 각각 독립 입력으로 만든다.
- 각 입력은 field 원문 그대로 유지하고, BOS/EOS 및 turn token은 tokenizer/chat template가 자동으로 구성한다.
- 한 row는 최대 3개의 request를 만들고, 여러 row의 request를 합쳐 배치 추론한다.

### 4.2 컨텍스트 길이 제한
- TranslateGemma의 2k 입력 컨텍스트를 넘지 않도록 설계한다.
- 각 field request는 `apply_chat_template()`가 만든 최종 입력 길이 기준으로 상한을 검사한다.
- `problem`과 `solution`은 가능하면 1회 호출로 처리한다.
- `thinking`이 길어서 상한을 넘으면 문단 우선 청크 분할 후 여러 request로 나눠 순차 복원한다.

### 4.3 값 보존 규칙
- 빈 문자열은 번역하지 않고 그대로 유지한다.
- `null`은 `null`로 유지한다.
- 대상 필드가 누락된 row는 해당 필드만 스킵하고 나머지 처리를 계속한다.
- 번역 전 입력 텍스트는 정규화하지 않고 원문 그대로 사용한다.
- 번역 결과 검증은 문자열 여부와 `strip()` 후 비어 있지 않은지만 확인한다.

### 4.4 row 복원 규칙
- batch generation 결과는 `(row_index, field_name, chunk_index)` 기준으로 다시 모은다.
- 청크 분할된 `thinking`은 `chunk_index` 순서대로 이어 붙인다.
- 복원된 각 필드는 원래 row의 동일 필드 위치에 다시 기록한다.

---

## 5. KV cache 및 상태 격리 정책

### 5.1 행 간 독립성
- 이전 row의 KV cache가 다음 row 번역에 영향을 주면 안 된다.
- 각 batch generation 호출은 완전히 독립된 상태에서 시작한다.

### 5.2 필드 간 독립성
- `problem`, `thinking`, `solution`은 별도 request로 처리한다.
- 같은 row 안에서도 이전 필드의 `past_key_values`를 다음 필드 request에 전달하지 않는다.

### 5.3 청크 간 독립성
- 긴 `thinking`을 여러 청크로 나누더라도 청크 간 `past_key_values`를 재사용하지 않는다.
- 모든 generation 호출은 새 cache로 시작한다.

---

## 6. 실행 파이프라인

1. Hugging Face 토큰과 gated repo 접근 가능 여부를 사전 확인한다.
2. 입력 JSONL을 줄 단위로 연다.
3. 출력 디렉터리가 없으면 생성한다.
4. 재시작 가능한 경우 기존 출력 줄 수만큼 입력을 건너뛴다.
5. 입력 row를 JSON으로 파싱한다.
6. 각 row의 `problem`, `thinking`, `solution`을 확인해 field request를 만든다.
7. 여러 row에서 모인 request를 `ROW_BATCH_SIZE` 기준으로 배치화해 공식 chat template 입력을 만든 뒤 generation한다.
8. 결과를 `(row_index, field_name, chunk_index)` 기준으로 다시 조립한다.
9. 비대상 필드는 원본 그대로 유지한 새 row를 만든다.
10. 완성된 row를 출력 JSONL에 즉시 append한다.
11. 처리 완료 row 수와 메타 정보를 즉시 갱신한다.
12. 끝까지 성공하면 job 요약 리포트를 남긴다.

실패 정책:
- Hugging Face 토큰 누락 또는 gated access 실패는 즉시 중단한다.
- JSON 파싱 실패는 즉시 중단한다.
- 출력 직렬화 실패는 즉시 중단한다.
- 모델 추론 실패는 고정 횟수 재시도 후 여전히 실패하면 job 전체를 중단한다.
- 출력 줄 수와 메타의 완료 카운트가 불일치하면 자동 복구하지 않고 즉시 중단한다.

---

## 7. 이어쓰기/스킵 정책

### 7.1 메타 파일
- 출력 JSONL 옆에 진행 메타 파일을 함께 둔다.
- 메타 파일에는 최소 아래를 기록한다.
  - `job_name`
  - `model_id`
  - `auth_mode`
  - `attention_backend`
  - `flash_attention_available`
  - `flash_attention_requested`
  - `flash_attention_enabled_for_runtime`
  - `fallback_attention_backend`
  - `input_path`
  - `output_path`
  - `target_fields`
  - `source_lang_code`
  - `target_lang_code`
  - `completed_rows`
  - `skipped_rows`
  - `retry_count`
  - `started_at`
  - `updated_at`

### 7.2 재실행 규칙
- 재실행 시 출력 파일 줄 수와 메타의 `completed_rows`를 대조한다.
- 둘이 일치하면 입력 앞부분을 건너뛰고 남은 row부터 이어서 처리한다.
- 둘이 불일치하면 자동 복구하지 않고 중단한다.
- 이미 번역된 row는 다시 덮어쓰지 않는다.

---

## 8. 로그와 산출물

### 8.1 로그
- 진행률은 `tqdm(..., file=sys.stdout)`으로 출력한다.
- 디버깅 로그에는 최소 아래를 남긴다.
  - 현재 job
  - 처리 row 수
  - 재시도 수
  - 스킵 수
  - 언어 코드
  - auth mode
  - attention backend
  - flash requested 여부
  - flash availability
  - runtime flash on/off
  - fallback backend

### 8.2 산출물
필수 산출물:
1. 번역 결과 JSONL
2. 진행 메타 파일
3. 완료 요약 리포트

요약 리포트 필수 항목:
- 총 입력 row 수
- 총 출력 row 수
- `model_id`
- `auth_mode`
- `attention_backend`
- `flash_attention_available`
- `flash_attention_requested`
- `flash_attention_enabled_for_runtime`
- `fallback_attention_backend`
- `source_lang_code`
- `target_lang_code`
- `translated_field_counts`
- `chunked_field_counts`
- `skipped_null_or_empty_counts`
- `missing_field_counts`
- `retry_count`
- `status`

---

## 9. 테스트 계획

1. 3~5줄 fixture JSONL로 입력 줄 수와 출력 줄 수가 같은지 확인한다.
2. 출력 row 순서가 입력과 완전히 같은지 확인한다.
3. `problem`, `thinking`, `solution`만 변경되고 나머지 필드는 그대로 유지되는지 확인한다.
4. 한 row에서 세 필드가 raw text 기반의 독립 request로 생성되고, special token은 chat template 렌더링에서만 추가되는지 확인한다.
5. `ROW_BATCH_SIZE=2`에서 여러 row의 request가 함께 batched generation 되는지 확인한다.
6. 긴 `thinking`이 문단 우선 청크 분할 후 다시 원래 순서대로 합쳐지는지 확인한다.
7. 빈 문자열, `null`, 누락 필드가 있는 row에서도 출력 스키마가 유지되는지 확인한다.
8. 부분 출력이 있는 상태에서 재실행 시 이미 끝난 row를 건너뛰고 이어서 처리하는지 확인한다.
9. 출력 줄 수와 메타의 `completed_rows`가 다르면 즉시 중단하는지 확인한다.
10. 실제 flash가 불가능한 입력에서도 일반 SDPA 경로로 폴백해 job이 계속되는지 확인한다.
