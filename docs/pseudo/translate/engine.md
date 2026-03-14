# 번역 엔진 의사코드

## 대상 파일
이 문서는 `src/translate/engine.py`를 작성하기 위한 의사코드다.

이 파일의 책임:
1. 모델/processor 로드
2. Hugging Face 인증 preflight
3. PyTorch built-in flash attention 사용 가능 여부 확인
4. 공식 chat template 메시지 생성
5. field 단위 원문 입력 생성
6. 토큰 길이 계산
7. field request 배치 번역
8. 긴 `thinking` 청크 분할
9. KV cache 미재사용 보장

---

## 주요 타입

### 1. `LoadedTranslationEngine`
필드:
1. `model`
2. `processor`
3. `device`
4. `model_id`
5. `max_input_tokens`
6. `auth_mode`
7. `attention_backend`
8. `flash_attention_available`
9. `flash_attention_requested`
10. `flash_attention_enabled_for_runtime`
11. `fallback_attention_backend`

### 2. `TranslationRequest`
필드:
1. `row_index`
2. `field_name`
3. `chunk_index`
4. `source_text`
5. `source_lang_code`
6. `target_lang_code`
7. `max_output_tokens`

### 3. `FieldTranslationPlan`
필드:
1. `field_name`
2. `requests`
3. `used_chunking`
4. `chunk_count`

### 4. `TranslationBatchResult`
필드:
1. `request`
2. `translated_text`
3. `retry_count`

---

## 인증 및 모델 로드 절차

`load_translation_engine()` 절차:

1. `torch`, `transformers`, `huggingface_hub` 의존성이 있는지 확인한다.
2. CUDA 사용 가능 여부를 확인한다.
3. `torch.backends.cuda.is_flash_attention_available()`로 PyTorch built-in flash attention 가능 여부를 확인한다.
4. PyTorch built-in flash attention이 가능하면 processor와 model을 로드한 뒤 짧은 probe generation으로 실제 runtime 입력에서 flash-only 실행이 가능한지 1회 확인한다.
5. probe에 성공하면 job 전체 동안 flash-only SDPA 경로를 사용한다.
6. probe에 실패하면 job 전체 동안 일반 SDPA 경로로 폴백한다.
7. `MODEL_ID`가 로컬 경로인지 Hugging Face repo id인지 판별한다.
8. repo id이면 Hugging Face token을 조회한다.
9. gated repo 접근이 필수이면 `HfApi().model_info()`로 접근 가능 여부를 사전 확인한다.
10. token이 없거나 접근 권한이 없으면 명확한 에러를 발생시킨다.
11. `AutoProcessor.from_pretrained(...)`로 processor를 로드한다.
12. `AutoModelForImageTextToText.from_pretrained(..., attn_implementation="sdpa")`로 model을 `dtype=torch.bfloat16`으로 로드한다.
13. model을 `cuda`로 이동하고 eval 모드로 전환한다.
14. `torch.compile`은 적용하지 않는다.
15. `LoadedTranslationEngine` 반환

강제 규칙:
- 모델 인스턴스는 프로세스 내 하나만 만든다.
- CPU fallback 추론은 허용하지 않는다.
- 인증 실패는 generation 단계까지 미루지 않고 로드 단계에서 즉시 중단한다.
- 외부 `flash_attn` 패키지는 사용하지 않는다.
- flash 사용 여부는 모델 로드시 한 번만 결정하고 job 전체 동안 유지한다.

---

## 공식 메시지 생성

`build_messages(source_text, source_lang_code, target_lang_code)` 절차:

1. `role = "user"` 메시지 1개를 만든다.
2. `content`에는 아래 dict 1개만 넣는다.
   - `type = "text"`
   - `source_lang_code`
   - `target_lang_code`
   - `text = source_text`
3. 완성된 메시지 list를 반환한다.

강제 규칙:
- 수동 시스템 프롬프트 문자열을 추가하지 않는다.
- text translation v1에서는 image content를 받지 않는다.

---

## field 입력 생성

`build_messages(source_text, source_lang_code, target_lang_code)`에 넘기는 `source_text`는 field 원문 그대로 유지한다.

강제 규칙:
- `problem`, `thinking`, `solution`을 하나로 concat하지 않는다.
- field 이름 자체를 자연어 프롬프트에 추가하지 않는다.
- `<|bos|>`, `<|eos|>` 같은 문자열을 입력 본문에 직접 넣지 않는다.
- special token과 turn 구조는 tokenizer와 `apply_chat_template()`가 구성한다.

---

## 토큰 길이 계산

`build_model_inputs(source_text, source_lang_code, target_lang_code)` 절차:

1. 공식 메시지 list 생성
2. `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")` 호출
3. 반환된 tensor dict를 target device로 옮긴다.
4. tensor dict 반환

`estimate_input_tokens(source_text, source_lang_code, target_lang_code)` 절차:

1. `build_model_inputs(...)` 호출
2. `input_ids.shape[-1]`를 읽는다.
3. 입력 tensor 참조를 정리하고 길이 반환한다.

`can_fit_single_call(source_text, source_lang_code, target_lang_code)` 절차:

1. `estimate_input_tokens(...)` 호출
2. `MAX_INPUT_TOKENS - MIN_RESERVED_OUTPUT_TOKENS`와 비교
3. 들어가면 `True`, 아니면 `False`

---

## 긴 텍스트 청크 분할

`split_text_into_chunks(source_text, source_lang_code, target_lang_code)` 절차:

1. 먼저 문단 기준으로 나눈다.
2. 문단 하나가 너무 길면 문장 경계 우선으로 추가 분할한다.
3. 그래도 길면 문자 길이 기반으로 안전 분할한다.
4. 각 청크는 끝까지 raw text 상태로 유지한다.
5. 모든 청크가 safe budget 안에 들어가는지 다시 확인한다.

`max_output_tokens_for_field(field_name)` 절차:

1. `problem`이면 `MAX_OUTPUT_TOKENS_PROBLEM`
2. `thinking`이면 `MAX_OUTPUT_TOKENS_THINKING`
3. `solution`이면 `MAX_OUTPUT_TOKENS_SOLUTION`

`plan_field_requests(row_index, field_name, source_text, source_lang_code, target_lang_code)` 절차:

1. 입력이 빈 문자열이면 빈 request list를 가진 `FieldTranslationPlan` 반환
2. raw `source_text` 그대로 한 번에 들어가는지 검사한다.
3. 한 번에 들어가면 `chunk_index = 0`인 `TranslationRequest` 1개 생성
4. 한 번에 안 들어가면 `thinking`만 청크 분할을 허용한다.
5. 청크 분할된 각 조각도 raw text 그대로 request list 생성
6. `used_chunking`, `chunk_count`를 채워 `FieldTranslationPlan` 반환

강제 규칙:
- `problem`과 `solution`은 기본적으로 청크 분할하지 않는다.
- 너무 긴 `problem`/`solution`은 즉시 예외 처리한다.

---

## 배치 generation

`build_batch_model_inputs(requests)` 절차:

1. request list를 순서대로 conversation list로 바꾼다.
2. `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)` 호출
3. 배치 텐서를 target device로 이동한다.
4. `attention_mask.sum(-1)`로 sample별 prompt 길이를 계산한다.

`generate_batch_once(requests)` 절차:

1. batch inputs를 만든다.
2. batch 내 `max_output_tokens` 최댓값으로 `max_new_tokens`를 정한다.
3. engine이 `flash_attention_enabled_for_runtime=True`이면 PyTorch SDPA flash-only 컨텍스트에서 `model.generate()` 호출
4. 아니면 일반 SDPA 경로로 `model.generate()` 호출
5. 각 sample에서 prompt 길이 이후 토큰만 잘라 decode한다.
6. 결과 문자열 list를 request 순서대로 반환한다.

핵심 규칙:
- 함수 인자에 `past_key_values`를 받지 않는다.
- 함수 반환값에도 `past_key_values`를 포함하지 않는다.
- 호출 단위 임시 cache는 함수 종료와 함께 버린다.

---

## 번역 결과 검증 및 재시도

`validate_translation_text(translated_text)` 절차:

1. 결과가 문자열인지 확인
2. 결과가 `strip()` 후 비어 있지 않은지 확인

`translate_requests_batch(requests)` 절차:

1. request list를 `MAX_REQUESTS_PER_BATCH` 단위로 나눈다.
2. 각 sub-batch를 한 번의 `model.generate()`로 처리한다.
3. batch generation 실패 시 batch를 더 작은 단위로 나눠 재시도한다.
4. sample별 텍스트 검증 실패는 singleton 재시도로 분리한다.
5. request 순서대로 `TranslationBatchResult`를 반환한다.

재시도 정책:
1. batch generation 예외 또는 sample validation 실패 시 retry count 증가
2. `RETRY_LIMIT` 초과 시 예외 발생

---

## 공개 API

이 파일에서 `runner.py`가 호출할 공개 함수는 아래만 둔다.

1. `load_translation_engine()`
2. `plan_field_requests(row_index, field_name, source_text, source_lang_code, target_lang_code)`
3. `translate_requests_batch(requests)`

`runner.py`는 model internals, processor internals, chat template internals, chunking internals를 직접 다루지 않는다.
