# 번역 파이프라인 공통 의사코드

## 목적
이 문서는 `docs/plans/translate.md`를 실제 코드로 옮길 때 전체 구조와 모듈 경계를 고정한다.  
번역 대상은 영어 JSONL을 한국어 JSONL로 바꾸는 공통 파이프라인이며, 현재는 1개 job만 사용하지만 이후 dataset이 추가되어도 공통 엔진은 그대로 유지한다.

구현 파일은 아래 3개로 분리한다.
1. `src/translate/jobs.py`
2. `src/translate/engine.py`
3. `src/translate/runner.py`

각 파일의 상세 의사코드는 같은 폴더의 개별 문서에 작성한다.

---

## 전체 원칙

1. `docs/personal/translate.md`의 지시를 최우선으로 따른다.
2. 모델은 `google/translategemma-4b-it`를 단일 GPU에 `bf16`으로 1개만 로드한다.
3. 추론 시 멀티프로세싱은 사용하지 않는다.
4. TranslateGemma 공식 사용법에 맞춰 `AutoProcessor` + `AutoModelForImageTextToText` + `apply_chat_template()`를 사용한다.
5. attention backend는 외부 `flash_attn` 패키지가 아니라 PyTorch 내장 SDPA flash backend를 우선 사용한다.
6. 각 row에서 `problem`, `thinking`, `solution`을 각각 raw text 독립 입력으로 만들고, special token은 `apply_chat_template()`에 맡긴 채 여러 row와 함께 배치 추론한다.
7. `thinking`이 길면 field 단위로 청크 분할하고, `problem`과 `solution`은 기본적으로 단일 호출을 유지한다.
8. 행 간, 필드 간, 청크 간 KV cache를 절대 재사용하지 않는다.
9. 입력/출력은 JSONL 스트리밍으로 처리한다.
10. 기본 재시작 정책은 `이어쓰기/스킵`이다.
11. 실행 제어는 CLI가 아니라 Python 파일 상단 설정 블록에서 한다.
12. gated repo 접근은 실행 시작 전에 인증 상태를 먼저 확인한다.
13. PyTorch built-in flash attention이 불가능하면 모델 로드시 일반 SDPA 경로로 전환하고 job 전체 동안 그 정책을 유지한다.
14. 번역 추론에서는 `torch.compile`을 사용하지 않고, `torch.cuda.empty_cache()`도 반복 호출하지 않는다.
15. 번역 전 입력 텍스트는 정규화하지 않고 원문 그대로 처리한다.
16. 번역 결과 검증은 결과가 문자열이고 `strip()` 후 비어 있지 않은지만 확인한다.

---

## 모듈 책임 분리

### 1. `jobs.py`
- 사용자 설정 상수 정의
- `TranslationJob` 데이터 구조 정의
- 활성화할 job 목록 반환
- 입력/출력 경로, 대상 필드, 언어 코드 계약 고정

### 2. `engine.py`
- 모델/processor 로드
- Hugging Face 인증 preflight
- 공식 chat template 메시지 생성
- PyTorch flash-only SDPA 컨텍스트 제어
- 입력 토큰 길이 점검
- field request 생성
- 긴 `thinking` 청크 분할
- KV cache 미재사용 보장
- 번역 결과 검증

### 3. `runner.py`
- job 순회
- JSONL 입력 읽기
- 이어쓰기/스킵 복구
- row batch 단위 오케스트레이션
- 메타 파일/요약 리포트 기록
- tqdm 및 디버그 로그 출력

---

## 실행 순서

`main()`은 아래 순서로 동작한다.

1. `jobs.py`에서 사용자 설정과 활성 job 목록을 읽는다.
2. enabled job이 없으면 로그를 남기고 종료한다.
3. `engine.py`를 사용해 Hugging Face 인증 확인, processor 로드, 모델 로드, PyTorch flash attention runtime 사용 여부 결정을 한 번만 수행한다.
4. enabled job을 순서대로 처리한다.
5. 각 job에서 `runner.py`가 입력 JSONL을 한 줄씩 읽는다.
6. 각 row에서 `problem`, `thinking`, `solution`을 꺼내 field별 request를 만든다.
7. 여러 row에서 모인 field request를 공식 chat template 메시지로 감싸 배치 generation한다.
8. 비대상 필드는 원본 그대로 유지한 새 row를 만든다.
9. 완성된 row를 출력 JSONL에 즉시 append한다.
10. row 하나가 끝날 때마다 메타 파일의 `completed_rows`를 갱신한다.
11. job 종료 후 요약 리포트를 저장한다.
12. 모든 job이 끝나면 최종 요약 로그를 남기고 종료한다.

---

## row 처리 공통 절차

`row_번역_처리(row, job, engine)` 절차는 아래와 같다.

1. 원본 row를 파싱한다.
2. 원본 row를 얕은 복사하지 말고 명시적으로 새 dict를 만든다.
3. row의 모든 key를 순회하면서 기본값은 원본 그대로 복사한다.
4. `job.target_fields`에 포함된 key만 번역 대상 후보로 표시한다.
5. 각 대상 필드에 대해 아래를 수행한다.
   - key가 없으면 스킵 카운트 증가
   - 값이 `null`이면 그대로 유지
   - 값이 빈 문자열이면 그대로 유지
   - 값이 문자열이 아니면 실패 처리
   - 문자열이면 field 단위 request 계획을 만든다
6. row batch 단위로 request를 모아 batched generation 한다.
7. `(row_index, field_name, chunk_index)` 기준으로 결과를 복원한다.
8. 세 필드 처리가 모두 끝나면 새 row를 JSON 직렬화한다.
9. 직렬화에 성공하면 출력 파일에 한 줄 append한다.
10. append 후 메타 파일을 갱신한다.

강제 규칙:
- 하나의 row는 최대 3개 field request를 만들 수 있다.
- 긴 `thinking`은 여러 request로 분할될 수 있다.
- row 순서 보존을 위해 출력 순서를 바꾸지 않는다.

---

## 재시작 공통 절차

`이어쓰기_복구(job)` 절차는 아래와 같다.

1. 출력 JSONL 존재 여부 확인
2. 메타 JSON 존재 여부 확인
3. 둘 다 없으면 신규 실행으로 판단하고 `completed_rows = 0`
4. 둘 다 있으면 출력 줄 수를 센다.
5. 메타의 `completed_rows`와 출력 줄 수를 비교한다.
6. 둘이 같으면 그 수만큼 입력 줄을 건너뛴다.
7. 둘이 다르면 자동 복구하지 말고 즉시 예외를 발생시킨다.

---

## 품질 검증 공통 절차

각 field 번역 후에는 아래를 검사한다.

1. 결과가 문자열인지 확인
2. 결과 문자열이 `strip()` 후 빈값이 아닌지 확인

검증 실패 시:
1. 재시도 가능 횟수 안이면 같은 입력으로 새 generation 호출
2. 재시도 후에도 실패하면 job 전체 중단

---

## 메타와 리포트 공통 계약

메타 JSON에는 최소 아래를 기록한다.
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

요약 리포트 JSON에는 최소 아래를 기록한다.
- `job_name`
- `model_id`
- `auth_mode`
- `attention_backend`
- `flash_attention_available`
- `flash_attention_requested`
- `flash_attention_enabled_for_runtime`
- `fallback_attention_backend`
- `total_input_rows`
- `total_output_rows`
- `source_lang_code`
- `target_lang_code`
- `translated_field_counts`
- `chunked_field_counts`
- `skipped_null_or_empty_counts`
- `retry_count`
- `status`

---

## 구현 중 금지 사항

1. `problem + thinking + solution`을 하나의 row source text로 concat하지 않는다.
2. 수동 시스템 프롬프트 문자열을 만들어 TranslateGemma를 일반 Causal LM처럼 사용하지 않는다.
3. 외부 `flash_attn` 패키지를 설치하거나 요구하는 경로를 사용하지 않는다.
4. 이전 row의 `past_key_values`를 다음 row에 전달하지 않는다.
5. 같은 row 내 이전 필드의 `past_key_values`를 다음 필드에 전달하지 않는다.
6. row batch 사이에서 cache를 이어붙이지 않는다.
7. 입력 전체를 메모리에 적재하지 않는다.
8. 사용자 설정값을 CLI 인자로 빼지 않는다.
