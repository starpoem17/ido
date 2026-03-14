# 번역 실행기 의사코드

## 대상 파일
이 문서는 `src/translate/runner.py`를 작성하기 위한 의사코드다.

이 파일의 책임:
1. enabled job 순회
2. JSONL 입력/출력 스트리밍
3. 이어쓰기/스킵 복구
4. row batch 단위 field 번역 오케스트레이션
5. 메타 파일과 요약 리포트 기록

---

## 실행 시작

`main()` 절차:

1. 표준 출력으로 시작 로그를 남긴다.
2. `get_enabled_jobs()` 호출
3. enabled job이 0개면 종료 로그를 남기고 return
4. `load_translation_engine()`를 1회 호출
5. job을 순서대로 순회하며 `run_job(job, engine)` 호출
6. 모든 job 성공 시 최종 성공 로그 출력

---

## job 실행

`run_job(job, engine)` 절차:

1. `job.validate()` 호출
2. 입력 파일 존재 여부 확인
3. 출력 디렉터리 생성
4. `prepare_resume_state(job)` 호출
5. 입력 총 줄 수 계산
6. 메타 파일 초기화 또는 기존 메타 로드
7. 입력 파일을 텍스트 모드로 연다.
8. 출력 파일을 append 모드로 연다.
9. `tqdm(total=총 줄 수, file=sys.stdout)` 생성
10. 이미 완료된 row 수만큼 progress를 먼저 갱신
11. 입력 줄을 순회하며 완료된 row 수 이전 구간은 건너뛴다.
12. 남은 줄을 `ROW_BATCH_SIZE`만큼 모은다.
13. 묶인 줄을 `process_row_batch(...)`로 넘긴다.
14. batch 결과 row들을 순서대로 output에 write 후 flush
15. row별로 메타의 `completed_rows` 갱신
16. progress update
17. 주기적으로 디버그 로그 출력
18. 종료 후 report 저장

시작 로그에는 아래를 포함한다.
- job 이름
- 입력 경로
- 출력 경로
- resume 상태
- `source_lang_code -> target_lang_code`
- `mode=official-chat-template`
- `auth_mode`
- `attention_backend`
- `flash_attention_available`
- `flash_attention_requested`
- `flash_attention_enabled_for_runtime`
- `fallback_attention_backend`
- `row_batch_size`

---

## resume state 준비

`prepare_resume_state(job)` 절차:

1. 출력 파일 존재 여부 확인
2. 메타 파일 존재 여부 확인
3. 둘 다 없으면 아래 반환
   - `completed_rows = 0`
   - `retry_count = 0`
   - 새 메타 초안
4. 둘 다 있으면 출력 줄 수를 센다.
5. 메타 JSON을 읽는다.
6. 메타의 `completed_rows`와 출력 줄 수를 비교한다.
7. 다르면 예외 발생
8. 같으면 기존 상태를 이어받아 반환

---

## row batch 처리

`process_row_batch(raw_lines, batch_start_index, job, engine, stats)` 절차:

1. raw line batch를 순회하며 각 row를 `json.loads` 한다.
2. dict가 아니면 예외 발생
3. 각 row에 대해 새 `translated_row = {}`를 만들고 원본 key/value를 복사한다.
4. 각 row마다 `field_plans` dict를 만든다.
5. `job.target_fields` 순서대로 아래를 수행한다.
   - field가 없으면 `stats.missing_fields[field] += 1`
   - 값이 `None`이면 `stats.null_fields[field] += 1`
   - 값이 문자열이 아니면 예외 발생
   - 빈 문자열이면 `stats.empty_fields[field] += 1`
   - 그 외에는 `engine.plan_field_requests(...)` 호출
   - 반환된 request들을 공통 request list에 추가
   - `FieldTranslationPlan`을 row별 계획 dict에 저장
6. 모든 row에서 모인 request들을 `engine.translate_requests_batch(...)`로 처리한다.
7. batch 결과를 `(row_index, field_name)` 기준으로 다시 모은다.
8. 각 row의 field plan을 순회하면서 `chunk_index` 순서로 번역 결과를 이어 붙인다.
9. 복원된 필드를 `translated_row[field_name]`에 기록한다.
10. `retry_count`와 `translated_fields`, `chunked_fields` 통계를 누적한다.
11. 최종 `translated_row`들을 입력 순서대로 JSONL 문자열로 반환한다.

마무리:
1. `translated_row`를 `json.dumps(..., ensure_ascii=False)` 한다.
2. 문자열 끝에 `\n` 추가
3. 직렬화 문자열 반환

---

## 메타 파일 갱신

`write_meta(job, engine, state, stats)` 절차:

1. 메타 dict 생성
2. 아래 필드 채움
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
3. 임시 파일에 JSON 저장
4. 원자적 rename으로 실제 메타 파일 교체

강제 규칙:
- row 하나가 파일에 성공적으로 append된 뒤에만 `completed_rows` 증가
- 메타 파일은 매 row마다 갱신한다

---

## 요약 리포트 작성

`write_report(job, engine, state, stats, total_input_rows, status)` 절차:

1. report dict 생성
2. 아래 필드를 채운다.
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
   - `missing_field_counts`
   - `retry_count`
   - `status`
3. JSON으로 저장

---

## 통계 구조

`JobStats` 구조를 둔다.

필드:
1. `translated_fields: dict[str, int]`
2. `chunked_fields: dict[str, int]`
3. `null_fields: dict[str, int]`
4. `empty_fields: dict[str, int]`
5. `missing_fields: dict[str, int]`
6. `retry_count: int`

초기화 규칙:
- `problem`, `thinking`, `solution`에 대해 모두 0으로 시작

---

## 예외 처리 규칙

1. Hugging Face 인증 preflight 실패는 engine load 단계에서 즉시 job 중단
2. JSON 파싱 실패는 즉시 job 중단
3. 대상 필드 값이 문자열/None이 아니면 즉시 job 중단
4. 모델 추론 예외는 engine 내부 retry 종료 후 바깥으로 전파
5. 메타 불일치 예외는 즉시 job 중단
6. flash probe 실패는 즉시 job 중단 사유가 아니라 일반 SDPA 경로 전환 사유로 기록한다.
7. job 실패 시 report의 `status = "failed"`로 저장
8. job 성공 시 report의 `status = "completed"`로 저장

---

## 로그 규칙

1. 시작 시 job 이름, 입력 경로, 출력 경로, 언어 코드, auth mode, attention backend, runtime flash 여부 출력
2. 주기 로그에는 처리 row 수와 누적 retry 수 출력
3. field 번역 실패 재시도 시 row index와 field 이름 출력
4. 완료 시 총 row 수와 field별 통계 출력

---

## 검증 포인트

구현 후 아래가 보장되어야 한다.

1. 입력 줄 수와 출력 줄 수가 같아야 한다.
2. 출력 row 순서는 입력과 같아야 한다.
3. 비대상 필드는 원본과 동일해야 한다.
4. 부분 실행 후 재실행 시 완료된 줄은 다시 번역하지 않아야 한다.
5. 메타 줄 수 불일치 시 즉시 중단해야 한다.
6. 메타와 리포트에 `auth_mode`, 언어 코드, attention backend, runtime flash 정책 정보가 남아야 한다.
