# 전처리 공통 의사코드

## 목적
이 문서는 `docs/plans/preprocess.md`를 실제 코드로 옮길 때 공통으로 따라야 하는 절차를 자연어 형태로 설명한다.
데이터셋별 차이는 각 개별 pseudo 문서에서 정의하고, 여기서는 공통 스키마, 직렬화, token_count, Lance append 절차만 고정한다.

## 공통 스키마
최종 row는 항상 아래 6개 컬럼만 가진다.

1. `data_type`
2. `source`
3. `split`
4. `content`
5. `messages`
6. `token_count`

구현할 때는 다음을 강제한다.

1. `PT` row라면 `content`는 문자열이고 `messages`는 `null`이어야 한다.
2. `FT` row라면 `content`는 `null`이고 `messages`는 role-content 구조의 배열이어야 한다.
3. `split`은 `train` 또는 `val`만 허용한다.
4. 모든 row는 같은 PyArrow schema로 캐스팅 가능해야 한다.

## 공통 직렬화 규칙
`token_count`는 저장된 row를 실제 학습 입력 문자열로 바꾼 뒤 계산한다.

1. `PT` row는 `content` 문자열 자체를 학습 입력으로 본다.
2. `FT` row는 `messages`를 personal 예시와 같은 순서로 직렬화한다.
3. `FT` 직렬화에서는 각 role 앞에 `<|system|>`, `<|user|>`, `<|assistant|>`를 붙인다.
4. 현재 personal 예시에는 `<|bos|>`, `<|eos|>`, `<|eot_id|>`가 등장하지 않으므로 전처리 단계 기본 직렬화에는 넣지 않는다.

## token_count 계산 절차
구현할 때는 아래 순서로 처리한다.

1. 실행 시작 시 `data/tokenizers/korean_bbpe_v1/tokenizer.json`을 한 번만 로드한다.
2. row를 `PT` 또는 `FT` 규칙에 따라 문자열로 직렬화한다.
3. 직렬화된 문자열을 토크나이저로 인코딩한다.
4. 인코딩 결과 토큰 수를 `token_count`에 넣는다.
5. 저장 직전에 한 번 더 같은 규칙으로 재계산해 `token_count`와 일치하는지 확인한다.

## adapter 공통 책임
모든 dataset adapter는 아래 순서를 따라 동작하게 구현한다.

1. 입력 파일 목록을 찾는다.
2. 파일을 안정적인 순서로 정렬한다.
3. 파일 하나를 읽고 원본 구조를 해석한다.
4. 각 원본 레코드를 personal 문서 예시 형태의 row로 바꾼다.
5. row를 만들 수 없으면 품질 이벤트를 남기고 건너뛴다.
6. row를 만들 수 있으면 공통 스키마 검증을 수행한다.
7. 직렬화 기준으로 `token_count`를 계산한다.
8. 검증된 row를 통합 Lance writer로 넘긴다.

## 품질 이벤트 기록 방식
품질 이벤트는 최소한 아래 정보를 남기도록 구현한다.

1. 어떤 데이터셋에서 발생했는지
2. 어느 split인지
3. 어떤 파일에서 발생했는지
4. 가능하면 어떤 세션 또는 레코드에서 발생했는지
5. 사유 코드
6. 사유 설명
7. 샘플 텍스트 일부
8. `skip`, `fixup`, `error` 중 어떤 성격인지

집계할 때는 사유별 전체 건수는 모두 세고, 샘플은 사유별 최대 20건까지만 남긴다.

## 통합 Lance append 절차
통합 writer는 아래 순서로 동작하도록 구현한다.

1. 출력 루트는 `data/korean_processed/`로 고정한다.
2. 실제 chunk는 `data/korean_processed/chunks/` 아래에 저장한다.
3. 첫 실행이면 저장소를 생성한다.
4. 이후 실행이면 기존 schema와 호환되는 경우에만 append한다.
5. row를 순서대로 buffer에 쌓는다.
6. 현재 chunk 바이트가 1GiB를 넘는 시점에, 방금 추가한 row까지 포함해 chunk를 닫는다.
7. 다음 row부터는 새 chunk에 쓴다.
8. 데이터셋 경계만을 이유로 chunk를 미리 닫지 않는다.

## manifest와 quality report 작성 절차
모든 adapter 처리가 끝나면 아래 순서로 후처리한다.

1. chunk별 경로, row 수, 바이트 수를 집계한다.
2. source별 row 수를 집계한다.
3. split별 row 수를 집계한다.
4. `PT` row 수와 `FT` row 수를 집계한다.
5. `schema_version`과 `token_count_version`을 manifest에 기록한다.
6. 품질 이벤트를 사유별로 집계한다.
7. 품질 이벤트 샘플을 사유별 최대 20건 저장한다.
8. manifest와 quality report를 `_manifests/` 아래에 저장한다.
9. 모든 데이터셋 append가 끝난 뒤 `_indices`를 한 번만 생성한다.

## 최종 검증
구현이 끝난 뒤에는 아래를 확인한다.

1. 모든 row가 6개 컬럼만 사용하는가
2. `PT` row와 `FT` row의 null 규칙이 지켜지는가
3. `token_count`가 직렬화 재계산 결과와 일치하는가
4. chunk별 row 수 합계와 전체 row 수가 일치하는가
5. manifest 집계와 quality report 집계가 서로 모순되지 않는가
