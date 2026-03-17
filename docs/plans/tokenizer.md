# Korean BBPE 토크나이저 생성 계획

## 0. 목적
이 문서는 [docs/personal/tokenizer.md](/home/hwajoong/projects/ido/docs/personal/tokenizer.md)와
[docs/personal/preprocess/preprocess.md](/home/hwajoong/projects/ido/docs/personal/preprocess/preprocess.md)를 기준으로
한국어 BBPE 토크나이저 생성 계획을 고정한다.

핵심 원칙:
1. 입력은 `data/korean_processed/_staging` 안의 dedup 완료 parquet다.
2. parquet의 canonical row에서 `content` 필드의 문자열만 추출해 토크나이저를 학습한다.
3. 토크나이징 방식은 BBPE를 사용한다.
4. source별로 `N MB`씩 균일 샘플링하고, 전체 데이터의 5~10% 수준을 사용한다.
5. `N` 값은 아직 미정이며 코드 맨 앞 사용자 설정 상수로 둔다.

---

## 1. 입력과 출력

### 1.1 입력
- 입력 루트는 `data/korean_processed/_staging`이다.
- 이 경로의 parquet는 exact dedup과 minhash + LSH dedup까지 완료된 결과로 본다.
- 토크나이저 학습에는 canonical schema의 `content` 문자열만 사용한다.
- `messages`, `token_count`, 기타 메타데이터는 토크나이저 학습 텍스트로 사용하지 않는다.

### 1.2 출력
- 출력 루트는 `data/tokenizers/korean_bbpe_v1/`로 둔다.
- 최소 산출물:
  1. `tokenizer.json`
  2. `vocab.json`
  3. `merges.txt`
  4. `meta.json`

`meta.json`에는 최소한 아래를 남긴다.
- 실행 시각
- 입력 parquet 경로 범위
- source별 샘플링 바이트 수
- source별 샘플링 row 수
- 전체 샘플링 비율
- 사용한 BBPE 설정
- 사용한 special token 목록

---

## 2. 토크나이저 규격

### 2.1 알고리즘
- 라이브러리는 `tokenizers`를 사용한다.
- 알고리즘은 `ByteLevelBPETokenizer`를 사용한다.
- 옵션은 아래를 고정한다.
  - `add_prefix_space=True`
  - `trim_offsets=True`

### 2.2 학습 대상 텍스트
- parquet row에서 `content`가 문자열이고 비어 있지 않은 경우만 사용한다.
- source별로 동일한 샘플링 규칙을 적용해 편향을 줄인다.
- 한 source에서 지나치게 많은 텍스트가 전체 샘플을 지배하지 않도록 source별 상한 `N MB`를 둔다.
- 전체 샘플량은 전체 dedup 완료 데이터의 5~10% 정도가 되도록 맞춘다.

### 2.3 special token
- preprocess와 학습 계획에서 사용하는 아래 토큰들을 함께 관리한다.
  - `<|bos|>`
  - `<|eos|>`
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
  - `<|eot_id|>`
  - `<|pad|>`

---

## 3. 실행 계획

### 3.1 전체 흐름
1. `data/korean_processed/_staging`의 dedup 완료 parquet 파일 목록을 수집한다.
2. parquet에서 `source`, `content`만 읽는다.
3. `content`가 비어 있거나 null인 row는 제외한다.
4. source별로 `N MB`까지 균일 샘플링한다.
5. 전체 샘플량이 전체 데이터의 5~10% 범위에 들어오는지 확인한다.
6. 샘플링된 `content` 문자열을 iterator로 공급해 `ByteLevelBPETokenizer`를 학습한다.
7. 생성된 토크나이저 파일을 `data/tokenizers/korean_bbpe_v1/`에 저장한다.
8. 실행 메타데이터와 샘플링 통계를 `meta.json`으로 저장한다.

### 3.2 샘플링 규칙
- 샘플링은 source별로 독립 수행한다.
- 각 source에서 최대 `N MB`까지만 선택한다.
- `N` 값은 parquet 생성 및 dedup 결과를 확인한 뒤 코드 상단 사용자 설정값으로 결정한다.
- 샘플링 구현은 재현 가능해야 하며, 선택 규칙과 seed는 `meta.json`에 남긴다.

### 3.3 실행 형식
- 실행 코드는 `uv run python -m ...` 형태를 따른다.
- 자주 바뀌는 값은 터미널 인자 대신 코드 상단 사용자 설정 블록에 둔다.
- 진행률은 `tqdm(..., file=sys.stdout)`로 출력한다.
- CPU 기반 처리 단계는 기본 7 프로세스 멀티프로세싱 기준으로 설계한다.

---

## 4. 검증 기준

1. 입력 parquet가 dedup 완료 staging 산출물과 일치해야 한다.
2. 학습 텍스트가 `content` 필드만으로 구성되어야 한다.
3. `ByteLevelBPETokenizer`와 `add_prefix_space=True`, `trim_offsets=True`가 적용되어야 한다.
4. source별 샘플링 바이트 수와 전체 샘플 비율이 메타에 기록되어야 한다.
5. `tokenizer.json`, `vocab.json`, `merges.txt`, `meta.json`이 생성되어야 한다.

---

## 5. 연결 문서
- 전처리 파이프라인 내 위치는 [docs/plans/preprocess.md](/home/hwajoong/projects/ido/docs/plans/preprocess.md)를 따른다.
- 실제 한국어 자연어 의사코드는 [docs/pseudo/tokenizer.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer.md)에서 상세화한다.
