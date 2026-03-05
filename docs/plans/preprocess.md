# Korean Raw 전처리 마스터 플랜 (확장형 문서 구조)

## 문서 목적
이 문서는 `data/korean_raw`의 다양한 데이터 형식을 공통 학습 포맷으로 변환하기 위한 전처리 계획서다.  
문서 구조를 데이터셋별로 분리해, `009`뿐 아니라 `010`, `011`, `019` 및 나머지 데이터셋 계획을 동일한 틀에서 계속 확장할 수 있도록 설계한다.
009(009.전문분야_기술과학_한국어 멀티세션 데이터)를 기본 틀로 잡고 이후 나머지 데이터들은 009와 같은 데이터 형식으로 진행되도록 방향을 잡는다

본 문서는 정책/논리/검증 기준을 정의하며, 코드 구현 내용은 포함하지 않는다.

---

## 1. 전체 데이터 공통 전략

### 1.1 공통 목표 스키마 (Canonical Row Schema v1)
`docs/personal/preprocess.md` 기준으로 최종 저장 스키마를 고정한다.

1. `data_type: String` (`PT` 또는 `FT`)
2. `source: String`
3. `split: String` (`train`, `val`)
4. `content: String | null`
5. `messages: List[Struct{role: String, content: String}] | null`
6. `token_count: int32`

공통 강제 조건:
- 모든 데이터셋은 하나의 통합 Lance 저장소로 병합할 수 있도록 동일한 PyArrow schema를 적용한다.
- `token_count`는 데이터셋별 예외 없이 동일한 단일 로직으로 계산한다.
- 스키마/토큰 카운트 규칙 변경 시 `schema_version`, `token_count_version`을 함께 갱신한다.

규칙:
- `PT` row: `content` 사용, `messages = null`
- `FT` row: `messages` 사용, `content = null`
- 하나의 row는 `PT` 또는 `FT` 중 하나만 충족해야 한다.

### 1.2 공통 파이프라인
모든 데이터셋은 아래 동일한 3단계를 따른다.

1. `DuckDB Normalize`
- 원본 구조를 row 단위로 평탄화
- split/source/data_type 매핑
- 품질 필터 적용

2. `Arrow Export`
- DuckDB 결과를 Arrow Table로 변환
- 스키마 및 null 규칙 검증

3. `Lance Persist`
- Arrow를 통합 Lance 저장소에 append 저장
- chunk(shard) 단위 관리

### 1.3 공통 샤딩 정책
- 샤드 목표 크기: `1,073,741,824 bytes` (1GiB)
- 샤딩은 행 단위로만 수행한다.
- 행 중간 분할은 금지한다.
- 임계값 초과 직전이라도 다음 행 추가 후 초과되면 그 행까지 포함해 해당 샤드를 종료한다.

예시:
- 현재 `999MB`, 다음 row 반영 후 `1005MB`가 되면 `1005MB`로 shard 종료

### 1.4 공통 검증 항목
1. 스키마 검증
- 컬럼 6개 및 타입 일치
- `split` 허용값 검증
- `PT/FT` 상호 배타 규칙 검증

2. 수량 검증
- 입력 row 수 대비 출력 row 수 및 제외 row 수 일치
- split별 집계 일치

3. 샤딩 검증
- shard 합계 row 수 = 전체 row 수
- 행 원자성 위반 없음

### 1.5 공통 산출물
통합 출력 루트: `data/korean_processed/`

권장 구조:
```text
data/
└── korean_processed/
    ├── chunks/
    │   ├── chunk-000001.lance
    │   ├── chunk-000002.lance
    │   └── ...
    ├── _indices/
    └── _manifests/
```

1. `chunks/chunk-xxxxxx.lance`
- 1GiB 내외 행 단위 샤드
- 서로 다른 source 데이터가 동일 저장소에 공존 가능
2. `_manifests/manifest-<version>.json`
- `schema_version`, `token_count_version`, chunk별 bytes/rows, split/source 집계
3. `_manifests/quality_report-<run_id>.json`
- 제외 규칙/제외 건수/샘플 파일 경로
4. `_indices/*`
- 검색/필터용 인덱스 산출물

### 1.6 데이터셋 전략 섹션 템플릿
아래 템플릿으로 각 데이터셋 섹션을 갱신한다.

1. 입력 경로 및 split 매핑
2. row 단위 정의
3. `PT/FT` 매핑 규칙
4. `messages/content` 구성 규칙
5. 이상치 정책
6. token_count 기준
7. 예상 산출량/리스크/확정 여부

---

## 2. 009.전문분야_기술과학_한국어 멀티세션 데이터 전략 (확정)

### 2.1 입력 경로
- `data/korean_raw/009.../3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json`
- `data/korean_raw/009.../3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json`

### 2.2 split 매핑
- `Training -> train`
- `Validation -> val`

### 2.3 row 단위 정의
- `sessionInfo.dialog`를 순차 파싱해 `직전 user + 다음 assistant`를 1 row로 생성
- `messages`는 항상 길이 3 (`system`, `user`, `assistant`)

### 2.4 FT 구성 규칙
- `data_type = FT`
- `source = "009.전문분야_기술과학_한국어 멀티세션 데이터"`
- `content = null`
- `system` 고정 문구:  
`너는 기술과학 전문 비서야. 사용자의 질문에 친절하고 과학적으로 대답해.`

### 2.5 이상치 정책
아래 조건 중 하나라도 해당하면 세션 전체 제외:
1. `assistant -> assistant` 연속 턴
2. 세션 종료 시 unpaired user 존재
3. 빈 발화 포함

### 2.6 token_count 정책
- `system + user + assistant` 자연어 concat 기준

### 2.7 현황 수치 (검증 기준)
- 전체 세션: `135,105`
- 이상 세션: `19` (`~0.014%`)
- 예상 출력 row: `train 957,180`, `val 119,631`, `total 1,076,811`

### 2.8 통합 저장소 적재 규칙
- 009 출력은 `data/korean_processed/chunks/`로 저장한다.
- 데이터셋 구분은 경로 분리 대신 `source` 컬럼으로 식별한다.
- 후속 데이터셋(010, 011, 019 등)도 동일 schema를 유지한 채 같은 저장소에 누적 적재한다.

---

## 3. 010.전문분야_사회과학_한국어 멀티세션 데이터 전략 (초안)

### 3.1 기본 방향
- 009와 동일한 멀티세션 계열로 간주하고, 1차 전략은 009 규칙 재사용
- `messages` 3개 고정 전략(`system/user/assistant`) 우선 적용
- 결과는 통합 저장소(`data/korean_processed/chunks`)에 적재

### 3.2 작성할 항목 (채움 대상)
1. 입력 경로/파일 포맷 확정 (`json`/`txt` 중 단일 소스 선택)
2. split 매핑 확정 (`Training/Validation`)
3. 발화자 키 매핑(`speaker1/2` 여부) 확인
4. 도메인 system 문구 확정(사회과학 전용)
5. 이상치 유형 및 제외 기준 확정
6. 예상 산출량(세션/row/제외 비율) 기록

### 3.3 승인 조건
- 009 대비 다른 스키마 필드가 있으면 Adapter 규칙만 분기하고 Canonical 스키마는 유지

---

## 4. 011.일상대화 한국어 멀티세션 데이터 전략 (초안)

### 4.1 기본 방향
- 멀티세션 대화형 데이터로 분류
- 009/010과 동일한 페어링 엔진을 재사용 가능한 기준으로 설계
- 결과는 통합 저장소(`data/korean_processed/chunks`)에 적재

### 4.2 작성할 항목 (채움 대상)
1. 입력 경로 확정
2. split 매핑 확정
3. speaker role 매핑 확정
4. 일상대화 도메인 system 문구 확정
5. 빈 발화/연속 assistant/unpaired user 정책 확정
6. 출력 row 수 및 제외 비율 산출

### 4.3 승인 조건
- 대화형인 경우 `FT` 기본
- PT 혼합이 확인되면 혼합 전략(PT/FT 동시 적재)으로 문서 갱신

---

## 5. 019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터 전략 (초안)

### 5.1 기본 방향
- 텍스트 분석 계열로 분류
- 대화형이 아닐 가능성을 전제로 `PT` 우선 검토, `FT` 혼합 여부는 구조 확인 후 결정
- 결과는 통합 저장소(`data/korean_processed/chunks`)에 적재

### 5.2 작성할 항목 (채움 대상)
1. 원본 레코드 단위 정의(문서/문단/질의응답)
2. `data_type` 결정 규칙
- 문서형이면 `PT(content)`
- 질의응답형이면 `FT(messages)`
3. split 경로/메타 매핑 확정
4. 긴 문서 분할 전략(문단/문장 단위) 확정
5. 법률 도메인 품질 규칙(빈 문서, 깨진 인코딩, 극단 길이) 확정

### 5.3 승인 조건
- 019는 009형 고정 규칙을 재사용하지 않고, 구조 확인 후 Adapter 분기 정책을 명시한다.

---

## 6. 나머지 데이터 형식 수정 계획 (확장 백로그)

### 6.1 대상 목록
- `020.주제별 텍스트 일상 대화 데이터`
- `021.용도별 목적대화 데이터`
- `023.국회 회의록 기반 지식검색 데이터`
- `030.웹데이터 기반 한국어 말뭉치 데이터`
- `045.지식검색 대화`
- `046.공감형 대화`
- `141.한국어 멀티세션 대화`
- `전문분야 말뭉치`
- `gsm8k`
- `novel24`

### 6.2 백로그 관리 표준
각 데이터셋은 아래 상태로 관리한다.
- `draft`: 경로/포맷만 파악
- `profiling`: 스키마/분포/이상치 측정 중
- `rule-locked`: PT/FT/품질 규칙 확정
- `ready`: 구현 가능 상태

### 6.3 데이터셋별 기록 템플릿
아래 형식으로 동일 문서에 섹션을 추가한다.

```md
## X. <dataset_name> 전략 (<status>)

### X.1 입력 경로 및 split
- ...

### X.2 row 단위 및 data_type
- ...

### X.3 변환 규칙
- ...

### X.4 이상치/품질 정책
- ...

### X.5 token_count 정책
- ...

### X.6 예상 산출량 및 리스크
- ...
```

### 6.4 우선순위 권장
1. 멀티세션 계열: `010`, `011`, `141`
2. 대화 계열: `020`, `021`, `045`, `046`
3. 문서/검색 계열: `019`, `023`, `030`, `전문분야 말뭉치`
4. 외부 코퍼스 계열: `gsm8k`, `novel24`

---

## 7. 문서 운영 규칙
이 문서는 구현 전 의사결정 문서로 유지하며, 데이터셋별 정책 확정 시 아래 순서로만 갱신한다.

1. 공통 전략 변경 필요 여부 검토
2. 해당 데이터셋 섹션에 규칙 확정값 반영
3. 백로그 상태 업데이트 (`draft -> profiling -> rule-locked -> ready`)
4. 산출량/제외 비율/리스크 수치 반영

이 구조를 유지하면 009 중심 문서에서 전체 데이터셋 로드맵 문서로 자연스럽게 확장할 수 있다.
