# parquet 생성 단계 스도코드

## 목적
이 문서는 raw 데이터를 dataset별 normalized parquet로 만드는 실행 코드의 자연어 의사코드다.
출력 row는 canonical schema 6개 feature를 사용하고, 아직 `token_count`는 비워 둘 수 있다.

## 사용자 설정값
1. `RAW_ROOT = "data/korean_raw"`
2. `OUT_ROOT = "data/korean_processed/_staging/parquet"`
3. `QUALITY_LOG_ROOT = "data/korean_processed/_staging/logs/parquet_generation"`
4. `NUM_WORKERS = 7`
5. `RANDOM_SEED = 사용자 설정값`

## 전체 실행 절차
1. dataset 문서 목록을 순서대로 읽는다.
2. 각 dataset adapter를 실행해 raw row를 canonical row 후보로 바꾼다.
3. dataset별 `split_key`를 기준으로 결정적 99:1 split을 부여한다.
4. row 스키마를 검증한다.
5. `token_count`는 null로 두고 dataset별 parquet를 저장한다.
6. 품질 이벤트는 dataset별 JSONL 로그와 summary JSON으로 저장한다.

## split 부여 helper
1. 각 row에서 dataset 문서가 정의한 `split_key`를 만든다.
2. `hash(source + "\t" + split_key)` 기반 안정 순서를 만든다.
3. row 수의 99%를 `train`, 1%를 `val`로 배정한다.
4. 원천에 train/val 디렉터리가 이미 있어도 최종 split은 이 helper 결과를 사용한다.

## row 검증 절차
1. `source`, `data_usage`, `split`, `content`, `messages`, `token_count` 컬럼이 모두 존재하는지 확인한다.
2. `content`와 `messages` 중 최소 하나가 채워졌는지 확인한다.
3. `data_usage`가 허용값인지 확인한다.
4. `split`이 `train` 또는 `val`인지 확인한다.
5. 실패한 row는 저장하지 않고 품질 로그에 남긴다.

## 산출물
1. dataset별 parquet 파일
2. dataset별 품질 이벤트 JSONL
3. dataset별 품질 이벤트 summary JSON
4. 실행 메타 JSON
