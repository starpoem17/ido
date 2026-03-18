data/korean_processed/near_dedup 안의 parquet 파일들을 읽고 토크나이저를 생성한다.
data/tokenizers/korean_bbpe_v1 의 경로 형식으로 생성한 토크나이저 파일을 저장한다.
통일된 parquet 구조에서 content 필드의 String만 추출하여 토크나이저를 생성한다.
토크나이징 방식은 BBPE를 사용한다.
전체 데이터를 사용하여 토크나이저를 빌드한다.
tokenizers 라이브러리의 ByteLevelBPETokenizer를 사용할 때, add_prefix_space=True 옵션과 trim_offsets=True 설정을 적용한다.