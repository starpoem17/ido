data/korean_processed/_staging 안의 parquet 파일들을 읽고 토크나이저를 생성한다.
data/tokenizers/korean_bbpe_v1 의 경로 형식으로 생성한 토크나이저 파일을 저장한다.
통일된 parquet 구조에서 content 필드의 String만 추출하여 토크나이저를 생성한다.
토크나이징 방식은 BBPE를 사용한다.
각 데이터 source 별로 N(아직 미정. parquet 생성 및 중복 제거 이후 결정할 예정.)mb를 샘플링하여 전체 데이터에서 5~10% 정도를 균일 분포로 뽑은 뒤 이를 바탕으로 토크나이저를 빌드한다.
tokenizers 라이브러리의 ByteLevelBPETokenizer를 사용할 때, add_prefix_space=True 옵션과 trim_offsets=True 설정을 적용한다.
