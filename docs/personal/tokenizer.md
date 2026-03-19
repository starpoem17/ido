data/korean_processed/near_dedup 안의 parquet 파일들을 읽고 토크나이저를 생성한다.
data/tokenizers/korean_bbpe_v1 의 경로 형식으로 생성한 토크나이저 파일을 저장한다.
예를 들어, data/tokenizers/korean_bbpe_v1가 이미 존재한다면 data/tokenizers/korean_bbpe_v2 디렉토리를 생성하여 안에 결과물을 저장한다.
통일된 parquet 구조에서 content 필드의 String만 추출하여 토크나이저를 생성한다.
토크나이징 방식은 BBPE를 사용한다.
전체 데이터를 사용하여 토크나이저를 빌드한다.
토크나이저의 사전 크기는 48k를 디폴트로 둔다.
tokenizers 라이브러리의 ByteLevelBPETokenizer를 사용할 때, add_prefix_space=True 옵션과 trim_offsets=True 설정을 적용한다.

토크나이저 벤치마크는 다음과 같은 흐름으로 진행한다. data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/train-00000-of-00018.parquet 안에는 token_count라는 필드가 존재한다. 해당 프로젝트에서 빌드한 토크나이저의 토큰화 결과와 해당 parquet에 이미 입력된 token_count 결과를 비교하여 토크나이저의 성능을 확인한다.

토크나이저 빌드 시 가장 높은 빈도로 등장한 상위 1000개 토큰을 메타데이터로 저장. 그리고 바이트 형식과 함께 인간이 알아볼 수 있는 형태로도 저장해서 사용자가 읽고 어떤 토큰이 높은 빈도로 등장했는지 알기 쉽도록 한다.

사용자가 터미널에 텍스트를 입력하면 이를 받아 토큰화한 뒤 어느 정도 압축률을 보였는지 출력하는 코드 작성.