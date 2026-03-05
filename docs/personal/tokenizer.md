data/korean_raw에서 각 13개 폴더별로 2gb씩(폴더 하나가 2gb가 안 되는 경우 전체 선택) 랜덤 샘플하여 토크나이저 생성용 텍스트로 삼는다.
각 폴더의 데이터 구조가 전부 다르기에 duckdb의 read_json을 활용하여 value 부분만 긁어온다.
토크나이저는 tokenizers 라이브러리를 사용해 만든다.
토크나이징 방식은 BBPE를 사용한다.
tokenizers 라이브러리의 ByteLevelBPETokenizer를 사용할 때, add_prefix_space=True 옵션과 trim_offsets=True 설정을 적용한다.