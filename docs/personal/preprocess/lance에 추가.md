[[lance에 넣기전.md]]에서 받은 row들을 lance에 넣기 위한 md파일이다. 

1. lance 스키마: 
    (1): 다른 필드는 기존 parquet의 필드처럼 넣는다. 여기서 주의해야 하는 것은 기존 message가 List[Dict] 형식에서 seralized string으로 바꾸어야 한다. Pt는 content기준으로, sft, reasoning은 message 기준으로 토큰 카운트 필드에 값을 넣는다는 것을 명심하자. 
2. 샤딩 규칙
    (1): 샤딩 목표는 하위파일에서 명시한 것 처럼 1기가다. 
3. append 단위 
    (1): source별로 append를 진행하자. 
    (2): 중간 실패시 shard 단위로 재시작 진행
4. 메타데이터
    (1) shard별 row의 개수, byte 크기
    (2) source별 row의 개수, 총 byte, 총 token 개수
    (3) data_usage별 row의 개수
    (4) split별 row의 개수
