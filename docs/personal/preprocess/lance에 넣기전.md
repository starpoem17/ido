1. parquet을 불러온다
2. parquet 속 `data_usage`를 살펴보며 `SFT`, `PT`, `REASONING` 중 어느 feature인지 살펴본다.
3. `data_usage`가 `SFT`인 필드
	(1) : messages 필드를 살펴본다.
	(1-1) : parquet 단계의 messages 필드는 `[{role, content}, ...]` 구조의 list이지만, Lance 적재 직전에는 특수토큰이 반영된 serialized string으로 바꾸어 같은 messages 필드에 저장한다.
	(2): 
	'messages': [[{'content': '당신은 기술과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다.', 'role': 'system'}, {'content': '안녕! 만나서 반가워.나 뭣 좀 물어봐도 될까?', 'role': 'user'}, {'content': '안녕하세요! 만나서 반가워요. 무엇을 도와드릴까요?', 'role': 'assistant'}, {'content': '요즘 남아프리카에 대해 관심이 많아져서. 특히 칼튼 센터에 대해 궁금한 것이 있어.', 'role': 'user'}, {'content': '칼튼 센터는 아프리카에서 가장 높은 사무실 건물로 알려져 있습니다. 50층의 건물이며 사무실과 상점을 모두 지니고 있습니다. 또한, 지하에는 180개가 넘는 상점이 있는 쇼핑 센터가 연결되어 있습니다.', 'role': 'assistant'}, {'content': '와, 그렇구나. 근데 칼튼 센터가 누구에 의해 설계되었는지 알 수 있을까?', 'role': 'user'}, {'content': '네, 칼튼 센터는 미국 건축 회사인 스키드모어, 오윙스 앤드 메릴에 의해 설계되었습니다. 발굴 작업은 1967년에 시작되어 완료되기까지 2년이 걸렸습니다.', 'role': 'assistant'}, {'content': '그럼, 칼튼 센터의 주요 시설은 무엇이 있어?', 'role': 'user'}, {'content': "칼튼 센터는 오피스와 쇼핑몰, 전망대, 호텔 등이 주로 갖추어져 있습니다. 특히 최상층인 50층에는 '탑 오브 아프리카'라는 명칭의 전망대가 가설되어 있습니다.", 'role': 'assistant'}]] 이런 식으로 구성되어 있음
	여기서 이제 특수토큰들을 넣어 SFT에 바로 넣을 수 있는 구조로 변경
	<|bos|><|system|>system내용<|eot_id|><|user|>user1<|eot_id|><|assistant|>assistant1<|eot_id|><|user|>user2<|eot_id|><|assistant|>assistant2<|eot_id|>...<|eos|> 
	(3) 다시 이 messages 기준으로 token count 진행
	(4) 만약 이 messages가 960보다 크다면 messages 분리 진행
	(4-1) token count에 대해서 960로 나누어서 대략 몇 개의 row로 나눌지 계산한다.
	(4-2) messages 분리 방법 : 목표 row 수에 맞추어 turn(user-assistant) pair 개수를 토대로 최대한 균등하게 turn 횟수를 나누어서 messages를 분리한다. 여기서는 token 길이에 최대한 맞추는 greedy 방식이 아니라 pair 개수 기준 균등 분할을 우선한다. 무조건 assisant의 내용이 message의 끝이 되게 나누어야 함. 
	(4-3) messages를 분리하고 다시 token count를 진행하여 960를 넘지 않는지 확인한다.
	(4-4) 균등 분할 후에도 960를 넘는 messages chunk가 있으면 같은 규칙으로 다시 분할한다.
	(5) 만약 token count가 960을 넘지 않는지 확인하고 row를 기존 row 대신에 추가한다. 여기서 명심해야 하는 건 content 필드 속 내용은 기존 content 필드에서 현재 messages 필드 속 내용을 대상으로 content 분리 + messages를 나누었기 때문에 나눈 기준에 맞추어서 각 분할 row마다 <|bos|><|system|>system내용<|eot_id|>을 다시 추가해서 messages 필드를 구성하기
	(5-1) 분할된 `SFT` row의 content는 해당 chunk에 포함된 user/assistant turn만 대상으로 하며, system은 제외하고 원순서를 유지한 채 공백 한 칸으로 이어붙여 다시 구성한다.
	(6) token_count 필드에는 messages 속의 token 개수를 기준으로 재계산해서 넣기
4. `data_usage`가 `PT`인 필드
	(1): content 필드를 확인한다.
	(2): token_count 필드를 확인하여 1024보다 큰지 살펴본다.
	(3) 만약 token_count 필드 속 feature가 1024보다 크다면 token_count 필드 속 feature를 960으로 나누어서 대략 몇 개의 row로 나눌지 판단
	(4) content 분리 방법 : 온점을 기준으로 나누면서 대략 960~1024 안에 한 row 속 content가 들어갈 수 있게 분리한다.
	(5) 그렇게 분리한 row들을 기존 row 대신에 추가한다. (`PT`에 대해서는 `messages = null`이기에 messages는 고려하지 말자)
	(6) 각 row에 대한 token_count는 변경된 row의 content를 기준으로 계산하여 필드에 값을 넣는다.

5. `data_usage`가 `REASONING`인 필드
	(1): reasoning row는 Lance 적재 직전에 `PT` 파생 row와 `REASONING` 파생 row로 나누어 처리한다.
	(2): `PT` 파생 row는 content를 PT 학습용으로 사용하며, content 길이가 1024를 넘지 않도록 위 `PT` 규칙과 동일한 방식으로 분리한다.
	(3): `PT` 파생 row는 `data_usage = PT`, `messages = null`로 저장하고 token_count는 content 기준으로 계산한다.
	(4): `REASONING` 파생 row는 reasoning에 사용될 전체 messages 길이를 유지하며 Lance 적재 시 별도 분할을 하지 않는다.
	(5): `REASONING` 파생 row는 `data_usage = REASONING`, `content = null`로 저장한다.
	(6): `REASONING` 파생 row의 messages는 `SFT`와 같은 방식으로 특수토큰 직렬화를 적용한다.
	(7): `REASONING` 파생 row의 token_count는 messages 기준으로 계산한다.
	
6. split 필드는 기존 파생된 필드에 유지해서 넣자. 

7. parquet 샤딩 문제를 해결하기 위해서 
	(1): 위에 명시한 방법 그대로 row를 변환하면 변환이 끝난 row마다 bytes를 계산한다. 그리고 현재 shard에 계속 row를 담다가 (목표크기=1gb)를 넘기 전에 flush한다. 
	(2): 새 shard를 열고 계속 이어서 작성

