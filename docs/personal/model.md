torch.compile 사용

GQA 사용

Gated attn

SwigGLU

RMSNorm

RoPE
    1024로 사전 학습. 이후 8192로 확장

torch 내장 flash attn 친화적 설계
    torch.backends.cuda.is_flash_attention_available()
    torch.backends.cuda.can_use_flash_attention(params, debug=True)으로 점검

bf16