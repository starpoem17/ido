from __future__ import annotations

from collections import deque
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from huggingface_hub import HfApi, get_token
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    HfApi = None
    get_token = None
    _HF_HUB_IMPORT_ERROR = exc
else:
    _HF_HUB_IMPORT_ERROR = None

try:
    from huggingface_hub.errors import GatedRepoError
except ImportError:  # pragma: no cover - runtime dependency guard
    GatedRepoError = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    AutoModelForImageTextToText = None
    AutoProcessor = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from . import jobs

SENTENCE_BREAK_CHARS = (".", "!", "?", "\n")


@dataclass
class FieldTranslationPlan:
    field_name: str
    requests: list["TranslationRequest"]
    used_chunking: bool
    chunk_count: int


@dataclass(frozen=True)
class TranslationRequest:
    row_index: int
    field_name: str
    chunk_index: int
    source_text: str
    source_lang_code: str
    target_lang_code: str
    max_output_tokens: int


@dataclass
class TranslationBatchResult:
    request: TranslationRequest
    translated_text: str
    retry_count: int


@dataclass
class LoadedTranslationEngine:
    model: Any
    processor: Any
    device: str
    model_id: str
    max_input_tokens: int
    auth_mode: str
    attention_backend: str
    flash_attention_available: bool
    flash_attention_requested: bool
    flash_attention_enabled_for_runtime: bool
    fallback_attention_backend: str


def _require_runtime_deps() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("torch is required for translation runtime") from _TORCH_IMPORT_ERROR
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError("transformers is required for translation runtime") from _TRANSFORMERS_IMPORT_ERROR
    if _HF_HUB_IMPORT_ERROR is not None:
        raise RuntimeError("huggingface_hub is required for gated model authentication") from _HF_HUB_IMPORT_ERROR


def _resolve_dtype() -> Any:
    if jobs.DTYPE != "bfloat16":
        raise RuntimeError(f"unsupported DTYPE={jobs.DTYPE}; this pipeline is pinned to bfloat16")
    return torch.bfloat16


def _torch_flash_attention_available() -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.backends.cuda, "is_flash_attention_available")
        and torch.backends.cuda.is_flash_attention_available()
    )


def _prepare_torch_flash_attention_state() -> bool:
    flash_available = _torch_flash_attention_available()
    if jobs.PREFER_TORCH_FLASH_ATTN and jobs.ATTN_IMPLEMENTATION != "sdpa":
        raise RuntimeError(
            f"ATTN_IMPLEMENTATION must be 'sdpa' when PREFER_TORCH_FLASH_ATTN is enabled, got {jobs.ATTN_IMPLEMENTATION!r}"
        )
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    return flash_available


@contextmanager
def _torch_flash_attention_only_context():
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION
        with torch.nn.attention.sdpa_kernel(backend, set_priority=True):
            yield
        return
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            yield
        return
    raise RuntimeError("PyTorch SDPA flash-only context is unavailable in this torch build")


def _is_local_model_path(model_id: str) -> bool:
    return Path(model_id).exists()


def _preflight_hf_access(model_id: str) -> tuple[str | None, str]:
    if _is_local_model_path(model_id):
        return None, "local-path"
    if not jobs.REQUIRE_HF_AUTH:
        return get_token(), "remote-no-preflight"

    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token is missing. "
            f"{model_id} is gated, so run `hf auth login` or set `HF_TOKEN` "
            "with an account that has accepted the model access request."
        )

    api = HfApi()
    try:
        api.model_info(model_id, token=token)
    except Exception as exc:
        if GatedRepoError is not None and isinstance(exc, GatedRepoError):
            raise RuntimeError(
                f"Access to gated repo `{model_id}` is not available for the current token. "
                "Approve access on Hugging Face and authenticate locally."
            ) from exc
        raise RuntimeError(
            f"Failed Hugging Face preflight for `{model_id}`. "
            "Check network connectivity, `hf auth login`, and model access approval."
        ) from exc
    return token, "remote-authenticated"


def _build_messages(source_text: str, source_lang_code: str, target_lang_code: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang_code,
                    "target_lang_code": target_lang_code,
                    "text": source_text,
                }
            ],
        }
    ]


def _build_model_inputs(
    engine: LoadedTranslationEngine,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> dict[str, Any]:
    messages = _build_messages(
        source_text=source_text,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
    )
    model_inputs = engine.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(engine.device) for key, value in model_inputs.items()}


def _build_batch_model_inputs(
    engine: LoadedTranslationEngine,
    requests: list[TranslationRequest],
) -> dict[str, Any]:
    conversations = [
        _build_messages(
            source_text=request.source_text,
            source_lang_code=request.source_lang_code,
            target_lang_code=request.target_lang_code,
        )
        for request in requests
    ]
    model_inputs = engine.processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    return {key: value.to(engine.device) for key, value in model_inputs.items()}


def _probe_flash_runtime_support(
    model: Any,
    processor: Any,
    *,
    device: str,
    max_input_tokens: int,
) -> bool:
    if not jobs.PREFER_TORCH_FLASH_ATTN:
        return False
    if not _torch_flash_attention_available():
        return False

    probe_engine = LoadedTranslationEngine(
        model=model,
        processor=processor,
        device=device,
        model_id=jobs.MODEL_ID,
        max_input_tokens=max_input_tokens,
        auth_mode="probe",
        attention_backend=jobs.ATTN_IMPLEMENTATION,
        flash_attention_available=True,
        flash_attention_requested=jobs.PREFER_TORCH_FLASH_ATTN,
        flash_attention_enabled_for_runtime=False,
        fallback_attention_backend=jobs.FALLBACK_ATTENTION_BACKEND,
    )
    try:
        model_inputs = _build_model_inputs(
            engine=probe_engine,
            source_text="hello",
            source_lang_code="en",
            target_lang_code="ko",
        )
        input_length = int(model_inputs["input_ids"].shape[-1])
        if input_length > max_input_tokens - jobs.MIN_RESERVED_OUTPUT_TOKENS:
            return False
        with torch.inference_mode():
            with _torch_flash_attention_only_context():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                )
        del output_ids
        del model_inputs
        gc.collect()
        return True
    except Exception:
        gc.collect()
        return False


def load_translation_engine() -> LoadedTranslationEngine:
    _require_runtime_deps()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this translation pipeline")

    flash_attention_available = _prepare_torch_flash_attention_state()
    token, auth_mode = _preflight_hf_access(jobs.MODEL_ID)
    model_kwargs: dict[str, Any] = {
        "dtype": _resolve_dtype(),
        "trust_remote_code": True,
    }
    if jobs.ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = jobs.ATTN_IMPLEMENTATION
    if token is not None:
        model_kwargs["token"] = token

    processor = AutoProcessor.from_pretrained(
        jobs.MODEL_ID,
        trust_remote_code=True,
        token=token,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(jobs.MODEL_ID, **model_kwargs)
    model.to(jobs.DEVICE)
    model.eval()

    flash_attention_enabled_for_runtime = _probe_flash_runtime_support(
        model=model,
        processor=processor,
        device=jobs.DEVICE,
        max_input_tokens=jobs.MAX_INPUT_TOKENS,
    )
    if jobs.PREFER_TORCH_FLASH_ATTN and not flash_attention_enabled_for_runtime and not jobs.ALLOW_FLASH_FALLBACK:
        raise RuntimeError(
            "PyTorch built-in Flash Attention was preferred but could not be used for TranslateGemma runtime inputs, "
            "and fallback is disabled."
        )
    attention_backend = jobs.FLASH_BACKEND_NAME if flash_attention_enabled_for_runtime else jobs.FALLBACK_ATTENTION_BACKEND

    return LoadedTranslationEngine(
        model=model,
        processor=processor,
        device=jobs.DEVICE,
        model_id=jobs.MODEL_ID,
        max_input_tokens=jobs.MAX_INPUT_TOKENS,
        auth_mode=auth_mode,
        attention_backend=attention_backend,
        flash_attention_available=flash_attention_available,
        flash_attention_requested=jobs.PREFER_TORCH_FLASH_ATTN,
        flash_attention_enabled_for_runtime=flash_attention_enabled_for_runtime,
        fallback_attention_backend=jobs.FALLBACK_ATTENTION_BACKEND,
    )


def estimate_input_tokens(
    engine: LoadedTranslationEngine,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> int:
    model_inputs = _build_model_inputs(
        engine=engine,
        source_text=source_text,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
    )
    token_count = int(model_inputs["input_ids"].shape[-1])
    del model_inputs
    gc.collect()
    return token_count


def _effective_input_limit() -> int:
    return jobs.MAX_INPUT_TOKENS - jobs.MIN_RESERVED_OUTPUT_TOKENS


def can_fit_single_call(
    engine: LoadedTranslationEngine,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> bool:
    return (
        estimate_input_tokens(
            engine=engine,
            source_text=source_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
        )
        <= _effective_input_limit()
    )


def _split_into_paragraph_units(source_text: str) -> list[str]:
    if not source_text:
        return [source_text]
    units: list[str] = []
    current_lines: list[str] = []
    for line in source_text.splitlines(keepends=True):
        current_lines.append(line)
        if not line.strip():
            units.append("".join(current_lines))
            current_lines = []
    if current_lines:
        units.append("".join(current_lines))
    return units or [source_text]


def _split_long_unit(unit: str) -> list[str]:
    if len(unit) <= 1:
        return [unit]

    line_units = [part for part in unit.splitlines(keepends=True) if part]
    if len(line_units) > 1:
        return line_units

    sentence_units: list[str] = []
    start = 0
    for idx, char in enumerate(unit):
        if char in SENTENCE_BREAK_CHARS:
            sentence_units.append(unit[start : idx + 1])
            start = idx + 1
    if start < len(unit):
        sentence_units.append(unit[start:])
    sentence_units = [part for part in sentence_units if part]
    if len(sentence_units) > 1:
        return sentence_units

    fallback_size = min(1000, max(1, len(unit) // 2))
    return [unit[idx : idx + fallback_size] for idx in range(0, len(unit), fallback_size)]


def split_text_into_chunks(
    engine: LoadedTranslationEngine,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> list[str]:
    if not source_text:
        return [source_text]

    units = _split_into_paragraph_units(source_text)
    chunks: list[str] = []
    current_parts: list[str] = []
    idx = 0

    while idx < len(units):
        unit = units[idx]
        candidate = "".join(current_parts) + unit
        if candidate and can_fit_single_call(
            engine=engine,
            source_text=candidate,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
        ):
            current_parts.append(unit)
            idx += 1
            continue

        if current_parts:
            chunks.append("".join(current_parts))
            current_parts = []
            continue

        smaller_units = _split_long_unit(unit)
        if len(smaller_units) == 1 and smaller_units[0] == unit:
            raise ValueError("source text contains a unit too long to fit even after fallback splitting")
        units[idx : idx + 1] = smaller_units

    if current_parts:
        chunks.append("".join(current_parts))

    if not chunks:
        raise ValueError("failed to split source text into safe chunks")
    return chunks


def _max_output_tokens_for_field(field_name: str) -> int:
    if field_name == "problem":
        return jobs.MAX_OUTPUT_TOKENS_PROBLEM
    if field_name == "thinking":
        return jobs.MAX_OUTPUT_TOKENS_THINKING
    if field_name == "solution":
        return jobs.MAX_OUTPUT_TOKENS_SOLUTION
    raise ValueError(f"unsupported field: {field_name}")


def validate_translation_text(translated_text: str) -> None:
    if not isinstance(translated_text, str):
        raise ValueError("translated_text must be a string")
    if not translated_text.strip():
        raise ValueError("translated_text is empty")


def _decode_generated_ids(engine: LoadedTranslationEngine, generated_ids: Any) -> str:
    if hasattr(engine.processor, "decode"):
        decoded = engine.processor.decode(generated_ids, skip_special_tokens=True)
    else:
        decoded = engine.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded.replace("\r\n", "\n").strip()


def _generation_pad_token_id(engine: LoadedTranslationEngine) -> int | None:
    tokenizer = getattr(engine.processor, "tokenizer", None)
    if tokenizer is None:
        return None
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def _generate_batch_once(
    engine: LoadedTranslationEngine,
    requests: list[TranslationRequest],
) -> list[str]:
    if not requests:
        return []

    model_inputs = _build_batch_model_inputs(engine=engine, requests=requests)
    input_width = int(model_inputs["input_ids"].shape[-1])
    if input_width > _effective_input_limit():
        raise ValueError("formatted chat input exceeded the safe input token budget")

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max(request.max_output_tokens for request in requests),
        "do_sample": jobs.DO_SAMPLE,
        "use_cache": True,
    }
    if jobs.DO_SAMPLE:
        generation_kwargs["temperature"] = jobs.TEMPERATURE
    pad_token_id = _generation_pad_token_id(engine)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        if engine.flash_attention_enabled_for_runtime:
            with _torch_flash_attention_only_context():
                output_ids = engine.model.generate(**model_inputs, **generation_kwargs)
        else:
            output_ids = engine.model.generate(**model_inputs, **generation_kwargs)

    generated_token_batches = output_ids[:, input_width:]
    translated_texts = [
        _decode_generated_ids(engine, generated_token_batches[batch_index])
        for batch_index in range(len(requests))
    ]

    del generated_token_batches
    del output_ids
    del model_inputs
    gc.collect()

    return translated_texts


def plan_field_requests(
    engine: LoadedTranslationEngine,
    row_index: int,
    field_name: str,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> FieldTranslationPlan:
    if not source_text:
        return FieldTranslationPlan(
            field_name=field_name,
            requests=[],
            used_chunking=False,
            chunk_count=0,
        )

    max_output_tokens = _max_output_tokens_for_field(field_name)
    if can_fit_single_call(
        engine=engine,
        source_text=source_text,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
    ):
        return FieldTranslationPlan(
            field_name=field_name,
            requests=[
                TranslationRequest(
                    row_index=row_index,
                    field_name=field_name,
                    chunk_index=0,
                    source_text=source_text,
                    source_lang_code=source_lang_code,
                    target_lang_code=target_lang_code,
                    max_output_tokens=max_output_tokens,
                )
            ],
            used_chunking=False,
            chunk_count=1,
        )

    chunk_texts = split_text_into_chunks(
        engine=engine,
        source_text=source_text,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
    )
    return FieldTranslationPlan(
        field_name=field_name,
        requests=[
            TranslationRequest(
                row_index=row_index,
                field_name=field_name,
                chunk_index=chunk_index,
                source_text=chunk_text,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
                max_output_tokens=max_output_tokens,
            )
            for chunk_index, chunk_text in enumerate(chunk_texts)
        ],
        used_chunking=True,
        chunk_count=len(chunk_texts),
    )


def translate_requests_batch(
    engine: LoadedTranslationEngine,
    requests: list[TranslationRequest],
) -> list[TranslationBatchResult]:
    if not requests:
        return []

    retry_counts = [0 for _ in requests]
    results: list[TranslationBatchResult | None] = [None for _ in requests]
    work_queue: deque[list[int]] = deque()
    batch_size = max(1, jobs.MAX_REQUESTS_PER_BATCH)
    for start in range(0, len(requests), batch_size):
        work_queue.append(list(range(start, min(start + batch_size, len(requests)))))

    while work_queue:
        batch_indices = work_queue.popleft()
        batch_requests = [requests[index] for index in batch_indices]

        try:
            translated_texts = _generate_batch_once(engine=engine, requests=batch_requests)
        except Exception:
            if len(batch_indices) > 1:
                midpoint = len(batch_indices) // 2
                work_queue.appendleft(batch_indices[midpoint:])
                work_queue.appendleft(batch_indices[:midpoint])
                continue

            request_index = batch_indices[0]
            retry_counts[request_index] += 1
            if retry_counts[request_index] > jobs.RETRY_LIMIT:
                raise
            gc.collect()
            work_queue.append(batch_indices)
            continue

        failed_singletons: list[list[int]] = []
        for request_index, translated_text in zip(batch_indices, translated_texts, strict=True):
            try:
                validate_translation_text(translated_text)
            except Exception:
                retry_counts[request_index] += 1
                if retry_counts[request_index] > jobs.RETRY_LIMIT:
                    raise
                failed_singletons.append([request_index])
                continue

            results[request_index] = TranslationBatchResult(
                request=requests[request_index],
                translated_text=translated_text,
                retry_count=retry_counts[request_index],
            )

        for failed_singleton in reversed(failed_singletons):
            gc.collect()
            work_queue.appendleft(failed_singleton)

    if any(result is None for result in results):
        raise RuntimeError("missing translation batch result after retries")
    return [result for result in results if result is not None]
