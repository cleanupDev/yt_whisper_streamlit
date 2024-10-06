import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_whisper_pipe():
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # MPS currently supports float32
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )

    return pipe
