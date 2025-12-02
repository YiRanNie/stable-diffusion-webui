import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"

print(f">>> [PromptOptimizer] Loading local LLM model: {MODEL_ID} (first run may take a while)...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
)

model.to(DEVICE)
model.eval()

print(f">>> [PromptOptimizer] Model loaded on {DEVICE}: {MODEL_ID}")


SYSTEM_PROMPT = """
You rewrite short English prompts for image generation.

Your task:

1. Rewrite the user's sentence into a slightly more detailed English description.
   - Keep all original objects, quantities, and spatial relationships.
   - Do NOT change the meaning.
   - Do NOT add new objects.
   - Add only light, neutral visual details (simple lighting, general environment).

2. Provide 3–5 short descriptive phrases.
   - Each 1–3 words.
   - Only neutral visual descriptors (lighting, mood, clarity).
   - Do NOT repeat any subjects from the original sentence.
   - Do NOT use dramatic or strong artistic styles.

3. Output ONLY JSON in this exact format:

{
  "refined_sentence": "...",
  "tags": ["...", "...", "..."]
}

Example output:

{
  "refined_sentence": "Two cats sit on the left side of a dog in a calm outdoor setting with gentle light.",
  "tags": [
    "soft lighting",
    "natural setting",
    "subtle detail"
  ]
}

English only. No explanation.
"""


def build_llm_input(user_prompt: str) -> str:
    return f'User prompt: "{user_prompt}"\nReturn JSON now:'


def _call_raw_llm(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    gen = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return gen.strip()


def call_llm(user_prompt: str) -> dict:
    raw = _call_raw_llm(build_llm_input(user_prompt))

    try:
        s = raw.index("{")
        e = raw.rindex("}") + 1
        raw = raw[s:e]
    except Exception:
        return {"refined_sentence": user_prompt, "tags": []}

    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    data.setdefault("refined_sentence", user_prompt)
    data.setdefault("tags", [])

    return data


def ensure_english(text: str) -> str:
    return re.sub(r"[\u4e00-\u9fff]+", "", text).strip()


def normalize_tags(tags):
    clean = []
    for t in tags:
        if not isinstance(t, str):
            t = str(t)
        t = ensure_english(t).strip()
        if t and t.lower() not in {x.lower() for x in clean}:
            clean.append(t)
    return clean[:5]


def stitch_prompt(base: str, data: dict) -> str:
    refined = ensure_english(data.get("refined_sentence", "").strip()) or base
    tags = normalize_tags(data.get("tags", []))
    if tags:
        return refined + ", " + ", ".join(tags)
    return refined


def optimize_prompt_llm(user_prompt: str) -> str:
    data = call_llm(user_prompt)
    return stitch_prompt(user_prompt, data)


DEFAULT_NEGATIVE = (
    "low quality, blurry, distorted, bad anatomy, extra limbs, extra fingers, "
    "wrong perspective, depth errors, overexposed, underexposed, "
    "watermark, text, logo"
)

NEGATIVE_SYSTEM_PROMPT = """
You write negative prompts for Stable Diffusion.

Input: a short English prompt that describes what the user WANTS.

Output: a comma-separated list of 8–16 short phrases describing things to AVOID in the image.

Rules:
- Do NOT repeat the main subjects from the user's prompt (for example: girl, cat, dog, city).
- Do NOT describe the scene itself.
- Only write visual problems and artifacts: low quality, blur, distortion, bad anatomy, extra limbs or fingers, wrong perspective, exposure problems, noise, text, watermark, logo.
- Each item should be 1–3 words.
- English only. No explanation. No extra text.

Example:

User prompt: "A girl sitting on a bench in a park"
Your answer:
"blurry, low quality, distorted, bad anatomy, extra fingers, bad hands, depth errors, wrong perspective, overexposed, underexposed, watermark, text, logo"
"""


def generate_negative_prompt(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": NEGATIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'User prompt: "{user_prompt}"\nAnswer with only the negative prompt:',
        },
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    text = ensure_english(text.replace("\n", " ")).strip()

    lower = text.lower()
    for prefix in ["negative prompt:", "negative:", "negatives:"]:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if text == "" or text.lower() == user_prompt.lower():
        return DEFAULT_NEGATIVE

    return text


STYLE_PRESETS = {
    "None": "",
    "Photorealistic": (
        "photorealistic, 8k uhd, ultra detailed, realistic lighting, natural skin texture"
    ),
    "Anime": (
        "anime style, clean lineart, vibrant colors, flat shading, 2d illustration"
    ),
    "Illustration": (
        "digital illustration, painterly, detailed brushwork, soft shading, concept art"
    ),
    "Cinematic": (
        "cinematic lighting, film still, dramatic contrast, wide shot, anamorphic lens bokeh"
    ),
    "3D Render": (
        "3d render, octane render, highly detailed, global illumination, physically based rendering"
    ),
}


def apply_style_to_prompt(prompt: str, style_name: str) -> str:
    if not prompt:
        return prompt
    style_key = style_name or "None"
    style_text = STYLE_PRESETS.get(style_key, "").strip()
    if not style_text:
        return prompt
    if style_text.lower() in prompt.lower():
        return prompt
    return prompt + ", " + style_text


def optimize_prompt_with_style(user_prompt: str, style_name: str) -> str:
    base = optimize_prompt_llm(user_prompt)
    return apply_style_to_prompt(base, style_name)
