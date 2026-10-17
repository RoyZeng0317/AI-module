"""SDXL —— 供 `/image sdxl` 手動選用的外部文字轉圖片模型。

CLAUDE.md 規則 #06 的顯式例外（跟 lib/NVIDIA.py 的 `/model nvidia`、
lib/IndexTTS2.py 的 `/voice indextts2` 同一個精神）：預設（`sinco`，見
tranning/imagegen/draw.py 的從零訓練卷積 VAE）完全不受影響，只有使用者
透過 `/image sdxl` 明確切換時才會被 import/呼叫。

為什麼需要這個例外：使用者想要的是 Image 2.0（OpenAI ChatGPT Images 2.0
或 xAI Grok Imagine Image 2.0，兩者皆為 2026 年的商用大型 diffusion
文字轉圖模型）那種效果——`draw.py` 是從零打造的卷積 VAE，架構上完全不支援
文字提示詞（見該檔案 docstring），而且 VAE 本身的像素級重建 loss 天生比
diffusion 模糊，這不是能從零訓練追上的規模差距（見 CLAUDE.md 這次的討論
脈絡）。SDXL（Stable Diffusion XL，Stability AI）是查證過目前能在單張 8GB
顯卡上跑的最成熟開源文字轉圖模型（社群 LoRA/ControlNet 資源最多，
Hugging Face 官方 `diffusers` 函式庫原生支援，不像 FLUX.2 系列那樣 VRAM
需求/授權條款眾說紛紜），用它作為真正開源、能本機跑的替代品。

跟 NVIDIA 雲端模式不同的地方：不是打 API，是**本機**載入 Stability AI
已經訓練好的開源權重——不是從零訓練，所以才需要 Rule 06 的明確例外。

安裝步驟（使用者自行在自己電腦上做一次，這支檔案本身不會自動下載任何
東西，也不會嘗試自動安裝）：
    pip install diffusers transformers accelerate safetensors invisible_watermark
第一次呼叫 sdxl_generate() 時，`diffusers` 會自動從 Hugging Face 下載
`stabilityai/stable-diffusion-xl-base-1.0` 的權重到 cache_dir（預設
lib/sdxl_checkpoints/，已經加進 .gitignore，不用手動下載/不會進版控）。

授權注意（私人使用在這個限制範圍內，但老實記錄下來）：SDXL 1.0 權重是
CreativeML Open RAIL++-M License——允許個人/研究使用，但附帶使用限制
條款（例如不能用來生成違法內容），完整條款見 Hugging Face 權重頁面；這個
專案是私人非商業使用。

硬體需求：fp16 在 1024x1024 解析度大約需要 8~10GB VRAM，在剛好 8GB 的卡
上偏緊——`sdxl_generate()` 預設開啟 `enable_model_cpu_offload()`（把暫時
用不到的子模組動態搬到系統 RAM，犧牲一點速度換取 VRAM 空間），跟這個
專案 RTX 4060 8GB 的算力預算相容，但推論時不能跟 sinco 自己的訓練同時
搶 VRAM。
"""

from pathlib import Path

MODEL = "stable-diffusion-xl-base-1.0"

# 固定指到這支檔案旁邊，而不是讓 diffusers 用它自己預設的 ~/.cache/huggingface
# ——跟 lib/IndexTTS2.py 的 DEFAULT_CHECKPOINT_DIR 同一個理由，保持這個專案
# 「外部模型權重都放在 lib/ 底下、固定路徑、.gitignore 排除」的一致慣例。
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "sdxl_checkpoints"

_pipeline = None


def _get_pipeline(cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR):
    """Lazily import diffusers + load the SDXL pipeline. Raises a friendly
    RuntimeError (not an ImportError the caller has to know to catch) if
    the `diffusers` package hasn't been installed yet — same contract as
    NVIDIA.py's _get_client() / IndexTTS2.py's _get_model().
    """
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers 套件未安裝，請先執行 `pip install diffusers transformers "
                "accelerate safetensors invisible_watermark`，再使用 /image sdxl"
            ) from exc

        _pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=str(cache_dir),
        )
        # 犧牲一點速度換 VRAM 空間，而不是直接 pipe.to("cuda") 把整個模型
        # 常駐在顯卡上——見檔案開頭「硬體需求」的說明，8GB 卡上偏緊。
        _pipeline.enable_model_cpu_offload()
    return _pipeline


def sdxl_generate(prompt: str, output_path: str | Path,
                   negative_prompt: str | None = None, num_inference_steps: int = 30,
                   cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> str:
    """用文字提示詞生成一張圖片，存成 output_path。回傳 output_path 字串——
    跟 tranning/imagegen/draw.py 的 generate() 類似的呼叫慣例，呼叫端
    （/image 指令）不用分兩套邏輯處理 sinco/sdxl 兩種模式的回傳值。
    """
    pipeline = _get_pipeline(cache_dir)
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt,
                      num_inference_steps=num_inference_steps).images[0]
    image.save(str(output_path))
    return str(output_path)
