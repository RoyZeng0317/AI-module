"""LTX-Video —— 供 `/video ltxvideo` 手動選用的外部文字轉影片模型。

CLAUDE.md 規則 #06 的顯式例外（跟 lib/NVIDIA.py 的 `/model nvidia`、
lib/IndexTTS2.py 的 `/voice indextts2`、lib/SDXL.py 的 `/image sdxl`
同一個精神）：這個專案目前完全沒有任何影片生成的架構（不是「現有的東西
不夠好」，是這個能力本來就是 0），所以沒有像 sinco/draw.py 那樣的預設
從零訓練選項可以切換回去——`/video` 指令只有 `ltxvideo` 這一個模式。

為什麼需要這個例外：使用者想要的是 Seedance（ByteDance Seed 團隊的
diffusion 文字/圖片轉影片模型，2026 年版本原生支援音訊-影片聯合生成）
那種效果。LTX-Video（Lightricks）是查證過目前能在單張 8GB 顯卡上跑的
最低門檻開源文字轉影片模型（官方 Hugging Face 頁面確認 8GB+ 可跑；比較
之下 HunyuanVideo 原版要 24GB、Wan 2.1 14B 要 16GB 以上），而且已經被
Hugging Face `diffusers` 函式庫原生收錄（`LTXPipeline`），不用額外 clone
一份單獨的推論程式碼庫、不用擔心它自己 repo 那套 API 版本之間常常改介面
的問題（查證時發現這件事：LTX-2/2.5 的原生推論套件 CLI 參數/設定檔名稱
在不同版本之間變動過，diffusers 這條路線比較穩定）。

跟 NVIDIA 雲端模式不同的地方：不是打 API，是**本機**載入 Lightricks
已經訓練好的開源權重——不是從零訓練，所以才需要 Rule 06 的明確例外。

安裝步驟（使用者自行在自己電腦上做一次，這支檔案本身不會自動下載任何
東西，也不會嘗試自動安裝；版本選 0.9 系列而非較新的 LTX-2/2.5，因為後者
官方文件列的最低需求是 32GB VRAM，遠超這個專案的預算）：
    pip install diffusers transformers accelerate safetensors imageio imageio-ffmpeg
第一次呼叫 ltxvideo_generate() 時，`diffusers` 會自動從 Hugging Face
下載 `Lightricks/LTX-Video` 的權重到 cache_dir（預設
lib/ltxvideo_checkpoints/，已經加進 .gitignore，不用手動下載/不會進
版控）。

授權注意（私人使用在這個限制範圍內，但老實記錄下來）：LTX-Video 的程式碼
是 Apache 2.0，但模型權重另外附了一份 LTX 社群授權——年營收低於 1000 萬
美元的組織可以免費商業使用，超過門檻或要轉散佈微調版本需要另外付費授權
（完整條款見 Hugging Face 權重頁面的 LICENSE 檔案）；這個專案是私人非
商業使用。

硬體需求：官方模型卡預設解析度 704x480、161 幀、50 步，8GB VRAM 可跑但
會偏慢（影片生成本來就比單張圖片慢很多）——不能跟 sinco 自己的訓練同時
搶 VRAM。
"""

from pathlib import Path

MODEL = "LTX-Video"

# 跟 lib/IndexTTS2.py 的 DEFAULT_CHECKPOINT_DIR / lib/SDXL.py 的
# DEFAULT_CHECKPOINT_DIR 同一個理由：固定路徑、放在 lib/ 底下、.gitignore
# 排除，不依賴 diffusers 自己預設的 ~/.cache/huggingface。
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "ltxvideo_checkpoints"

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

_pipeline = None


def _get_pipeline(cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR):
    """Lazily import diffusers + load the LTX-Video pipeline. Raises a
    friendly RuntimeError (not an ImportError the caller has to know to
    catch) if the `diffusers` package hasn't been installed yet — same
    contract as NVIDIA.py's _get_client() / IndexTTS2.py's _get_model() /
    SDXL.py's _get_pipeline().
    """
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from diffusers import LTXPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers 套件未安裝，請先執行 `pip install diffusers transformers "
                "accelerate safetensors imageio imageio-ffmpeg`，再使用 /video ltxvideo"
            ) from exc

        _pipeline = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16, cache_dir=str(cache_dir),
        )
        _pipeline.enable_model_cpu_offload()
    return _pipeline


def ltxvideo_generate(prompt: str, output_path: str | Path,
                       negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                       width: int = 704, height: int = 480, num_frames: int = 161,
                       num_inference_steps: int = 50, fps: int = 24,
                       cache_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> str:
    """用文字提示詞生成一段影片，存成 output_path（.mp4）。回傳 output_path
    字串——呼叫端（/video 指令）不用另外處理回傳值的形狀。
    """
    from diffusers.utils import export_to_video

    pipeline = _get_pipeline(cache_dir)
    frames = pipeline(prompt=prompt, negative_prompt=negative_prompt, width=width,
                       height=height, num_frames=num_frames,
                       num_inference_steps=num_inference_steps).frames[0]
    export_to_video(frames, str(output_path), fps=fps)
    return str(output_path)
