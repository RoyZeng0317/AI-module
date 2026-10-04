"""IndexTTS2 —— 供 `/voice indextts2` 手動選用的外部零樣本語音克隆模型。

CLAUDE.md 規則 #06 的顯式例外（跟 lib/NVIDIA.py 的 `/model nvidia` 同一個
精神）：預設（`sinco`，見 tranning/voice_clone.py）仍是本專案自行從零訓練
的 Tacotron 式語音克隆模型；這支檔案只在使用者透過 `/voice indextts2`
明確切換時才會被 import/呼叫，不影響任何預設路徑，不會被其他模式間接載入。

跟 NVIDIA 雲端模式不同的地方：NVIDIA.py 是打雲端 API，這支檔案是**本機**
載入一個別人已經訓練好的開源模型權重——不是從零訓練，所以才需要 Rule 06
的明確例外，而不是像 sinco 的程式邏輯一樣預設就能用。

安裝步驟（使用者自行在自己電腦上做一次，這支檔案本身不會自動下載任何東西，
也不會嘗試自動安裝；兩個目錄都已經列在 .gitignore，不會不小心進版控）：
    cd lib
    git clone https://github.com/index-tts/index-tts.git
    cd index-tts && pip install -e .
    cd ..
    pip install modelscope
    modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir indextts2_checkpoints

授權注意（私人使用在這個限制範圍內，但老實記錄下來）：index-tts 的程式碼
本身是 Apache 2.0，但預訓練權重另外附了一份 INDEX_MODEL_LICENSE，商業用途
需要另外取得書面授權——CLAUDE.md 這個專案目前是私人使用、不對外公布，不涉
及商業用途。

硬體需求（官方文件）：NVIDIA GPU + 約 6GB VRAM，跟這個專案 RTX 4060 8GB 的
算力預算相容，但推論時不能跟 sinco 自己的訓練同時搶 VRAM。
"""

from pathlib import Path

MODEL = "IndexTTS-2.5"

# 固定指到這支檔案旁邊，而不是單純的相對路徑 "checkpoints" —— 後者會因為
# Python 的執行目錄（cwd）不同而指到不同地方，呼叫端很容易在專案根目錄跑跟
# 在 lib/ 底下跑得到兩個不同結果。跟 modelscope download 時要給的 --local-dir
# 保持一致：
#     modelscope download --model IndexTeam/IndexTTS-2.5 \
#         --local_dir lib/indextts2_checkpoints
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "indextts2_checkpoints"

_model = None


def _get_model(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR):
    """Lazily import + load the IndexTTS2 checkpoint. Raises a friendly
    RuntimeError (not an ImportError/FileNotFoundError the caller has to
    know to catch) if the `indextts` package hasn't been installed or the
    checkpoint hasn't been downloaded yet — same contract as NVIDIA.py's
    _get_client() raising on a missing API key.
    """
    global _model
    if _model is None:
        try:
            from indextts.infer_v2 import IndexTTS2 as _IndexTTS2Impl
        except ImportError as exc:
            raise RuntimeError(
                "indextts 套件未安裝，請先依照 lib/IndexTTS2.py 開頭的安裝步驟安裝 "
                "index-tts 並下載 checkpoint，再使用 /voice indextts2"
            ) from exc

        checkpoint_dir = Path(checkpoint_dir)
        cfg_path = checkpoint_dir / "config.yaml"
        if not cfg_path.exists():
            raise RuntimeError(
                f"找不到 IndexTTS2 checkpoint：{checkpoint_dir}，請先用 modelscope "
                "download 下載（見 lib/IndexTTS2.py 開頭的安裝步驟）"
            )
        _model = _IndexTTS2Impl(cfg_path=str(cfg_path), model_dir=str(checkpoint_dir))
    return _model


def indextts2_speak(text: str, spk_audio_prompt: str | Path, output_path: str | Path,
                     checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> str:
    """用一段參考語音（spk_audio_prompt，幾秒鐘的 wav 就可以）模仿那個聲音唸出
    text，存成 output_path。回傳 output_path 字串——跟
    tranning/voice_clone.py 的 synthesize_to_file() 回傳約定一致，呼叫端
    （/voice 指令）不用分兩套邏輯處理 sinco/indextts2 兩種模式的回傳值。
    """
    model = _get_model(checkpoint_dir)
    model.infer(spk_audio_prompt=str(spk_audio_prompt), text=text, output_path=str(output_path))
    return str(output_path)
