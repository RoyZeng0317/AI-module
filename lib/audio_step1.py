"""
語音識別練習 第 1 步：波形 → 頻譜圖（純 NumPy 手寫）→ Mel 頻譜圖（librosa 對照）

安裝：pip install numpy scipy matplotlib librosa
用法：python audio_step1.py voice.wav
      （沒給檔案時會自動產生一段測試音）
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------- 1. 讀取聲音：聲音就是一長串數字 ----------
def load_audio(path=None, target_sr=16000):
    if path is None:
        # 產生測試音：前半秒 300Hz（低音），後半秒 1200Hz（高音）
        sr = target_sr
        t = np.linspace(0, 1, sr, endpoint=False)
        audio = np.where(t < 0.5, np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 1200 * t))
        audio += 0.05 * np.random.randn(sr)          # 加一點雜訊
        return sr, audio.astype(np.float32)

    sr, audio = wavfile.read(path)
    if audio.ndim == 2:                               # 雙聲道 → 取平均變單聲道
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio /= (np.abs(audio).max() + 1e-9)             # 正規化到 -1 ~ 1
    return sr, audio


# ---------- 2. 手寫頻譜圖：切片 + FFT ----------
def spectrogram(audio, sr, frame_ms=25, hop_ms=10):
    frame = int(sr * frame_ms / 1000)                 # 每段幾個取樣點（16kHz → 400）
    hop = int(sr * hop_ms / 1000)                     # 每次往前移多少（16kHz → 160）
    window = np.hanning(frame)                        # 讓每段頭尾平滑，減少雜訊

    n_frames = 1 + (len(audio) - frame) // hop
    frames = np.stack([audio[i * hop: i * hop + frame] * window for i in range(n_frames)])

    spec = np.abs(np.fft.rfft(frames, axis=1))        # 每段拆成各頻率的強度
    spec_db = 20 * np.log10(spec + 1e-6)              # 轉成分貝，接近人耳感受
    freqs = np.fft.rfftfreq(frame, d=1 / sr)
    return spec_db.T, freqs, hop                      # 轉置：縱軸頻率、橫軸時間


# ---------- 3. 畫圖對照 ----------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sr, audio = load_audio(path)
    print(f"取樣率 {sr} Hz，共 {len(audio)} 個數字，約 {len(audio) / sr:.2f} 秒")

    spec_db, freqs, hop = spectrogram(audio, sr)
    print(f"頻譜圖形狀：{spec_db.shape}（頻率格數, 時間格數）")

    n_plots = 3
    try:
        import librosa
        import librosa.display
    except ImportError:
        librosa = None
        n_plots = 2
        print("沒有安裝 librosa，略過 Mel 頻譜圖")

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    duration = len(audio) / sr

    axes[0].plot(np.arange(len(audio)) / sr, audio, linewidth=0.5)
    axes[0].set_title("① 波形：每秒上萬個振幅數字")
    axes[0].set_xlabel("時間 (秒)")
    axes[0].set_xlim(0, duration)

    axes[1].imshow(spec_db, origin="lower", aspect="auto",
                   extent=[0, spec_db.shape[1] * hop / sr, freqs[0], freqs[-1]], cmap="magma")
    axes[1].set_title("② 手寫頻譜圖：橫軸時間、縱軸頻率、越亮能量越強")
    axes[1].set_xlabel("時間 (秒)")
    axes[1].set_ylabel("頻率 (Hz)")

    if librosa is not None:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis="time",
                                 y_axis="mel", ax=axes[2], cmap="magma")
        axes[2].set_title("③ Mel 頻譜圖（80 格）：Whisper 等模型實際吃的輸入")

    plt.tight_layout()
    out = "audio_step1.png"
    plt.savefig(out, dpi=120)
    print(f"已存成 {out}")


if __name__ == "__main__":
    main()
