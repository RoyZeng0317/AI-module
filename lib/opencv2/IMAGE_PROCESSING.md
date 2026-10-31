# Lenna：for迴圈卷積與直方圖均衡化

本次依使用者確認採**獨立影像處理練習**。使用Pillow讀寫、像素容器與圖表組裝；數值運算均為Python迴圈，沒有NumPy、OpenCV、SciPy或內建濾鏡／均衡化。沒有修改或執行MNIST訓練。

## 執行

在專案根目錄執行；輸入可用這次提供的WebP，也可使用練習資料的TIF。

```powershell
# 同時處理提供的兩張圖；省略圖片時採專案練習的Lenna_RGB.tif
.venv/Scripts/python.exe lib/opencv2/convolution.py "C:/Users/roy/Downloads/Lenna_RGB.webp" "C:/Users/roy/Downloads/Lenna_Gray.webp"
# 依練習公式拆出Y/Cb/Cr，再均衡化Y
.venv/Scripts/python.exe lib/opencv2/histogram_equalization.py "C:/Users/roy/Downloads/Lenna_RGB.webp"
```

輸出位於 `output/image-processing/convolution/` 與 `output/image-processing/histogram/`。原圖512×512，參考灰階圖850×846；不縮放、不改來源，不能宣稱與參考圖逐像素一致。

## 卷積

`convolve(matrix, kernel, padding="edge")` 使用四層for迴圈逐像素／逐核元素累加，核先翻轉180度，是真正的數學卷積。預設複製邊界而維持尺寸；`--padding zero` 補零，`--padding valid` 不補邊。

依 `新手大禮包/旋積運算/旋積運算.docx` 的三個核：

```text
Filter 1       Filter 2                 Filter 3
1 0 1          5×5全1，再乘1/25         -1  0  1
0 4 0                                   -2  0  2
1 0 1                                   -1  0  1
```

Filter 1不擅自除以8；Filter 2是平均濾波；Filter 3翻核後符號與直接套Sobel核的互相關相反。RGB各通道分開處理；手動灰階使用0.299R+0.587G+0.114B，這個色彩轉換本身不是空間卷積。

- `matrix_valid.json`：練習MatrixA的完整原始結果，Filter1/3為6×5（高×寬）、Filter2為4×3。第一列分別為 `[2010,1989,2023,1998,2035]`、`[251.28,251.48,251.64]`（顯示取至小數點後2位）、`[-12,-3,5,-26,-9]`。
- 各圖片子目錄的 `rgb_filter*.json`／`gray_filter*.json`：不裁切、不四捨五入的原始數值，保留負值及超過255的值。
- `*_clipped.png`：只在顯示時裁切至0–255；Filter1可能因此大片變白，是原核係數的結果。
- `*_minmax.png`：只在顯示時把所有通道共同值域映射至0–255，不改原始JSON或核係數；常數矩陣的min–max顯示為0。
- `gray_original.png`：手動轉灰階，沒有套空間濾波。圖片結果保留各自來源尺寸。

## Y、Cb、Cr與均衡化

RGB直接表示紅／綠／藍；YCbCr分開亮度Y與藍色色差Cb、紅色色差Cr。Cb、Cr存成灰階圖片只是查看通道值，並非兩張不同的亮度照片。此處依 `直方圖均衡化程式練習.docx` 的有限範圍公式，不用Pillow內建YCbCr轉換：

```text
Y  = 16  + (65.738R + 129.057G + 25.064B) / 256
Cb = 128 + (-37.945R - 74.494G + 112.439B) / 256
Cr = 128 + (112.439R - 94.154G - 18.285B) / 256
```

逐像素四捨五入並限制至0–255；再用for迴圈統計Y的256個bins、累積CDF，建立映射：

```text
new_y = round_half_up((CDF[y] - CDF_min) * 255 / (像素總數 - CDF_min))
```

單一灰階的圖片保留原值，避免除以0。這是全域均衡化，不是CLAHE；離散映射不保證直方圖完全平坦，也不能只憑值域變大就保證視覺效果較好。

輸出 `*_Y.png`、`*_Cb.png`、`*_Cr.png`、`*_Y_equalized.png`、前後直方圖PNG及 `*_histogram.csv`（原計數、累積計數、LUT、均衡化後計數）。`*_comparison.png` 為原RGB／Y／均衡化Y對照。

本次Lenna_RGB.webp實測：262144個像素、Y範圍34–224，均衡化後0–255。

## 驗證

```powershell
.venv/Scripts/python.exe -m pytest lib/opencv2/test_convolution.py lib/opencv2/test_histogram_equalization.py --basetemp output/image-processing-tests -q
```

20項測試通過，含非對稱核、Sobel正負值、邊界、通道獨立、已知YCbCr像素、手算CDF與常數圖。獨立審查另核對Word OMML內的Filter2、實際圖片與原始JSON；單獨載入兩模組不會匯入NumPy／cv2／SciPy。工程與實際輸出檢查通過，視覺效果仍待使用者驗收。

## 二值化

同樣是獨立練習：Pillow只負責讀寫、像素容器與圖表組裝；灰階、直方圖、門檻判斷與Otsu均為Python迴圈，沒有NumPy、OpenCV、SciPy，也沒有呼叫Pillow的 `point`、`convert('1')`、`convert('L')` 或內建濾鏡。直方圖統計與圖表直接重用 `histogram_equalization.py` 的 `histogram()`／`histogram_chart()`。

```powershell
# 省略圖片時採卷積練習的Lenna_RGB.tif；可一次給多張
.venv/Scripts/python.exe lib/opencv2/binarization.py
.venv/Scripts/python.exe lib/opencv2/binarization.py lib/opencv2/koala.jpg "C:/Users/roy/Downloads/Lenna_RGB.webp" --threshold 100
```

**灰階**採卷積練習同一組係數，但這裡逐像素四捨五入（.5進位）成0–255整數，以整數 `(299R+587G+114B+500)//1000` 計算避免浮點誤差；不是直方圖練習的有限範圍Y（16–235）。灰階輸入直接沿用原值，RGBA忽略alpha。

**固定門檻**（預設128，`--threshold` 可改，允許0–256）：

```text
binary = 255  若 gray >= T
binary = 0    若 gray <  T
```

T本身算白（>=）。T=0全白、T=256全黑。

**Otsu**：用256個bins的直方圖，對每個T把 `[0, T-1]` 當背景、`[T, 255]` 當前景（與上面 >= 慣例一致），計算類間變異數：

```text
σ²_B(T) = w0·w1·(μ0 − μ1)² = (S0·N − S·n0)² / (n0·n1·N²)
```

n0/S0為背景像素數與灰階總和，N/S為全體。以 `fractions.Fraction` 精確比較，取σ²_B最大的T；**同分取最小T**。因為慣例是 >= T，此處的T比OpenCV（`> T`）的Otsu門檻大1。任一類為空時σ²_B定義為0、不做除法，所以**常數圖片**的256個σ²_B全為0，T=0，輸出全白。

輸出位於 `output/image-processing/binarization/<圖片檔名>/`：`gray.png`、`fixed_T<值>.png`、`otsu_T<值>.png`、`histogram.png`（紅線固定門檻、綠線Otsu）、`histogram.csv`（bin、count、以該bin為T時的類間變異數）、`comparison.png`（灰階／固定／Otsu並排）及 `report.json`。

本次實測Otsu門檻：Lenna_RGB.tif（512×512）T=118、Lenna_Gray.tif（850×846）T=120、koala.jpg（451×680）T=120。

已知限制：只做全域單一門檻，不是自適應（區域）門檻，光照不均時同一個T不適用整張圖；Otsu假設直方圖大致雙峰，Lenna實際是多峰，門檻只代表類間變異數最大的切點，不保證切出想要的物體。

```powershell
.venv/Scripts/python.exe -m pytest lib/opencv2/test_binarization.py --basetemp output/image-processing-tests -q
```

測試含手算灰階進位、門檻邊界（值恰為T）、雙峰矩陣的手算Otsu（T=21、σ²_B=1740²/324）、與逐T直接用定義計算的變異數比對、常數圖與小圖端到端輸出；並檢查原始碼沒有匯入NumPy／cv2／SciPy。工程測試通過，視覺效果仍待使用者驗收。
