# Quartus II 鍵盤電路設計教學

**Date**: August 23, 2026 at 03:15 PM
**Turns**: 39
**Source**: [Gemini Chat](https://gemini.google.com/app/0f4cbe3078f03c47)

---

## Turn 1

### 👤 使用者

我想要嘗試使用quartus13.0設計鍵盤的電路
使用一般的 or gate or and gate, etc.


## Turn 2

### 👤 使用者

等等
Y1 = K3+K2是怎麼來的?


## Turn 3

### 👤 使用者

我這個設計要使用什麼樣的IC?有CPLD、FPGA等多款IC


## Turn 4

### 👤 使用者

是二對七的解碼器嗎?

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 5

### 👤 使用者

等一下，四變數的真值表我要怎麼畫?

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 6

### 👤 使用者

要有連結導向

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 7

### 👤 使用者

我需要的是完整的ic io port的圖

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 8

### 👤 使用者

我覺得用一般的波形圖不夠直觀
好像還是要有FPGA的實體機器才能清楚多一些

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 9

### 👤 使用者

Info: ***********************
Info: Running Quartus II 32-bit Analysis & Synthesis
Info: Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Info: Processing started: Sun Aug 23 09:31:59 2026
Info: Command: quartus_map --read_settings_files=on --write_settings_files=off FPGA_Mod16_Scanner -c FPGA_Mod16_Scanner
Warning (20028): Parallel compilation is not licensed and has been disabled
Info (12021): Found 1 design units, including 1 entities, in source file fpga_mod16_scanner.bdf
Info (12023): Found entity 1: FPGA_Mod16_Scanner
Info (12127): Elaborating entity "FPGA_Mod16_Scanner" for the top level hierarchy
Warning (275080): Converted elements in bus name "Q" using legacy naming rules. Make any assignments on the new names, not on the original names.
Warning (275081): Converted element name(s) from "Q[3..0]" to "Q3..0"
Error (275024): Width mismatch in port "IN2" of instance "inst4" and type AND2 -- source is ""Q[3..0]""
Error (12153): Can't elaborate top-level user hierarchy
Error: Quartus II 32-bit Analysis & Synthesis was unsuccessful. 2 errors, 3 warnings
Error: Peak virtual memory: 374 megabytes
Error: Processing ended: Sun Aug 23 09:31:59 2026
Error: Elapsed time: 00:00:00
Error: Total CPU time (on all processors): 00:00:00
Error (293001): Quartus II Full Compilation was unsuccessful. 4 errors, 3 warnings
我知道Q[3..0]是錯誤的，但是有Q1-Q4 It's named在74161的地方QA-QD，協助處理

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 10

### 👤 使用者

現在我在waveform波型模擬測試出現的結果總和為254, 253, 251, 254這樣

### 🤖 助理

對於這個**基礎鍵盤編碼器（鍵盤解碼）的電路設計，選擇哪一種 IC 取決於你這個專案的階段目標**（教學練習、實體 PCB 製作、還是未來功能擴充）。
以下為你比較常見的 IC 方案與推薦選型：

### 1. 各類 IC 方案比較

| IC 類型 | 代表晶片 / 家族 | 優點 | 缺點 | 適用場景 |
| --- | --- | --- | --- | --- |
| 通用邏輯 IC(74 系列) | 74LS148 (8-to-3 優先編碼器)74HC32 (四 OR 閘) | 1. 價格極便宜2. 免燒錄、接線即用3. 面包板（Breadboard）直插 | 1. 腳位固定，無法修改邏輯2. 板上佈線繁雜3. 難以擴充複雜功能 | 數位邏輯初學者只想用麵包板驗證邏輯門 |
| CPLD | MAX II (EPM240 / EPM570)Altera/Intel | 1. Quartus II 13.0 完全支援2. 上電即啟動（內建 Flash）3. 價格低廉、單電源運作 | 1. 內部邏輯資源（LE）較少2. 逐漸被低端 FPGA 取代 | 輕量級硬體專案純組合邏輯、小型邏輯整合 |
| FPGA | Cyclone IV E (EP4CE6)Altera/Intel | 1. Quartus II 13.0 完全支援2. 資源豐富（可擴充顯示器、MCU）3. 開發板生態系非常完整 | 1. 需要外接 EPCS Flash 晶片2. 核心電壓多軌供電較複雜 | 長遠專案與進階開發後續要加掃描、彈跳消除或鍵盤陣列 |

### 2. 具體 IC 選擇建議

#### 方案 A：使用 Quartus II 13.0 搭配 CPLD (最推薦的硬體實作選擇)
如果你打算將 Quartus 繪製好的 `.bdf` 燒錄到實體晶片中，**CPLD 是最輕量且直覺的選擇**。

- **推薦晶片：****Altera MAX II EPM240T100C5N**
- **推薦原因：**
  1. Quartus II 13.0 Web Edition 內建支援，無需額外授權。
  2. 使用 TQFP-100 封裝，有現成的開發板（含 USB Blaster 燒錄介面），價格僅百元台幣左右。
  3. 非揮發性（Non-volatile），燒錄完斷電後邏輯不會消失，上電直接運行。

#### 方案 B：使用 Quartus II 13.0 搭配 FPGA (適合後續要擴充複雜功能的專案)
如果你未來計畫將這個鍵盤電路升級（例如：4×4 陣列掃描、去彈跳電路 Debounce、七段顯示器解碼、甚至是 PS/2 / USB 介面通訊）：

- **推薦晶片：****Altera Cyclone IV E EP4CE6E22C8N**
- **推薦原因：**
  1. Quartus II 13.0 對 Cyclone IV 的編譯速度極快。
  2. 邏輯資源（6,000 LEs）遠超目前需求，未來要加入時脈（Clock）與計數器（Counter）非常方便。
  3. 市面上有極多帶有獨立按鍵、矩陣鍵盤與七段顯示器的最小系統板（Minimal System Board）。

#### 方案 C：直接用 TTL 74 系列硬體邏輯閘 (純硬體手動接線)
如果你的作業或專案指定要用**實體 Gate 晶片**在麵包板上接線：

- **74HC32**（內含 4 個 2-Input OR Gate）：一顆晶片即可實現上述 $Y_1 = K_3 + K_2$ 與 $Y_0 = K_3 + K_1$ 的邏輯。
- **74LS148 / 74HC148**：標準的 8-to-3 優先編碼器，內部結構就是由 AND/OR/NOT 閘組合而成。

### 3. 總結建議

1. **僅在電腦模擬 / 繳交邏輯設計作業：** 直接在 Quartus II 13.0 中選擇預設元件（例如 `Cyclone IV E: EP4CE6E22C8`）進行編譯與 Waveform 模擬即可，不需要購買實體 IC。
2. **要上板實測（燒錄）：** 建議購買一片 **Altera MAX II (EPM240)** 或 **Cyclone IV (EP4CE6)** 的開發板，配合 **USB Blaster 燒錄器**，即可將在 Quartus 中繪製的 OR 閘電路直接下載到晶片中驗證。

## Turn 11

### 👤 使用者

這個4對2的優先編碼器我是每個都要給高電位與低電位
還是我要給頻率?


## Turn 12

### 👤 使用者

Device family: Cyclone IV GX
Running quartus eda_testbench
>> quartus_eda --gen_testbench --check_outputs=on --tool=modelsim_oem --format=verilog 4x2_Priority_Encoder -c 4x2_Priority_Encoder {--vector_source=C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/Waveform.vwf} {--testbench_file=./simulation/qsim/4x2_Priority_Encoder.vt}
PID = 21976
***********************
Running Quartus II 32-bit EDA Netlist Writer
Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Processing started: Tue Aug 18 23:33:27 2026
Command: quartus_eda --gen_testbench --check_outputs=on --tool=modelsim_oem --format=verilog 4x2_Priority_Encoder -c 4x2_Priority_Encoder --vector_source=C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/Waveform.vwf --testbench_file=./simulation/qsim/4x2_Priority_Encoder.vt
Generated Verilog Test Bench File ./simulation/qsim/4x2_Priority_Encoder.vt for simulation
Quartus II 32-bit EDA Netlist Writer was successful. 0 errors, 0 warnings
Peak virtual memory: 357 megabytes
Processing ended: Tue Aug 18 23:33:28 2026
Elapsed time: 00:00:01
Total CPU time (on all processors): 00:00:00
Running quartus eda_timing_netlist
>> quartus_eda --functional=off --timescale=1ps --simulation --tool=modelsim_oem --format=verilog 4x2_Priority_Encoder -c 4x2_Priority_Encoder
PID = 23452
***********************
Running Quartus II 32-bit EDA Netlist Writer
Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Processing started: Tue Aug 18 23:33:28 2026
Command: quartus_eda --functional=off --timescale=1ps --simulation=on --tool=modelsim_oem --format=verilog 4x2_Priority_Encoder -c 4x2_Priority_Encoder
Generated file 4x2_Priority_Encoder_6_1200mv_85c_slow.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder_6_1200mv_0c_slow.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder_min_1200mv_0c_fast.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder_6_1200mv_85c_v_slow.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder_6_1200mv_0c_v_slow.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder_min_1200mv_0c_v_fast.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Generated file 4x2_Priority_Encoder_v.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_4x2_Priority_Encoder/simulation/modelsim/" for EDA simulation tool
Quartus II 32-bit EDA Netlist Writer was successful. 0 errors, 0 warnings
Peak virtual memory: 357 megabytes
Processing ended: Tue Aug 18 23:33:28 2026
Elapsed time: 00:00:00
Total CPU time (on all processors): 00:00:01
***********************
Running quartus modelsim
>> vsim -c -do 4x2_Priority_Encoder.do
PID = 30084
Reading C:/altera/13.0sp1/modelsim_ase/tcl/vsim/pref.tcl
# 10.1d
# do 4x2_Priority_Encoder.do
# ** Warning: (vlib-34) Library already exists at "work".
#
# Model Technology ModelSim ALTERA vlog 10.1d Compiler 2012.11 Nov 2 2012
# -- Compiling module \4x2_Priority_Encoder
#
# Top level modules:
# \4x2_Priority_Encoder
# Model Technology ModelSim ALTERA vlog 10.1d Compiler 2012.11 Nov 2 2012
# ** Error: 4x2_Priority_Encoder.vt(29): near "4": syntax error, unexpected INTEGER NUMBER, expecting IDENTIFIER or automatic or TYPE_IDENTIFIER or static
# ** Error: c:/altera/13.0sp1/modelsim_ase/win32aloem/vlog failed.
# Executing ONERROR command at macro ./4x2_Priority_Encoder.do line 4
顯示 Error occured during modelsim simulation


## Turn 13

### 👤 使用者

我在K3跟K2給了高電位1，輸出為Y1與Y0是高電位1，這樣是對的嗎?


## Turn 14

### 👤 使用者

下一步要畫什麼電路?


## Turn 15

### 👤 使用者

只有二對七嗎?不是四對七嗎?

### 🤖 助理

你的質疑非常敏銳，但**這兩種說法都完全正確**，差別只在於「輸入訊號的維度」**以及**「有沒有包含《無按鍵》這個狀態」：

### 1. 為什麼前面說「二對七（2-to-7 Decoder）」？
因為在我們上一階段完成的 **4-to-2 優先編碼器**中：

- 4 個原始按鍵 ($K_3 \sim K_0$) 經過編碼後，輸出的**資料線只有 2 條**（$Y_1, Y_0$）。
- 2-bit 的二進位可以組合出 $2^2 = 4$ 種狀態（`00`, `01`, `10`, `11`），剛好對應數字 `0`, `1`, `2`, `3`。
- 將這 **2 條資料線**解碼驅動七段顯示器的 7 個 LED 段（$a \sim g$），這部分的解碼邏輯就是 **2-to-7**。

### 2. 那什麼時候會是「四對七（4-to-7 Decoder）」？
在常見的數位邏輯教科書或標準 IC（例如 **74LS47 / 74HC4511**）中，最普遍的確實是 **4-to-7（或稱 BCD-to-7 Segment Decoder）**！原因如下：

1. **直接輸入 BCD 碼（4-bit）：**
  - 為了要顯示 `0` ~ `9` 十個數字，至少需要 4-bit 的二進位碼（$2^4 = 16$ 種狀態，取其中的 `0000` ~ `1001`）。此時輸入有 4 條線（$D, C, B, A$），輸出 7 條線，這就是標準的 **4-to-7 解碼器**。
2. **處理「無按鍵」與「無效狀態」：**
  - 如果不經過 4-to-2 編碼器，而是把 4 個按鍵直接輸入給解碼器，輸入就是 4-bit。
  - 更重要的是：**當「沒有按下任何鍵」時，七段顯示器應該要全滅（全 0）**。
  - 2-bit 的 $Y_1, Y_0$ 在未按鍵時輸出是 `00`，如果不搭配 $Valid$ 訊號，解碼器會無法區分「按了 Key 0」還是「完全沒按鍵」，都會顯示數字 `0`。

### 3. 兩種架構比較

#### 架構 A：4-to-2 編碼器 + 2-to-7 解碼器（我們剛才討論的）

```
[4個按鍵] ---> [ 4-to-2 Encoder ] --(Y1, Y0)--> [ 2-to-7 Decoder ] ---> [ 七段顯示器 ]
                                \--(Valid)------> [ 致能 Blanking ]

```

- **優點：** 中間傳輸的資料線很少（只有 2 條）。
- **處理無按鍵：** 必須用 $Valid$ 訊號控制七段顯示器（當 $Valid = 0$ 時將七段顯示器強制關閉/Blanking）。

#### 架構 B：完整 4-to-7 鍵盤解碼器（一次到位）

```
[4個按鍵 (K3, K2, K1, K0)] ---> [ 4-to-7 Decoder ] ---> [ 七段顯示器 (a~g) ]

```

- **輸入：** 直接輸入 $K_3, K_2, K_1, K_0$ 4 個按鍵訊號。
- **真值表涵蓋 5 種狀態：**
  - `0000`（未按鍵）$\rightarrow$ 七段顯示器全滅（$a \sim g$ 全為 0）
  - `0001`（按 Key 0）$\rightarrow$ 顯示 `0`
  - `0010`（按 Key 1）$\rightarrow$ 顯示 `1`
  - `0100`（按 Key 2）$\rightarrow$ 顯示 `2`
  - `1000`（按 Key 3）$\rightarrow$ 顯示 `3`

### 總結與建議
如果你希望「沒有按任何鍵時，七段顯示器是暗的」，直接畫一個 **4-to-7 解碼器**（輸入為 4 個按鍵 $K_3 \sim K_0$，輸出 $a \sim g$）邏輯會最完整！
要我為你提供包含「未按鍵時全滅」的 **4-to-7 真值表與邏輯公式**嗎？

## Turn 16

### 👤 使用者

我現在在做的是一般108鍵的那種鍵盤喔
不是單存的數字鍵盤

### 🤖 助理

原來你要做的是**標準 108 鍵（全尺寸 104/108 Key）電腦鍵盤**！這個規模與單純的 4 鍵/16 鍵數字小鍵盤完全不同。
如果用傳統的單獨開關對應（1 鍵對 1 條線），108 個按鍵就需要 108 支 GPIO 腳位，且邏輯閘電路會龐大到無法實現。因此，工業界與標準 108 鍵盤（如 USB/PS/2 鍵盤）的核心設計架構是：**16×8 矩陣陣列（Matrix） + 順序掃描（Scanning） + 掃描碼解碼（Scan Code Decoder / ROM）**。
下面為你拆解 108 鍵盤在 Quartus II 中的硬體架構與實現邏輯：

### 1. 108 鍵盤的核心硬體架構
108 個按鍵只需要 **16 條行線（Columns, C0​∼C15​）** 與 **8 條列線（Rows, R0​∼R7​）** 的交叉點即可覆蓋（$16 \times 8 = 128$ 個交叉點，足夠放入 108 鍵）：

```
[ 時脈訊號 (Clock) ]
        │
        ▼
[ 1/16 輪流掃描計數器 ] ──(行選通 C0~C15)──> ┌──────────────────────────┐
                                             │  16x8 按鍵矩陣 (108鍵)    │
[ 8-to-3 編碼器 / 讀取 ] <──(列讀取 R0~R7)─── └──────────────────────────┘
        │
        ▼
[ 7-bit 鍵盤座標碼 (Row, Col) ]
        │
        ▼
[ 掃描碼對照表 Look-up Table (ROM/Decoder) ]
        │
        ▼
[ 輸出 ASCII 或 PS/2 Scan Code ]

```

### 2. 108 鍵盤需要的四大模組（電路模組化）
要在 Quartus II 中完成 108 鍵盤設計，不能只用純組合邏輯，必須搭配時脈（Clock）進行**分工設計**：

| 模組名稱 | 使用元件 / 邏輯 | 功能說明 |
| --- | --- | --- |
| 1. 矩陣掃描計數器 (Row/Col Scanner) | Counter + 4-to-16 Decoder | 產生約 1 kHz 的掃描頻率，高速輪流將 C0​∼C15​ 拉低/拉高（每次只掃描一列）。 |
| 2. 列讀取與按鍵鎖存 (Read & Latch) | 8-to-3 Priority Encoder + D-FF | 當掃描到某一行時，讀取 R0​∼R7​ 哪一線有導通，鎖存當前座標 (Column,Row)。 |
| 3. 按鍵去彈跳 (Debounce Filter) | Shift Register / Counter | 實體 108 鍵機械/薄膜開關有 5~10ms 的物理彈跳，過濾掉開關雜訊。 |
| 4. 鍵碼對照表 (Keycode LUT / ROM) | 7-to-8 Decoder / Memory | 將 7-bit 矩陣座標碼（27=128 個對應位址）轉換為標準 PS/2 Scan Code 或 ASCII 碼（例如：座標 (2, 3) 輸出 'A' 的 Scan Code 1C）。 |

### 3. 如何在 Quartus II 13.0 中實現？
對於 108 鍵盤這種規模，如果你只用純邏輯閘（AND, OR, NOT）在 Schematic (`.bdf`) 裡面拉線，圖紙會變得極度複雜且難以維護。
**建議的實作方式：**

1. **模組化圖形設計（Block Design）：** 把「掃描器」、「矩陣解碼器」、「ROM 對照表」分別畫成獨立的子模組（Symbol），最後在主圖紙拉線組合。
2. **ROM（Look-Up Table）記憶體模組：** Quartus II 13.0 裡面有現成的 **MegaFunction（LPM_ROM）**。你可以寫一個 `.mif` 檔案，輸入矩陣座標（0~107），直接輸出對應 108 鍵的鍵碼（Scan Code），這樣就不需要手動拉幾百個 OR 閘去建構 108 鍵的譯碼器。

### 4. 下一步你可以怎麼開始？
要實作 108 鍵盤，建議**分階段完成**：

1. **先完成 4x4 (16鍵) 陣列掃描與解碼**：先理解矩陣掃描與時脈（Clock）的運作方式。
2. **擴充為 16x8 (128點) 矩陣掃描**：將掃描計數器由 4-bit 擴充為 7-bit（4-bit Col + 3-bit Row）。
3. **建立 108 鍵 Scan Code 對照表**：利用 ROM 進行鍵碼對映。

## Turn 17

### 👤 使用者

先從4x4陣列掃描開始


## Turn 18

### 👤 使用者

C0-C3跟T0-T3嗎?


## Turn 19

### 👤 使用者

等等，這是四變數還是八變數?


## Turn 20

### 👤 使用者

我覺得我們先聰4x4的掃描電路中從除頻器開始，使用74xxIC進行設計，我忘記是不是7490了


## Turn 21

### 👤 使用者

Device family: Cyclone IV GX
Running quartus eda_testbench
> quartus_eda -- gen_testbench -- check_outputs=on -- tool=modelsim_oem -- format=verilog FPGA_Mod_50MHz -c FPGA_Mod_50MHz { -- vector_source=C: /Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod_50MHz/Wa
veform.vwf} { -- testbench_file =. /simulation/qsim/FPGA_Mod_50MHz.vt}
PID = 3676
Running Quartus II 64-Bit EDA Netlist Writer
Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Processing started: Fri Aug 21 05:04:33 2026
Command: quartus_eda -- gen_testbench -- check_outputs=on -- tool=modelsim_oem -- format=verilog FPGA_Mod_50MHz -c FPGA_Mod_50MHz -- vector_source=C: /Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod_50M
Hz/Waveform.vwf -- testbench_file =. /simulation/qsim/FPGA_Mod_50MHz. vt
Generated Verilog Test Bench File ./simulation/qsim/FPGA Mod_50MHz.vt for simulation
Quartus II 64-Bit EDA Netlist Writer was successful. 0 errors, 0 warnings
Peak virtual memory: 4570 megabytes
Processing ended: Fri Aug 21 05:04:33 2026
Elapsed time: 00:00:00
Total CPU time (on all processors) : 00:00:00
Running quartus eda_func netlist
>> quartus_eda -- functional=on -- simulation -- tool=modelsim_oem -- format=verilog FPGA_Mod_50MHz -c FPGA_Mod_50MHz
PID = 12024
****
Running Quartus II 64-Bit EDA Netlist Writer
Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Processing started: Fri Aug 21 05:04:33 2026
Command: quartus_eda -- functional=on -- simulation=on -- tool=modelsim_oem -- format=verilog FPGA Mod 50MHz -c FPGA Mod 50MHz
Generated file FPGA Mod 50MHz.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod_50MHz/simulation/modelsim/" for EDA simulation tool
Quartus II 64-Bit EDA Netlist Writer was successful. 0 errors, 0 warnings
Peak virtual memory: 4570 megabytes
Processing ended: Fri Aug 21 05:04:33 2026
Elapsed time: 00:00:00
Total CPU time (on all processors) : 00:00:00
Running quartus modelsim
>> vsim -c -do FPGA Mod_50MHz.do
PID = 18652
Reading C:/altera/13.0sp1/modelsim_ase/tcl/vsim/pref.tcl
# 10.1d
# do FPGA Mod 50MHz.do
# ** Warning: (vlib-34) Library already exists at "work".
#
#
Model Technology ModelSim ALTERA vlog 10.1d Compiler 2012.11 Nov 2 2012
-- Compiling module FPGA_Mod_50MHz
# Top level modules:
FPGA_Mod_50MHz
# Model Technology ModelSim ALTERA vlog 10.1d Compiler 2012.11 Nov 2 2012
# -- Compiling module FPGA Mod_50MHz_vlg_sample_tst
# -- Compiling module FPGA Mod_50MHz_vlg_check_tst
# ** Error: FPGA_ Mod_50MHz.vt (58) : near "50": syntax error, unexpected INTEGER NUMBER, expecting ')'
# ** Error: c:/altera/13.0sp1/modelsim_ase/win32aloem/vlog failed.
# Executing ONERROR command at macro ./FPGA Mod_50MHz.do line 4
為什麼會出現錯誤?我改選擇 Quartus II Simulator 卻沒有任何輸出結果


## Turn 22

### 👤 使用者

是不能出現大小寫是嗎?


## Turn 23

### 👤 使用者

>> quartus_eda -- gen_testbench -- check_outputs=on -- tool=modelsim_oem -- format=verilog FPGA_Mod50MHz -c FPGA_Mod50MHz { -- vector_source=C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/Wav
orm.vwf} { -- testbench_file =. /simulation/qsim/FPGA_Mod50MHz.vt}
PID = 32380
Running Quartus II 64-Bit EDA Netlist Writer
Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Processing started: Fri Aug 21 05:23:20 2026
Command: quartus_eda -- gen_testbench -- check_outputs=on -- tool=modelsim_oem -- format=verilog FPGA_Mod50MHz -c FPGA_Mod50MHz -- vector_source=C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MH
Waveform.vwf -- testbench_file =. /simulation/qsim/FPGA Mod50MHz.vt
Generated Verilog Test Bench File ./simulation/qsim/FPGA_Mod50MHz.vt for simulation
Quartus II 64-Bit EDA Netlist Writer was successful. 0 errors, 0 warnings
Peak virtual memory: 4570 megabytes
Processing ended: Fri Aug 21 05:23:20 2026
Elapsed time: 00:00:00
Total CPU time (on all processors) : 00:00:00
Running quartus eda timing netlist
>> quartus_eda -- functional=off -- timescale=1ps -- simulation -- tool=modelsim_oem -- format=verilog FPGA_Mod50MHz -c FPGA_Mod50MHz
PID = 2804
Running Quartus II 64-Bit EDA Netlist Writer
Version 13.0.1 Build 232 06/12/2013 Service Pack 1 SJ Web Edition
Processing started: Fri Aug 21 05:23:20 2026
Command: quartus_eda -- functional=off -- timescale=1ps -- simulation=on -- tool=modelsim_oem -- format=verilog FPGA_Mod50MHz -c FPGA_Mod50MHz
Generated file FPGA Mod50MHz_6_1200mv_85c_slow.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA_Mod50MHz_6_1200mv_Oc_slow.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA_Mod50MHz_min_1200mv_Oc_fast.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA_Mod50MHz.vo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA Mod50MHz_6_1200mv_85c_v_slow.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA Mod50MHz_6_1200mv_Oc_v_slow.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA_Mod50MHz_min_1200mv_Oc_v_fast.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Generated file FPGA Mod50MHz_v.sdo in folder "C:/Users/Roy/Documents/GitHub/quartus/Quartus13.0/FPGA/FPGA_Mod50MHz/simulation/modelsim/" for EDA simulation tool
Quartus II 64-Bit EDA Netlist Writer was successful. 0 errors, 0 warnings
Peak virtual memory: 4586 megabytes
Processing ended: Fri Aug 21 05:23:21 2026
Elapsed time: 00:00:01
Total CPU time (on all processors) : 00:00:01
******
Running quartus modelsim
> vsim -c -do FPGA_Mod50MHz.do
PID = 19080
Reading C:/altera/13.0sp1/modelsim_ase/tcl/vsim/pref.tcl
# 10.1d
# do FPGA Mod50MHz.do
# ** Warning: (vlib-34) Library already exists at "work".
Model Technology ModelSim ALTERA vlog 10.1d Compiler 2012.11 Nov 2 2012
-- Compiling module FPGA_Mod50MHz
Top level modules:
FPGA Mod50MHz
# Model Technology ModelSim ALTERA vlog 10.1d Compiler 2012.11 Nov 2 2012
-- Compiling module FPGA_Mod50MHz_vlg_sample_tst
-- Compiling module FPGA_Mod50MHz_vlg_check_tst
** Error: FPGA_Mod50MHz. vt (58) : near "50": syntax error, unexpected INTEGER NUMBER, expecting ')'
** Error: c:/altera/13.0sp1/modelsim_ase/win32aloem/vlog failed.
Executing ONERROR command at macro ./FPGA_Mod50MHz.do line 4


## Turn 24

### 👤 使用者

你剛剛還說MHz這樣的大小寫是通過的，根本就不行啊，要就是用FPGA_Mod50


## Turn 25

### 👤 使用者

那我應該買什麼樣的FPGA機台?


## Turn 26

### 👤 使用者

我現在對於學習FPGA的部分有點模糊
我只知道這個的成本很低
但是一台機台真的很不便宜


## Turn 27

### 👤 使用者

幫我找出符合條件的蝦皮賣場產品


## Turn 28

### 👤 使用者

幫我尋找有關除頻電路的quartus II教學影片，附上教學影片


## Turn 29

### 👤 使用者

epm3064 datasheet


## Turn 30

### 👤 使用者

請提供 EPM3064ATC44 (44-pin TQFP) 的詳細引腳定義（包含 VCC、GND、JTAG 與 I/O 腳位分布）。


## Turn 31

### 👤 使用者

但是在ic母座上的io port就跟ic不一樣

### 🤖 助理

你觀念非常清楚！TQFP 封裝（表面貼裝）的腳位排序，跟插在 IC 母座/麵包板上的 **DIP 或 PLCC 轉接座**，順序是完全不一樣的。
如果要在實驗板或 IC 母座（洞洞板/麵包板）上使用 EPM3064，最常見的是 **PLCC-44 封裝（EPM3064ALC44）**。

### EPM3064ALC44 (PLCC-44 母座視角) 完整腳位圖
以下為 **PLCC-44 母座從正上方看下去（Top View）** 的完整 Pinout 對照表：

```
                    ┌─────────────────────────┐
                    │ 06 05 04 03 02 01 44    │
                07 ─┤                         ├─ 43
           TMS  08  │                         │  42  GCLR
                09  │                         │  41  VCCINT
                10  │                         │  40  OE1
          VCCIO 11  │     EPM3064ALC44        │  39  GCLK2
                12  │       (PLCC-44)         │  38  GCLK1
                13  │                         │  37  GND
                14  │                         │  36  GND
                    │ 15 16 17 18 19 20 21    │
                    └─────────────────────────┘

```

### PLCC-44 實體母座 44 支腳位詳細功能表

| PLCC Pin # | 腳位名稱 / 功能 | 腳位類型 | 鍵盤掃描 / 除頻器接線建議 |
| --- | --- | --- | --- |
| 1 | TDI | JTAG | 接 USB-Blaster TDI |
| 2 | I/O | GPIO | 可做 4x4 鍵盤 Row 0 |
| 3 | I/O | GPIO | 可做 4x4 鍵盤 Row 1 |
| 4 | I/O | GPIO | 可做 4x4 鍵盤 Row 2 |
| 5 | I/O | GPIO | 可做 4x4 鍵盤 Row 3 |
| 6 | I/O | GPIO | 可做 4x4 鍵盤 Col 0 |
| 7 | TMS | JTAG | 接 USB-Blaster TMS (10k 上拉) |
| 8 | I/O | GPIO | 可做 4x4 鍵盤 Col 1 |
| 9 | I/O | GPIO | 可做 4x4 鍵盤 Col 2 |
| 10 | I/O | GPIO | 可做 4x4 鍵盤 Col 3 |
| 11 | VCCIO | 電源 | 接 3.3V |
| 12 | I/O | GPIO | 通用 I/O |
| 13 | I/O | GPIO | 通用 I/O |
| 14 | I/O | GPIO | 通用 I/O |
| 15 | VCCINT | 電源 | 接 3.3V |
| 16 | I/O | GPIO | 通用 I/O |
| 17 | GND | 電源 | 系統接地 (GND) |
| 18 | I/O | GPIO | 通用 I/O |
| 19 | I/O | GPIO | 通用 I/O |
| 20 | I/O | GPIO | 通用 I/O |
| 21 | I/O | GPIO | 通用 I/O |
| 22 | GND | 電源 | 系統接地 (GND) |
| 23 | VCCIO | 電源 | 接 3.3V |
| 24 | I/O | GPIO | 通用 I/O |
| 25 | I/O | GPIO | 通用 I/O |
| 26 | TCK | JTAG | 接 USB-Blaster TCK (10k 下拉) |
| 27 | I/O | GPIO | 通用 I/O |
| 28 | I/O | GPIO | 通用 I/O |
| 29 | I/O | GPIO | 通用 I/O |
| 30 | I/O | GPIO | 通用 I/O |
| 31 | I/O | GPIO | 通用 I/O |
| 32 | TDO | JTAG | 接 USB-Blaster TDO |
| 33 | I/O | GPIO | 通用 I/O |
| 34 | I/O | GPIO | 通用 I/O |
| 35 | VCCINT | 電源 | 接 3.3V |
| 36 | GND | 電源 | 系統接地 (GND) |
| 37 | GND | 電源 | 系統接地 (GND) |
| 38 | GCLK1 | 專用 | 50MHz 主時脈輸入點 |
| 39 | GCLK2 / OE2 | 專用 | 全域時脈 2 |
| 40 | OE1 | 專用 | 全域輸出致能 |
| 41 | VCCINT | 電源 | 接 3.3V |
| 42 | GCLR | 專用 | 系統 Reset 腳 (Active Low) |
| 43 | I/O | GPIO | 通用 I/O |
| 44 | I/O | GPIO | 通用 I/O |

### PLCC-44 母座上腳位的辨識技巧

1. **尋找 Pin 1：** 母座頂面通常會有一個斜角凹槽（Chamfer），或者在特定邊的中間有一個圓點標記，該點對應的就是 **Pin 1**。
2. **順序方向：** 找到 Pin 1 後，依照 **逆時針方向** 依序數為 Pin 2, Pin 3 ... 到 Pin 44。
3. **Quartus 軟體設定：** 在 Quartus II 的 Pin Planner 中，指派 Pin 時請確認選取的是 **PLCC44** 封裝，軟體就會自動對照上述的實體腳位編號。

## Turn 32

### 👤 使用者

如果是我要自己動手建置一台開發版機台要怎麼做?

### 🤖 助理

自己動手自製（DIY）一台 CPLD/FPGA 開發板是學習硬體設計與 PCB Layout 最棒的專案！
要成功製作出一台能穩壓、燒錄、順利運作 4×4 鍵盤掃描的開發板，硬體架構主要由 **5 大電路模組** 組成：

```
[外部 5V / USB 電源] ──► [電源管理 3.3V] ──► 給 CPLD 核心 (VCCINT) 與 I/O (VCCIO)
                                             │
[10-pin JTAG 排針] ◄── (JTAG 腳位 1, 7, 26, 32) ─┤
                                             │
[50MHz 石英晶體 Oscillator] ──► (Pin 38 GCLK1) ┼─► [EPM3064ALC44 CPLD]
                                             │
[按鍵 + 電容 Filter] ────────► (Pin 42 GCLR) ─┤
                                             │
                                             └──► [2.54mm 排針 GPIO 介面]
                                                      │
                                                      ├── 4×4 矩陣鍵盤
                                                      └── 七段顯示器 / LED

```

### 1. 五大核心子電路設計

#### ① 電源管理模組 (Power Supply)

- **需求：** EPM3064A 需要 **3.3V** 核心與 I/O 電壓（$V_{CCINT}$ 與 $V_{CCIO}$）。
- **實作：** 使用 **AMS1117-3.3** 降壓 IC，將外部 DC 5V（或 Type-C 5V）降至 3.3V。
- **關鍵元件：**
  - 輸入端放 $10\mu\text{F}$ 濾波電容。
  - 輸出端放 $10\mu\text{F}$ 電容。
  - **在 CPLD 的每個 VCC 腳位（Pin 11, 15, 23, 35, 41）旁，務必貼近放置一顆 0.1μF 去耦貼片電容**（防止數位雜訊導致晶片重啟）。

#### ② JTAG 燒錄介面 (Programmer Interface)

- **需求：** 連接 USB-Blaster 燒錄器。
- **接線：** 使用 2.54mm 2x5 (10-pin) 簡易牛角座（Box Header）：
  - **Pin 1 (TCK):** 接 CPLD **Pin 26**（外接 10kΩ 下拉電阻至 GND）
  - **Pin 3 (TDO):** 接 CPLD **Pin 32**
  - **Pin 5 (TMS):** 接 CPLD **Pin 7**（外接 10kΩ 上拉電阻至 3.3V）
  - **Pin 9 (TDI):** 接 CPLD **Pin 1**
  - **Pin 2, 10:** 接 3.3V ； **Pin 4, 10:** 接 GND

#### ③ 全域時脈來源 (Clock Source)

- **需求：** 提供 50 MHz 系統時脈給除頻器。
- **實作：** 選用 **50MHz 有源石英晶體振盪器（Active Oscillator，4 腳位）**。
  - VCC 接 3.3V，GND 接地，**OUT 直接接到 CPLD 的 Pin 38 (GCLK1)**。

#### ④ 全域復位電路 (Global Reset)

- **需求：** 開機或手動按壓時將系統歸零。
- **接線：** 一個輕觸按鍵一端接地，另一端接 **Pin 42 (GCLR)**，並加上 10kΩ 上拉電阻到 3.3V（形成 Active-Low 低電位觸發復位）。

#### ⑤ GPIO 擴充介面 (I/O Expansion)

- **接線：** 將其餘 30 支通用 I/O 引腳（如 Pin 2~6, 8~10, 12~14...）全部引出到 2.54mm 單排針（Pin Header），方便用杜邦線外接 4×4 矩陣鍵盤模組。

### 2. 動手建置的四個實作步驟
11. 繪製原理圖 (Schematic) 與 Layout使用 EasyEDA (嘉立創) 或 KiCad 繪製原理圖1.1. 繪製原理圖 (Schematic) 與 Layout:使用 EasyEDA (嘉立創) 或 KiCad 繪製原理圖.建立原理圖並把上述 5 個模組畫出。進行 PCB Layout：把 PLCC-44 母座放在中央，電源線與地線盡量拉粗（0.5mm 以上）。22. 製作 PCB選擇洞洞板手工焊接，或直接送廠打樣2.2. 製作 PCB:選擇洞洞板手工焊接，或直接送廠打樣.方案 A（極低成本/打樣）： 將 PCB Gerber 檔上傳至嘉立創/JLCPCB 等廠商，進行免費或幾十元台幣的 PCB 打樣，3~5 天收到成品。方案 B（純手工）： 買一張萬用洞洞板（Perfboard）與 PLCC-44 轉接板，用鍍錫銅線與 OK 線手工焊接。3按電源 -3.:按電源 -.晶振 -> CPLD 順序進行" title="3. 焊接與三用電表測量">先焊電源部分： 通電確認 AMS1117 輸出精準的 3.3V（切勿在未量測前插上 CPLD，避免過壓燒毀）。焊上 PLCC-44 母座、JTAG 與晶振。量測短路： 拿三用電表蜂鳴檔測量 3.3V 與 GND 是否短路。44. 燒錄測試與 GPIO 驗證使用 Quartus II 與 USB-Blaster 上傳程式4.4. 燒錄測試與 GPIO 驗證:使用 Quartus II 與 USB-Blaster 上傳程式.將 EPM3064ALC44 晶片插入母座。插上 USB-Blaster，開啟 Quartus II 13.0 的 Programmer，點擊 Auto Detect。若成功抓到 EPM3064A，代表硬體電路完全成功！可以開始燒錄 4×4 掃描程式。
### 💡 開發板零件採購清單 (BOM 表)

| 零件名稱 | 規格與數量 | 估計價格 (NTD) |
| --- | --- | --- |
| CPLD 核心晶片 | EPM3064ALC44-10N | 約 $100 - $150 |
| IC 座 | PLCC-44 腳座（直插洞洞板用） | 約 $15 - $25 |
| 電源 IC | AMS1117-3.3V SOT-223 | 約 $5 |
| 時脈 | 50MHz 有源晶振 (OSC 3.3V) | 約 $20 - $35 |
| JTAG 插座 | 2.54mm 2×5P 簡易牛角座 | 約 $5 - $10 |
| 被動元件 | 0.1μF 貼片/陶瓷電容 x5、10kΩ 電阻 x3 | 約 $10 |
| 介面與排針 | Type-C 母座板 / DC 插座、2.54mm 單排針 | 約 $20 |

## Turn 33

### 👤 使用者

這樣感覺好像給自己做了一個太難的目標了
應該先從基本的開始


## Turn 34

### 👤 使用者

所以我要做1/16 輪流掃描計數器是要怎麼做?


## Turn 35

### 👤 使用者

所以是除16的計數器嗎?


## Turn 36

### 👤 使用者

我現在對於用74191的用法做成1/16輪流掃描的部分不夠理解，我要使用Block的做法進行

### 🤖 助理

**74191** 是一顆 **4-bit 可逆（Up/Down）二進制同步計數器**，它的核心功能與 74161 非常相似，同樣能輸出 0 到 15（`0000` ~ `1111`）共 16 種狀態。
要在 Quartus II 中使用 Block（.bdf 原理圖）將 74191 作為 **1/16 輪流掃描核心**，整體架構只需要 3 個 Block 組合：

```
[除頻器 Output] ──► [74191 計數器 (0~15)] ──► [74138 解碼器] ──► 掃描線 (Row 0 ~ Row 3)

```

### 一、 74191 引腳功能與控制設定
在 Quartus 原理圖庫裡，74191 的主要引腳與必備設定如下：

| 腳位名稱 | 功能說明 | 1/16 掃描時的控制設定 |
| --- | --- | --- |
| CLK | 時脈輸入 | 接除頻後的掃描時脈（例如 1kHz 訊號） |
| D/U | 上/下計數選擇 (Down/Up) | 接地 (GND) → 代表向上計數（0 → 1 → ... → 15） |
| LOAD | 異步加載 (Active Low) | 接 VCC (高電位) → 關閉預置載入，讓它自由計數 |
| CTEN | 計數致能 (Active Low) | 接地 (GND) → 致能計數器 |
| A, B, C, D | 預置數據輸入 | 不需連接（或拉至 GND） |
| QA, QB, QC, QD | 4-bit 計數輸出 | 輸出端（QA=20, QB=21, QC=22, QD=23） |

### 二、 Block 原理圖拉線步驟 (Step-by-Step)

#### 1. 配置 74191 讓它跑 0~15 循環

1. 從 Symbol Tool 拿出 **74191**。
2. 將 **D/U** 與 **CTEN** 引腳拉一條線接到 **GND**（控制向上計數且持續運作）。
3. 將 **LOAD** 引腳拉一條線接到 **VCC**。
4. 將除頻後的脈衝訊號接到 **CLK**。

#### 2. 將計數輸出接至 74138 解碼器（產生輪流低電位掃描）
為了將 16 種計數狀態轉換成 4 條掃描線（Row 0 ~ Row 3），我們搭配一顆 **74138 (3-to-8 解碼器)**：

1. 拿出 **74138** 解碼器。
2. 將 74138 的致能腳 **G1** 接 **VCC**，**G2A** 與 **G2B** 接 **GND**（使能解碼器）。
3. 用**單條細線**將 74191 的高位元腳位接到 74138 的選擇輸入端：
  - 74191 的 **QC** 接至 74138 的 **A**（最低位選擇）
  - 74191 的 **QD** 接至 74138 的 **B**（高位選擇）
  - 74138 的 **C** 接 **GND**
4. 此時，74138 的輸出端 **Y0, Y1, Y2, Y3** 就會隨著計數器推進，輪流輸出**低電位 (0)**，這四支腳就是你的 **4×4 鍵盤 Row 0 ~ Row 3 掃描線**！

### 三、 動作時序與狀態變換

| 計數脈衝 (CLK) | 74191 輸出 (QD,QC,QB,QA) | 十進位數值 | 74138 輸出 (Y3,Y2,Y1,Y0) | 當前動作 |
| --- | --- | --- | --- | --- |
| 第 1 個脈衝 | 0000 | 0 | 1110 | 掃描 Row 0 (掃描第 0~3 號按鍵) |
| 第 5 個脈衝 | 0100 | 4 | 1101 | 掃描 Row 1 (掃描第 4~7 號按鍵) |
| 第 9 個脈衝 | 1000 | 8 | 1011 | 掃描 Row 2 (掃描第 8~11 號按鍵) |
| 第 13 個脈衝 | 1100 | 12 | 0111 | 掃描 Row 3 (掃描第 12~15 號按鍵) |
| 第 17 個脈衝 | 0000 | 0 | 1110 | 溢位回到 0，重新開始新一輪掃描 |

### 💡 關鍵避坑提醒

- **腳位別用 Bus 混接：** 74191 的 $QA \sim QD$ 是單獨的腳位，請用細線（Signal Line）一對一拉到 74138 的輸入腳，避免再出現 Quartus 的 `Width mismatch` 報錯。
- **為什麼 74191 比 74161 好理解？** 因為 74191 只要把 `D/U` 設為 0、`LOAD` 設為 1，它就是一顆純粹的 0~15 自動循環計數器，不需額外外接清零腳。

## Turn 37

### 👤 使用者

現在我已經處理出mod16
但是scanner怎麼做就不知道了


## Turn 38

### 👤 使用者

顯示的是FE、FD、FB等，那下一步要做什麼?

### 🤖 助理

看到 **FE (1111 1110) → FD (1111 1101) → FB (1111 1011) → F7 (1111 0111)**，代表你的 **Scanner 掃描核心已經 100% 成功**！
這意味著你的 FPGA / CPLD 正在以極快速度輪流向 4x4 鍵盤的 4 條橫列（Row 0 ~ Row 3）發送探針訊號。

### 下一步：建置「按鍵讀取與鎖存 (Key Read & Latch)」
現在 Scanner 已經在發送訊號了，下一步就是要**讀取按鍵按下時傳回來的直行訊號（Col 0 ~ Col 3）**，並把按下的按鍵號碼（0~15）給鎖存（Latch）出來。
整個邏輯架構如下：

```
[ Row 掃描輸出 ] ──► 4x4 矩陣鍵盤 ──► [ Col 讀取輸入 ]
                                              │
[ Mod16 當前計數值 (0~15) ] ─────────────────┼──► [ 按鍵鎖存器 (Latch / Register) ]
                                              │          │
[ 偵測到 Col 有低電位 (按鍵按下) ] ──────────┘          ▼
                                                [ 輸出：最終按鍵碼 Key_Code (0~15) ]

```

### 原理圖 (BDF) 具體實作步驟

#### 1. 加入 4x4 鍵盤的 Col 輸入腳位

- 在 BDF 畫面上新增一個 4-bit 的 Input 腳位（或 4 支單獨立 Input 腳位），命名為 **Col[3..0]**。
- **重點觀念：** 4x4 鍵盤平時沒有按壓時，Col 腳位因為外接上拉電阻（Pull-up），預設都是高電位 **1111**。當某個按鍵被按下且正好掃描到該列時，對應的 Col 腳位會被拉低成 **0**。

#### 2. 製作「按鍵按下偵測邏輯」（Key Press Detector）
我們要判斷「現在有沒有任何一個按鍵被按下」：

- 使用一個 **4 輸入 NAND 門（NAND4）** 或將 `Col[0]`, `Col[1]`, `Col[2]`, `Col[3]` 通過邏輯門。
- **邏輯：** 當 `Col[3..0]`**不等於**`1111`（即有任何一支 Pin 出現 `0`）時，產生的邏輯訊號 `Key_Pressed` 就會變成高電位 **1**。

#### 3. 使用 74373 / D 正反器鎖存按鍵碼（Latch）
當按鍵被按下的瞬間，我們要把 Mod16 計數器當下的數值（0~15）抓起來存著，否則掃描器一跳過，數字就跑掉了。

- 放出一顆 **74373**（8-bit 鎖存器）或 **DFF（D 正反器）**：
  - **Data 輸入端 (D)：** 接你 Mod16 計數器的輸出 $Q_3, Q_2, Q_1, Q_0$（代表 0~15）。
  - **Enable / Clock 控制端 (G 或 CLK)：** 接步驟 2 產生的 `Key_Pressed` 訊號。
  - **Data 輸出端 (Q)：** 這 4 支腳位輸出的就是**最後按下的按鍵號碼（Key Code 0~15）**！

### 最終驗證目標
完成這個步驟後，你在 Waveform 模擬時：

1. 強制給予 `Col[3..0]` 一個低電位（例如在 Row 1 掃描時，將 `Col[1]` 拉低，代表按下第 5 號鍵）。
2. 你會看到輸出腳位瞬間被鎖定並持續顯示 **0101 (5)**！即使 Scanner 繼續往後掃描，輸出依然保持在 5，直到下一次按鍵被按下。

## Turn 39

### 👤 使用者

你這樣的形容有夠難以理解
可能給予架構圖或多或少能夠加以理解

### 🤖 助理

抱歉！剛剛用純文字講電路鎖存確實太抽象了。
我們把整個系統拆成 3 個最核心的方塊，這張**原理圖（BDF）架構圖**能幫你一眼看懂訊號是怎麼流動的：

### 4×4 鍵盤掃描與鎖存 完整架構圖

```
                 【系統時脈】 (例如 50MHz 或 1kHz)
                      │
                      ▼
             ┌─────────────────┐
             │  Mod16 計數器   │
             │ (74191 / 74161) │
             └───────┬─────────┘
                     │
         ┌───────────┴──────────────────────────────┐
         │ 計數值 (0~15 的 4-bit 訊號)              │
         ▼                                          ▼
┌──────────────────┐                     ┌──────────────────┐
│ Scanner 掃描器   │                     │  Data  (D0~D3)   │
│   (74138 解碼)   │                     │                  │
└────────┬─────────┘                     │  74373 鎖存器    │
         │ FE, FD, FB, F7 (低電位輪流掃) │  (Latch Register)│
         ▼                               │                  │
  [ 4×4 鍵盤 Row ]                       │  Output (Q0~Q3)  │
         │                               └─────────▲────────┘
   (若按下按鍵)                                    │
         │                                         │ 鎖存脈衝 (G / CLK)
         ▼                                         │ (當有按鍵按下時觸發)
  [ 4×4 鍵盤 Col ]                                 │
         │ (傳回 Col0~Col3)                        │
         └─────────────► ┌─────────────────────────┴─┐
                         │  按鍵按下判斷邏輯 (Detector)│
                         │    (用 NAND 門偵測低電位)  │
                         └───────────────────────────┘

```

### 3 個步驟看懂這個架構在做什麼：

1. **左邊（發射）：****Mod16 計數器** 負責從 0 數到 15，**Scanner (74138)** 把數字變成你剛剛模擬出來的 `FE, FD, FB, F7`，發射給鍵盤的 Row 0~3。
2. **中間（接收）：** 鍵盤平時沒按時，Col 輸入全是 `1111`。當你按了按鍵，Col 會傳回包含 `0` 的訊號。**NAND 門**只要看到 `0` 出現，就代表「**有人按按鍵了！**」，發出一個抓取訊號。
3. **右邊（記憶）：****74373 鎖存器** 接到抓取訊號的瞬間，就把 **Mod16 當下的數字（0~15）** 拍照存下來。這樣畫面上就會一直顯示你按下的數字，不會隨著 Scanner 一直變動。

### BDF 拉線重點總結
你現在已經做好了圖中的 **Mod16** 與 **Scanner (74138)**！
下一步在 Quartus BDF 上只需要加兩顆元件：

- 放一顆 **NAND4 (4輸入與非門)**：輸入接 4 條 Col 線，輸出接 74373 的 `G` 腳。
- 放一顆 **74373 (Latch)**：數據輸入 $D$ 接 Mod16 的 4 位元輸出，輸出 $Q$ 就是最終按鍵碼。

---

*Exported from [Voyager](https://github.com/Nagi-ovo/voyager)*  
*Generated on August 23, 2026 at 03:15 PM*