cd# 前端部署到 Raspberry Pi（PowerShell + scp）

> **先講清楚我做了什麼假設**（沒有實際 Pi 可以測試，這份文件沒有被執行過，
> 是照專案現有檔案結構寫的，請照你實際環境核對再用）：
> - 專案裡「網頁前端」實際會被瀏覽器執行的，是 PyScript（Pyodide，在瀏覽器
>   裡跑 Python）版本，位於 `web/frontend/src/components/`，只有三個檔案：
>   `index.html`、`style.css`、`action.py`，純靜態、不需要 build/npm。
>   （`web/frontend/` 底下另外還有一支 `script.js`，那是 `firebase.json`
>   設定 Firebase Hosting 用的另一個版本，跟這份文件無關，這裡不動它。）
> - `action.py` 開頭的 `BACKEND_URL = ""` 是空字串（相對路徑），代表現在
>   預設「前端跟後端 API 同一個網域」。部署到 Pi 之後，如果後端沒有也架在
>   同一台 Pi 上，這一行要手動改成後端的完整網址（見下面步驟三）。
> - Pi 上的靜態伺服器方案我選了最簡單、Raspberry Pi OS 內建就有的
>   `python3 -m http.server`，沒有另外假設你已經裝 nginx。想用 nginx 的話
>   步驟四/五要自己換掉。
> - 不會動 `web/backend/.env`、`web/frontend/src/.env`
>   （CLAUDE.md 規則 #05：.env 一律由你自己手動輸入，這份文件沒有任何一步
>   會讀寫這兩個檔案）。

## 佔位變數（每個指令範例都要換成你的實際值）

| 佔位 | 意思 | 範例 |
|---|---|---|
| `<PI_HOST>` | Pi 的 IP 或主機名稱 | `192.168.1.50` 或 `raspberrypi.local` |
| `<PI_USER>` | SSH 使用者名稱 | `pi` |
| `<PI_DIR>` | Pi 上放前端檔案的目錄 | `/home/pi/ai-module-frontend` |

## 前置需求

- **Windows 這邊**：PowerShell 要能用 `scp`／`ssh`（Windows 10/11 內建
  OpenSSH Client，通常已經有）：
  ```powershell
  Get-Command scp
  Get-Command ssh
  ```
  如果報錯找不到指令：設定 → 應用程式 → 選用功能 → 新增功能 → 安裝
  「OpenSSH 用戶端」。
- **Pi 那邊**：SSH 服務要開著，先確認能連得上：
  ```powershell
  ssh <PI_USER>@<PI_HOST>
  ```
  第一次連線會問要不要信任主機指紋，輸入 `yes`。
- Pi 上要有 `python3`（Raspberry Pi OS 預設就有，`python3 --version` 確認）。

## 步驟一：在 Pi 上建立目標資料夾

```powershell
ssh <PI_USER>@<PI_HOST> "mkdir -p <PI_DIR>"
```

## 步驟二：用 scp 把前端三個檔案複製過去

在 PowerShell（專案根目錄）執行：

```powershell
scp `
  "web\frontend\src\components\index.html" `
  "web\frontend\src\components\style.css" `
  "web\frontend\src\components\action.py" `
  <PI_USER>@<PI_HOST>:<PI_DIR>/
```

（PowerShell 續行符號是反引號 `` ` ``，不是 bash 的反斜線 `\`——照抄上面這段
到 PowerShell 就能跑，不用自己改。）

確認檔案真的到了：

```powershell
ssh <PI_USER>@<PI_HOST> "ls -la <PI_DIR>"
```

## 步驟三：SSH 進 Pi，用 `nano` 檢查／修改後端網址

```powershell
ssh <PI_USER>@<PI_HOST>
```

進到 Pi 的 shell 之後：

```bash
nano <PI_DIR>/action.py
```

找到這一行（檔案開頭附近）：

```python
BACKEND_URL = ""
```

依你實際狀況改成其中一種：

```python
# 後端還是架在 Render.com（web/render.yaml 那個部署）
BACKEND_URL = "https://ai-module-backend.onrender.com"
```

```python
# 後端也一起搬到同一台 Pi 上跑（例如 uvicorn 監聽 8000 port）
BACKEND_URL = "http://127.0.0.1:8000"
```

改完在 `nano` 裡：`Ctrl+O` 存檔 → `Enter` 確認檔名 → `Ctrl+X` 離開。

## 步驟四：在 Pi 上啟動靜態網頁伺服器（先測試用）

```bash
cd <PI_DIR>
python3 -m http.server 8080
```

區網內用瀏覽器打開 `http://<PI_HOST>:8080` 應該就能看到前端頁面。這個方式
是前景執行，`Ctrl+C` 或 SSH 斷線就會停掉，適合先確認能不能動，正式長期
掛著請看下一步。

## 步驟五（選用）：設成開機自動啟動的 systemd 服務

在 Pi 上（SSH session 裡）用 `nano` 建立 service 檔：

```bash
sudo nano /etc/systemd/system/ai-module-frontend.service
```

貼上以下內容（`<PI_DIR>`、`<PI_USER>` 記得換成實際值）：

```ini
[Unit]
Description=AI-module 前端靜態伺服器
After=network.target

[Service]
Type=simple
User=<PI_USER>
WorkingDirectory=<PI_DIR>
ExecStart=/usr/bin/python3 -m http.server 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

存檔離開（`Ctrl+O` → `Enter` → `Ctrl+X`），然後啟用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-module-frontend
sudo systemctl status ai-module-frontend
```

`status` 顯示 `active (running)` 就代表開機會自動啟動，不用每次手動下指令。

## 之後要更新前端內容

本機改完 `index.html`／`style.css`／`action.py` 之後，重新執行**步驟二**的
`scp` 指令覆蓋過去就好；`http.server` 每次請求都是即時讀檔，不用重啟服務
（如果改成 nginx 之類有快取的方案就要另外處理）。

## 一鍵重新部署的 PowerShell 腳本（選用）

存成專案裡的 `deploy_frontend.ps1`，之後只要跑 `.\deploy_frontend.ps1` 就會
複製最新檔案並重啟服務：

```powershell
param(
    [string]$PiHost = "<PI_HOST>",
    [string]$PiUser = "<PI_USER>",
    [string]$PiDir  = "<PI_DIR>"
)

$files = @(
    "web\frontend\src\components\index.html",
    "web\frontend\src\components\style.css",
    "web\frontend\src\components\action.py"
)

scp $files "$PiUser@${PiHost}:$PiDir/"
ssh "$PiUser@$PiHost" "sudo systemctl restart ai-module-frontend"
```

執行前記得把檔案開頭的 `<PI_HOST>`／`<PI_USER>`／`<PI_DIR>` 換成你的實際值，
或執行時用參數帶入：

```powershell
.\deploy_frontend.ps1 -PiHost "192.168.1.50" -PiUser "pi" -PiDir "/home/pi/ai-module-frontend"
```

---

# 進階：用 Tailscale Serve 掛在 `https://<機器名>.<tailnet>.ts.net/AI-Module/`

> **一樣先講清楚假設**（沒有實際 Pi5、也沒有你 tailnet 的 admin console
> 存取權限，這份文件同樣沒有被執行過，請照你實際環境核對）：
> - 這個網址格式（`xxx.ts.net`）是 **Tailscale** 自動配發的 tailnet
>   HTTPS 網域，跟 **DuckDNS**（`xxx.duckdns.org`）是兩個不相關的機制，
>   DuckDNS 接不上這種網址，這份文件全程只用 Tailscale，不會碰 DuckDNS
>   相關設定。
> - 上面「基本版」是前端三個靜態檔案獨立用 `http.server` 開，後端另外開
>   `/api/detect` 讓前端 `fetch`。這裡改成**前後端合併成一個 FastAPI
>   process**（`web/backend/app.py` 直接把 `frontend/src/components/`
>   三個檔案也一起服務），原因是 Tailscale Serve 的 `--set-path` 會把完整
>   路徑（含 `/AI-Module` 前綴）原樣轉給後端、不會自動 strip 掉——如果
>   前後端分開兩個 process，會需要兩條 `tailscale serve --set-path` 各自
>   對齊前綴，比較容易兜不起來，合併成一個 process、一個前綴、一次
>   mount 最單純。
> - `app.py` 已經改成支援 `MOUNT_PREFIX` 環境變數（預設空字串＝掛根目錄，
>   跟原本行為完全一樣；設成 `/AI-Module` 才會把所有路由掛在這個前綴
>   下）。`action.py` 也已經改成用不帶開頭斜線的相對路徑呼叫 API，會自動
>   跟著頁面網址所在的子路徑走，不需要另外改。
> - **這個做法假設 Tailscale Serve 不會 strip 前綴**（查證官方文件
>   `tailscale.com/kb/1242/tailscale-serve` 只寫「append」沒提到會
>   rewrite，但沒有明文保證）——步驟五有驗證指令，如果打開網址後 API
>   全部 404，見步驟五最後的疑難排解。

## 前置需求（在 Pi5 上）

- Tailscale 已安裝、已登入、能在 `tailscale status` 看到這台機器（你說
  已經裝了，這裡不重複裝機步驟）。
- **tailnet 要先手動開 HTTPS**（這步只能在網頁 admin console 做，沒辦法
  用指令）：登入 <https://login.tailscale.com/admin/dns>，確認
  MagicDNS 已開啟，「HTTPS Certificates」按 Enable HTTPS（會跳出「機器
  名稱會公開在 Certificate Transparency ledger」的提示，確認接受）。
  這步跳過的話，`tailscale serve` 會發不出 HTTPS 憑證。
- `python3`／`pip3`（Raspberry Pi OS 內建）。

## 步驟一：把整個 `web/` 資料夾（不含 `.env`）跟 `tranning/` scp 過去

`web/backend/app.py` 會 `import` `tranning/chats.py`，所以這次不能只複製
`web/`，`tranning/` 也要一起帶過去。在 PowerShell（專案根目錄）執行：

```powershell
ssh <PI_USER>@<PI_HOST> "mkdir -p <PI_DIR>/web/backend <PI_DIR>/web/frontend/src/components <PI_DIR>/tranning"

scp -r `
  "web\backend\app.py" `
  "web\backend\detector.py" `
  <PI_USER>@<PI_HOST>:<PI_DIR>/web/backend/

scp `
  "web\frontend\src\components\index.html" `
  "web\frontend\src\components\style.css" `
  "web\frontend\src\components\action.py" `
  <PI_USER>@<PI_HOST>:<PI_DIR>/web/frontend/src/components/

scp -r "tranning" <PI_USER>@<PI_HOST>:<PI_DIR>/
```

`tranning/` 底下如果有大型 checkpoint（`chat_runs/`、`code_runs/`）第一次
會傳比較久，之後只有程式碼變動時可以改用 `scp` 個別檔案，不用整包重傳。

## 步驟二：在 Pi5 上裝 Python 依賴

```powershell
ssh <PI_USER>@<PI_HOST>
```

進到 Pi 的 shell 之後：

```bash
cd <PI_DIR>
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart pydantic opencv-python-headless numpy ultralytics
```

Pi5 是 ARM、沒有 CUDA，`detector.py` 的 `_device()` 會自動退回 CPU 跑
`yolo11n.pt`（nano 模型），第一次呼叫 `/api/detect`（或 `/ws/detect`）時
會自動下載權重檔，需要能連外網。

## 步驟三：手動先跑一次確認能動

```bash
cd <PI_DIR>
source .venv/bin/activate
MOUNT_PREFIX=/AI-Module PORT=8000 python web/backend/app.py
```

另開一個 SSH session（或用 `curl 127.0.0.1:8000/AI-Module/api/health`
本機測）確認回 `{"status":"ok"}` 沒問題後，`Ctrl+C` 關掉，進下一步設成
背景常駐服務。

## 步驟四：設成開機自動啟動的 systemd 服務

```bash
sudo nano /etc/systemd/system/ai-module-backend.service
```

貼上（`<PI_DIR>`、`<PI_USER>` 換成實際值）：

```ini
[Unit]
Description=AI-module 後端（前後端合併，掛在 /AI-Module 子路徑）
After=network.target

[Service]
Type=simple
User=<PI_USER>
WorkingDirectory=<PI_DIR>
Environment=MOUNT_PREFIX=/AI-Module
Environment=PORT=8000
ExecStart=<PI_DIR>/.venv/bin/python web/backend/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

存檔離開（`Ctrl+O` → `Enter` → `Ctrl+X`），啟用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-module-backend
sudo systemctl status ai-module-backend
```

## 步驟五：開 Tailscale Serve 掛到 `/AI-Module`

```bash
sudo tailscale serve --bg --https=443 --set-path=/AI-Module http://127.0.0.1:8000
tailscale serve status
```

`tailscale serve status` 應該會列出 `/AI-Module` 這個 mount 指到
`http://127.0.0.1:8000`。用瀏覽器（或另一台在同一個 tailnet 的裝置）打開：

```
https://raspberrypi.tail8767da.ts.net/AI-Module/
```

**驗證**（Pi5 上直接跑，或從同 tailnet 的其他裝置跑都可以）：

```bash
curl -s https://raspberrypi.tail8767da.ts.net/AI-Module/api/health
# 應該回 {"status":"ok"}
```

**疑難排解**：如果這個 curl 回 404，代表 Tailscale Serve 把 `/AI-Module`
前綴 strip 掉了（跟本文件開頭的假設相反）——這時把 systemd service 檔的
`Environment=MOUNT_PREFIX=/AI-Module` 整行刪掉（改回掛根目錄），
`sudo systemctl daemon-reload && sudo systemctl restart ai-module-backend`，
再測一次同一個 curl 網址應該就會通。

## 之後要更新程式碼

改完本機的 `web/backend/*.py`、`web/frontend/src/components/*` 之後，重複
**步驟一**的 scp，再 `ssh <PI_USER>@<PI_HOST> "sudo systemctl restart ai-module-backend"`
讓新程式碼生效（跟基本版的 `http.server` 不同，這是常駐 process，改完檔案
不會自動重新載入）。

---

# 進階（二）：用 DuckDNS + Caddy，前後端各自獨立網址

> **先講清楚假設**（一樣沒有實際 Pi、沒有你的 DuckDNS 帳號/路由器可以測試，
> 這份文件沒有被執行過，請照實際環境核對）：
> - 這條路線跟上面「進階：Tailscale Serve」是**互斥的兩條路**，差別是
>   Tailscale 走它自己的 VPN 網路（不用開路由器 port，但只有裝了 Tailscale
>   的裝置能連），DuckDNS 走**真正的公開網際網路**（誰都能連，但要自己開
>   路由器 port、自己搞定 HTTPS 憑證）。兩者只需要選一個，不用兩個都做。
> - 這裡採用**前後端各自獨立網址**（不是上一版那種合併成一個 process 掛
>   同一個網域）：`<DUCKDNS_SUBDOMAIN_WEB>.duckdns.org` 是純靜態前端三個
>   檔案，`<DUCKDNS_SUBDOMAIN_API>.duckdns.org` 是 FastAPI 後端，Caddy
>   用**兩個 site block**分流。這樣前後端可以分開重啟/更新，也符合「前端
>   單獨拉一個網址」的需求。代價是前端呼叫後端變成**跨網域**請求——
>   `web/backend/app.py` 目前 `CORSMiddleware` 是 `allow_origins=["*"]`
>   （[app.py:66](web/backend/app.py#L66)），已經放行任何來源，這裡不用
>   額外改後端 CORS 設定。
> - DuckDNS 本身**只做動態 DNS**，不提供 HTTPS，這裡一樣用 **Caddy** 自動
>   跟 Let's Encrypt 要憑證、自動續期。
> - **最大風險**：如果你的網路業者（尤其台灣不少家用網路、行動網路分享）
>   把你家對外 IP 放在 **CGNAT** 後面，路由器就沒有真正公開的 IP 可以開
>   port，這條路線整個走不通——步驟二會有指令幫你先確認這件事。
> - 這個架構下 `MOUNT_PREFIX` 留空（後端掛根目錄），但 `action.py` 的
>   `BACKEND_URL` **這次不能留空**，要改成後端那個網域的完整網址（見
>   步驟五），因為前端頁面跟後端 API 現在是兩個不同網域，相對路徑行不通。
> - 開了公開 port 就是把這台 Pi 暴露在整個網際網路上，跟 Tailscale（只有
>   你自己的裝置能連）風險等級不同，請自行評估是否要開。

## 佔位變數（沿用最上面表格，額外多這些）

| 佔位 | 意思 | 範例 |
|---|---|---|
| `<DUCKDNS_SUBDOMAIN_WEB>` | 前端要用的子網域（不含 `.duckdns.org`） | `ai-module-roy` |
| `<DUCKDNS_SUBDOMAIN_API>` | 後端 API 要用的子網域（不含 `.duckdns.org`） | `ai-module-roy-api` |
| `<DUCKDNS_TOKEN>` | duckdns.org 帳號頁面顯示的 token（兩個子網域共用同一個） | `xxxxxxxx-xxxx-...` |

## 前置需求

- 去 <https://www.duckdns.org> 用 GitHub/Google 帳號登入，建立**兩個**子
  網域（例如 `ai-module-roy` 跟 `ai-module-roy-api`；DuckDNS 免費帳號預設
  可以建立到 5 個子網域，兩個夠用），記下它給你的 **token**（帳號頁面最
  上面那一串，兩個子網域共用同一個 token）。
- **路由器要能設定 port forwarding**（這步只能在路由器管理介面做，通常是
  瀏覽器打開 `192.168.1.1` 之類的網址，依廠牌不同，沒辦法用指令自動化）：
  把外部 80、443 port 轉發到 Pi 的區網 IP（`ssh <PI_USER>@<PI_HOST> "hostname -I"`
  可以查 Pi 目前的區網 IP，建議在路由器上把這個 IP 設成固定/保留，不然
  Pi 重開機换了 IP，port forwarding 會失效）。兩個子網域都指向同一台 Pi、
  同一個 IP，靠 Caddy 用**主機名稱**分流，不需要開兩組不同的 port。

## 步驟一：Pi 上裝一個 DuckDNS IP 同步的 cron job

DuckDNS 不會自動知道你家 IP 換了，要 Pi 自己定期回報。DuckDNS 的
update API 支援一次更新多個子網域（`domains=` 用逗號分隔，不要有空白）。
SSH 進 Pi：

```bash
mkdir -p ~/duckdns
cat > ~/duckdns/duck.sh <<'EOF'
#!/bin/bash
echo url="https://www.duckdns.org/update?domains=<DUCKDNS_SUBDOMAIN_WEB>,<DUCKDNS_SUBDOMAIN_API>&token=<DUCKDNS_TOKEN>&ip=" | curl -k -o ~/duckdns/duck.log -K -
EOF
chmod +x ~/duckdns/duck.sh
~/duckdns/duck.sh
cat ~/duckdns/duck.log
```

`duck.log` 印出 `OK` 才代表成功（`<DUCKDNS_SUBDOMAIN_WEB>`、
`<DUCKDNS_SUBDOMAIN_API>`、`<DUCKDNS_TOKEN>` 記得換成實際值）。成功後設成
每 5 分鐘自動跑一次：

```bash
crontab -e
```

在編輯器裡加最後一行（第一次執行 `crontab -e` 可能會先問要用哪個編輯器，
選 `nano` 最簡單）：

```
*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1
```

存檔離開。

## 步驟二：確認家用 IP 不是 CGNAT（有沒有真正公開的 IP）

在 Pi 上跑：

```bash
curl -s ifconfig.me
```

跟你在路由器管理介面看到的「WAN IP」比對，兩個一樣才代表你有真正公開的
IP、port forwarding 才有意義。如果路由器管理介面顯示的 WAN IP 是
`100.64.0.0/10`、`10.x.x.x`、`172.16.x.x`～`172.31.x.x` 這種私有網段，
代表你在 CGNAT 後面，這條 DuckDNS 路線走不通，建議改用上面的 Tailscale
方案。

## 步驟三：scp 程式碼、裝依賴

跟「進階：Tailscale Serve」的**步驟一～二**完全一樣（後端 `app.py`／
`detector.py`、前端三個靜態檔、`tranning/` 都要 scp 過去；前端這次會被
Caddy 直接當靜態檔案服務，不經過 `app.py`，但路徑還是沿用
`<PI_DIR>/web/frontend/src/components/`，不用額外建目錄）：

```powershell
ssh <PI_USER>@<PI_HOST> "mkdir -p <PI_DIR>/web/backend <PI_DIR>/web/frontend/src/components <PI_DIR>/tranning"

scp -r `
  "web\backend\app.py" `
  "web\backend\detector.py" `
  <PI_USER>@<PI_HOST>:<PI_DIR>/web/backend/

scp `
  "web\frontend\src\components\index.html" `
  "web\frontend\src\components\style.css" `
  "web\frontend\src\components\action.py" `
  <PI_USER>@<PI_HOST>:<PI_DIR>/web/frontend/src/components/

scp -r "tranning" <PI_USER>@<PI_HOST>:<PI_DIR>/
```

裝依賴（同「進階：Tailscale Serve」步驟二）：

```bash
ssh <PI_USER>@<PI_HOST>
cd <PI_DIR>
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart pydantic opencv-python-headless numpy ultralytics
```

## 步驟四：後端跑起來、設成 systemd 服務

先手動跑一次確認能動（`MOUNT_PREFIX` 留空、掛根目錄）：

```bash
cd <PI_DIR>
source .venv/bin/activate
PORT=8000 python web/backend/app.py
```

另開一個 SSH session 跑 `curl 127.0.0.1:8000/api/health` 確認回
`{"status":"ok"}` 後 `Ctrl+C` 關掉。接著設成常駐服務：

```bash
sudo nano /etc/systemd/system/ai-module-backend.service
```

```ini
[Unit]
Description=AI-module 後端 API（獨立網域，只服務 /api、/ws，不服務前端靜態檔）
After=network.target

[Service]
Type=simple
User=<PI_USER>
WorkingDirectory=<PI_DIR>
Environment=PORT=8000
ExecStart=<PI_DIR>/.venv/bin/python web/backend/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-module-backend
sudo systemctl status ai-module-backend
```

> 註：`app.py` 本身仍然「順便」會服務 `FRONTEND_DIR` 的靜態檔（它沒有開關
> 可以關掉這個行為），但因為前端這次是走另一個網域
> `<DUCKDNS_SUBDOMAIN_WEB>.duckdns.org`（見步驟六），瀏覽器根本不會去打
> `<DUCKDNS_SUBDOMAIN_API>.duckdns.org/`，所以這個順便服務的靜態檔路徑實際
> 上不會被用到，可以忽略，不影響功能。

## 步驟五：改前端的 `BACKEND_URL`，指到後端獨立網域

因為前端跟後端現在是兩個不同網域，`action.py` 的
`BACKEND_URL = ""`（相對路徑）行不通，要**手動改成後端的完整網址**。
在 Pi 上：

```bash
nano <PI_DIR>/web/frontend/src/components/action.py
```

找到（[action.py:25](web/frontend/src/components/action.py#L25)）：

```python
BACKEND_URL = ""
```

改成（`<DUCKDNS_SUBDOMAIN_API>` 換成實際值）：

```python
BACKEND_URL = "https://<DUCKDNS_SUBDOMAIN_API>.duckdns.org"
```

存檔離開（`Ctrl+O` → `Enter` → `Ctrl+X`）。之後每次本機改完
`action.py` 重新 scp 上去，都要記得這一行在 Pi 上是**手動改過的版本**，
直接覆蓋會被本機的 `BACKEND_URL = ""` 蓋掉——建議 scp 完後這行再手動改一次，
或是把這行改法記在自己的部署筆記裡。

## 步驟六：裝 Caddy，兩個 site block 分流

Raspberry Pi OS（Debian 系）用官方套件庫裝最新版 Caddy：

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

編輯 Caddy 設定檔：

```bash
sudo nano /etc/caddy/Caddyfile
```

整份內容換成（`<DUCKDNS_SUBDOMAIN_WEB>`、`<DUCKDNS_SUBDOMAIN_API>`、
`<PI_DIR>` 都換成實際值；前端這個 block 直接用 `file_server` 服務靜態檔，
不經過 `app.py`）：

```
<DUCKDNS_SUBDOMAIN_WEB>.duckdns.org {
    root * <PI_DIR>/web/frontend/src/components
    file_server
}

<DUCKDNS_SUBDOMAIN_API>.duckdns.org {
    reverse_proxy 127.0.0.1:8000
}
```

存檔離開，重啟 Caddy：

```bash
sudo systemctl restart caddy
sudo systemctl status caddy
```

Caddy 會自動幫兩個網域各自跟 Let's Encrypt 要憑證、自動續期（需要路由器
已經把 80、443 都轉發過來，80 port 是給 ACME HTTP-01 驗證用的）。

## 驗證

```bash
curl -s https://<DUCKDNS_SUBDOMAIN_API>.duckdns.org/api/health
# 應該回 {"status":"ok"}
```

瀏覽器打開 `https://<DUCKDNS_SUBDOMAIN_WEB>.duckdns.org/` 應該能看到前端
頁面、鎖頭圖示（憑證）正常，且對話／相機偵測功能能正常打到
`<DUCKDNS_SUBDOMAIN_API>.duckdns.org`（瀏覽器 F12 開發者工具的 Network
分頁可以看到請求打去哪個網域，跨網域請求應該回 200，不會被 CORS 擋——
`allow_origins=["*"]` 已經放行）。

**疑難排解**：
- 打不開／連線逾時：先確認步驟二的公開 IP 檢查有過，再確認路由器 80/443
  port forwarding 真的指到 Pi 目前的區網 IP（IP 是否因為沒設保留而換掉了）。
- 網址打得開但沒有 HTTPS 鎖頭／憑證錯誤：`sudo journalctl -u caddy -n 50`
  看 Caddy log，通常是 80 port 沒轉發成功，ACME HTTP-01 驗證會失敗；兩個
  網域是各自獨立驗證，其中一個失敗不會影響另一個。
- 前端頁面打得開，但對話／相機偵測完全沒反應：F12 開發者工具 Console
  看有沒有 CORS 或連線錯誤，確認 `action.py` 裡的 `BACKEND_URL`
  （步驟五）真的改成 API 網域的完整網址，不是空字串。
- `/ws/detect` 連不上（相機偵測沒畫面）：確認瀏覽器網址列是
  `https://`（不是 `http://`），`_detect_ws_url()`
  （[action.py:40-54](web/frontend/src/components/action.py#L40-L54)）
  會依 `BACKEND_URL` 的 scheme 自動換成 `wss://`，Caddy 對 WebSocket
  的反向代理是內建支援、不需要額外設定。

## 之後要更新程式碼

- 後端變動：重新 scp `web/backend/*.py`，
  `ssh <PI_USER>@<PI_HOST> "sudo systemctl restart ai-module-backend"`。
- 前端變動：重新 scp `index.html`／`style.css`／`action.py`，因為
  Caddy 的 `file_server` 是即時讀檔，**不用重啟任何服務**；但別忘了
  `action.py` 裡的 `BACKEND_URL` 要照步驟五重新手動改一次（本機版本是
  空字串，會被覆蓋掉）。
- Caddy 設定（`Caddyfile`）沒變的話不用重啟 Caddy。
