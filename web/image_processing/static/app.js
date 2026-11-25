"use strict";
const $ = id => document.getElementById(id);
let selectedFile = null, previewURL = null, generation = 0, controller = null, ticker = null, startedAt = 0;
const kernels = {
  filter1: ["1  0  1\n0  4  0\n1  0  1", "保留原係數，不除以 8。亮度可能超過顯示範圍。"],
  filter2: ["5 × 5\n全 1 / 25", "對 5 × 5 鄰域取平均，觀察平滑效果。"],
  filter3: ["−1  0  1\n−2  0  2\n−1  0  1", "Sobel 核先翻轉 180°；原始結果保留正負值。"]
};
const operation = () => document.querySelector('input[name="operation"]:checked').value;
function status(message, busy = false) { $("status").textContent = message; $("status").classList.toggle("busy", busy); }
function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function image(id, url) {
  const element = $(id);
  element.hidden = !url;
  if (url) element.src = url; else element.removeAttribute("src");
  const placeholder = $(id.replace("-image", "-placeholder"));
  if (placeholder) placeholder.hidden = !!url;
}
function stop() {
  generation++;
  controller?.abort(); controller = null;
  clearInterval(ticker); ticker = null;
  $("process-button").disabled = !selectedFile;
  $("process-button").firstChild.textContent = "開始處理 ";
}
function resetResults() {
  $("error").hidden = true;
  $("result-details").hidden = true;
  $("histograms").hidden = true;
  $("download-list").replaceChildren();
  image("after-image", null);
  $("before-label").textContent = "處理前";
  $("after-label").textContent = "處理後";
  $("after-caption").textContent = "處理完成後可下載結果";
}
function choose(file) {
  stop(); resetResults();
  selectedFile = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  image("before-image", null);
  $("before-placeholder").querySelector("p").textContent = "原始影像將顯示在這裡";
  $("selection").hidden = true;
  $("before-caption").textContent = "選擇圖片以開始";
  $("process-button").disabled = true;
  if (!file) { status("等待上傳"); return; }
  if (file.size > 10 * 1024 * 1024) { showError("圖片超過 10 MiB 上限，請選擇較小的檔案。"); status("檔案過大"); return; }
  if (!/\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name)) { showError("請選擇 PNG、JPG、WebP、TIFF 或 BMP 圖片。"); status("格式不支援"); return; }
  selectedFile = file;
  $("filename").textContent = file.name;
  $("filesize").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MiB`;
  $("selection").hidden = false;
  $("before-caption").textContent = "原尺寸處理 · 尺寸與格式由本機服務驗證";
  previewURL = URL.createObjectURL(file);
  $("before-image").onerror = () => {
    image("before-image", null);
    $("before-placeholder").querySelector("p").textContent = "瀏覽器無法預覽；處理後由本機服務提供 PNG";
  };
  image("before-image", previewURL);
  $("process-button").disabled = false;
  status("圖片已就緒");
}
function localURL(value) {
  const url = new URL(value, window.location.origin);
  if (!["http:", "https:"].includes(url.protocol) || url.origin !== window.location.origin) throw new Error("服務回傳了非本機的結果網址。");
  if (!/^\/api\/jobs\/[a-f0-9]{32}(?:\/files\/[A-Za-z0-9_-]+\.(?:png|json|csv))?$/.test(url.pathname)
      || url.hash || (url.search && url.search !== "?download=1")) throw new Error("服務回傳了非預期的結果網址。");
  return url.href;
}
async function jsonResponse(response) {
  let data;
  try { data = await response.json(); } catch { throw new Error(`本機服務回應無法讀取（HTTP ${response.status}）。`); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `請求失敗（HTTP ${response.status}）。`);
  return data;
}
function render(data, elapsed, mode, options) {
  const previews = data.previews || {}, input = data.input || {};
  $("before-image").onerror = null;
  image("before-image", previews.before ? localURL(previews.before) : null);
  image("after-image", previews.after ? localURL(previews.after) : null);
  $("before-label").textContent = mode === "histogram" ? "處理前 · 亮度 Y" : `處理前 · ${options.channel === "rgb" ? "RGB" : "灰階"}`;
  $("after-label").textContent = mode === "histogram" ? "處理後 · 均衡化 Y" : "處理後 · 卷積";
  $("before-caption").textContent = `${input.width} × ${input.height} px · ${input.name || selectedFile.name}`;
  $("after-caption").textContent = mode === "histogram" ? "CDF-min 全域均衡化 · Y 灰階" : `${options.display === "minmax" ? "Min–max 顯示映射" : "0–255 顯示裁切"} · 原始數值見 JSON`;
  $("summary").replaceChildren();
  for (const label of [`${input.width} × ${input.height} 像素`, `本機運算 ${elapsed.toFixed(1)} 秒`, "原始數值可下載"]) {
    const item = document.createElement("span"); item.textContent = label; $("summary").append(item);
  }
  if (previews.histogram_before && previews.histogram_after) {
    $("histogram-before").src = localURL(previews.histogram_before);
    $("histogram-after").src = localURL(previews.histogram_after);
    $("histograms").hidden = false;
  }
  $("download-list").replaceChildren();
  for (const file of data.files || []) {
    const link = document.createElement("a"), title = document.createElement("span"), name = document.createElement("small"), arrow = document.createElement("b");
    link.className = "download-link"; link.href = localURL(file.download_url); link.download = file.name;
    title.textContent = file.label || file.name; name.textContent = file.name; arrow.textContent = "↓"; arrow.setAttribute("aria-hidden", "true");
    title.append(name); link.append(title, arrow); $("download-list").append(link);
  }
  $("result-details").hidden = false;
}
async function processFile() {
  if (!selectedFile || controller) return;
  resetResults();
  const token = ++generation, activeFile = selectedFile, mode = operation();
  const options = Object.fromEntries(["filter", "padding", "channel", "display"].map(id => [id, $(id).value]));
  controller = new AbortController(); const signal = controller.signal;
  $("process-button").disabled = true;
  $("process-button").firstChild.textContent = "處理中… ";
  startedAt = Date.now();
  let state = "上傳中";
  const update = () => status(`${state} · ${((Date.now() - startedAt) / 1000).toFixed(1)} 秒`, true);
  update(); ticker = setInterval(update, 300);
  try {
    const query = new URLSearchParams({ operation: mode, ...options });
    const job = await jsonResponse(await fetch(`/api/jobs?${query}`, { method: "POST", body: activeFile, headers: { "Content-Type": activeFile.type || "application/octet-stream", "X-File-Name": encodeURIComponent(activeFile.name) }, signal }));
    if (token !== generation) return;
    while (token === generation) {
      const data = await jsonResponse(await fetch(localURL(job.status_url || `/api/jobs/${job.id}`), { signal }));
      if (token !== generation) return;
      if (data.status === "error" || data.status === "failed") throw new Error(data.message || data.error || "影像處理失敗。");
      if (data.status === "done" || data.status === "succeeded") {
        render(data.result || data, Number(data.elapsed_seconds ?? (Date.now() - startedAt) / 1000), mode, options);
        status("處理完成"); break;
      }
      state = data.status === "queued" ? "排隊中" : "處理中"; update();
      await new Promise(resolve => setTimeout(resolve, 700));
    }
  } catch (error) {
    if (token === generation && error.name !== "AbortError") { showError(error instanceof TypeError ? "無法連線至本機服務，請確認伺服器正在執行。" : error.message || "影像處理失敗，請重試。"); status("處理失敗"); }
  } finally {
    if (token === generation) { clearInterval(ticker); ticker = null; controller = null; $("process-button").disabled = !selectedFile; $("process-button").firstChild.textContent = "開始處理 "; }
  }
}
$("file-input").addEventListener("change", event => choose(event.target.files[0]));
$("clear-button").addEventListener("click", () => { $("file-input").value = ""; choose(null); });
$("process-button").addEventListener("click", processFile);
for (const event of ["dragenter", "dragover"]) $("drop-zone").addEventListener(event, e => { e.preventDefault(); $("drop-zone").classList.add("drag-over"); });
for (const event of ["dragleave", "drop"]) $("drop-zone").addEventListener(event, e => { e.preventDefault(); $("drop-zone").classList.remove("drag-over"); });
$("drop-zone").addEventListener("drop", event => { $("file-input").value = ""; choose(event.dataTransfer.files[0]); });
$("filter").addEventListener("change", () => { const [kernel, note] = kernels[$("filter").value]; $("kernel").textContent = kernel; $("filter-description").textContent = note; });
document.querySelectorAll('input[name="operation"]').forEach(input => input.addEventListener("change", () => {
  $("convolution-options").hidden = operation() !== "convolution";
  $("histogram-options").hidden = operation() !== "histogram";
}));
