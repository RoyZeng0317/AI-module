// Google Identity Services（Sign In With Google）——取代原本的 Firebase
// Authentication，不再依賴任何 Firebase 專案設定，直接對 Google 官方的
// OAuth 2.0 服務。跟 web/frontend/src/components/google-auth-init.js 同一份
// 邏輯（那份還會另外掛 Analytics，這裡是 FastAPI 實際服務的介面，不需要）。
// GOOGLE_CLIENT_ID 是在 Google Cloud Console 申請的 OAuth 用戶端 ID
// （Web application 類型），這是公開資訊、不是密鑰，可以直接寫在前端；真正
// 的安全邊界在後端 web/backend/auth.py 用同一組 ID 當 audience 驗證 ID
// token 的簽章。三個地方（這支檔案、web/frontend 那份、
// web/backend/auth.py）都要填同一組 Client ID，否則驗證一律失敗。
const GOOGLE_CLIENT_ID = "1074357742383-tm3gopumph7ndvurmlgfdtm9bsncd9hp.apps.googleusercontent.com";

// action.py（PyScript/Pyodide）沒辦法直接掛回呼給 GIS 的 initialize()，所以
// 把幾個包好 Promise 的函式掛到 window 上，沿用原本 firebase-init.js 的
// 命名（sincoSignIn/sincoSignOut/sincoGetIdToken/sincoOnAuthChanged），
// 只多一個 sincoRenderGoogleButton 給正式登入按鈕用，action.py 其餘呼叫端
// 完全不用改。
let _idToken = null;
let _authChangeCallback = null;

function _decodeJwtPayload(jwt) {
  const b64 = jwt.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
  const json = decodeURIComponent(
    atob(b64).split("").map((c) => "%" + c.charCodeAt(0).toString(16).padStart(2, "0")).join("")
  );
  return JSON.parse(json);
}

function _handleCredential(response) {
  _idToken = response.credential;
  const claims = _decodeJwtPayload(response.credential);
  const user = { uid: claims.sub, email: claims.email, name: claims.name, photo: claims.picture };
  if (_authChangeCallback) _authChangeCallback(user);
}

google.accounts.id.initialize({
  client_id: GOOGLE_CLIENT_ID,
  callback: _handleCredential,
  auto_select: true,
});

// One Tap 提示——不保證一定會跳出來（瀏覽器封鎖第三方 cookie、使用者之前
// 關過好幾次都會讓 Google 端靜音這個提示），正式登入動線走的是
// sincoRenderGoogleButton() 畫出來的官方按鈕，這支只是備用觸發方式。
window.sincoSignIn = () => new Promise((resolve, reject) => {
  google.accounts.id.prompt((notification) => {
    if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
      reject(new Error("登入提示未顯示，請改點 Google 登入按鈕"));
    }
  });
});

window.sincoSignOut = () => {
  google.accounts.id.disableAutoSelect();
  _idToken = null;
  if (_authChangeCallback) _authChangeCallback(null);
  return Promise.resolve();
};

window.sincoGetIdToken = () => Promise.resolve(_idToken);

window.sincoOnAuthChanged = (callback) => {
  _authChangeCallback = callback;
};

// 畫出官方「Sign in with Google」按鈕——GIS 的登入彈窗只有點這顆真的由
// Google 產生的按鈕才會可靠地跳出來，自己刻的按鈕用程式碼轉發點擊事件並不
// 可靠（Google 端有防點擊劫持的限制）。
window.sincoRenderGoogleButton = (elementId) => {
  const el = document.getElementById(elementId);
  if (el) {
    google.accounts.id.renderButton(el, { theme: "outline", size: "medium", type: "standard", width: 180 });
  }
};
