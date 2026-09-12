"""驗證前端 Google Identity Services（Sign In With Google）發出的 ID token。

只用 Google 公開憑證驗證簽章/audience/expiry（`google.oauth2.id_token`），
不需要 service account 金鑰、不需要 Firebase 專案，不會動到 .env（Rule 05）。
GOOGLE_CLIENT_ID 是 Google Cloud Console 申請的 OAuth 用戶端 ID（Web
application 類型），必須跟兩份 google-auth-init.js
（web/frontend/src/components/、web/admin/frontend/src/components/）裡寫的
Client ID 完全一致——那邊是 GIS 初始化用的 client_id，這邊是驗證 token 用的
audience，同一組 ID 對不上就會驗證失敗。
"""

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

GOOGLE_CLIENT_ID = "1074357742383-tm3gopumph7ndvurmlgfdtm9bsncd9hp.apps.googleusercontent.com"

_request = google_requests.Request()


def verify_id_token(authorization_header: str | None) -> dict | None:
    """驗證成功回傳 Google claims（含 "sub" = uid），失敗或沒帶 header 回傳 None
    ——呼叫端據此 fallback 回舊的 IP 辨識方式，不會讓沒登入的舊呼叫端壞掉。"""
    if not authorization_header or not authorization_header.startswith("Bearer "):
        return None
    token = authorization_header.removeprefix("Bearer ").strip()
    if not token:
        return None
    try:
        return id_token.verify_oauth2_token(token, _request, audience=GOOGLE_CLIENT_ID)
    except ValueError:
        return None
