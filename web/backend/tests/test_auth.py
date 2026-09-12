"""auth.py 單元測試——只測 header 解析與失敗時的 fallback 行為，不真的打
Google 網路請求（verify_oauth2_token 本身是 google-auth 套件已驗證過的
第三方程式碼，這裡只驗證 auth.py 自己包的那層邏輯）。
"""

from unittest.mock import patch

from web.backend.auth import verify_id_token


def test_verify_id_token_returns_none_without_header():
    assert verify_id_token(None) is None
    assert verify_id_token("") is None


def test_verify_id_token_returns_none_for_non_bearer_header():
    assert verify_id_token("Basic xyz") is None


def test_verify_id_token_returns_claims_on_success():
    with patch("web.backend.auth.id_token.verify_oauth2_token", return_value={"sub": "uid-1"}):
        claims = verify_id_token("Bearer real-token")
    assert claims == {"sub": "uid-1"}


def test_verify_id_token_returns_none_when_verification_fails():
    with patch("web.backend.auth.id_token.verify_oauth2_token", side_effect=ValueError("bad token")):
        assert verify_id_token("Bearer fake-token") is None
