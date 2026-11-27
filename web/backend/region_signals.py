"""內部地區訊號；非國籍、真實所在地證明或帳戶處置器。"""
import ipaddress, json
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "data" / "dbip-country-lite.mmdb"
RULES = json.loads((BASE_DIR / "region_rules.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _reader(path, modified):
    import maxminddb
    return maxminddb.open_database(path)


def lookup_country(address, database_path=None):
    """只查本機MMDB；缺庫、私有位址或查詢失敗皆為未知。"""
    try:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            return None
        path = Path(database_path) if database_path is not None else DATABASE_PATH
        record = _reader(str(path), path.stat().st_mtime_ns).get(str(ip)) or {}
        country = record.get("country", {}).get("iso_code")
        return country if isinstance(country, str) and len(country) == 2 else None
    except (ImportError, OSError, ValueError, TypeError, RuntimeError):
        return None


def assess_region(message, client_timezone=None, client_ip=None):
    """分開保留訊號，不把語言或UTC+8當成中國大陸確認依據。"""
    timezone = client_timezone.strip() if client_timezone else None
    beijing = timezone in RULES["beijing_timezones"]
    country = lookup_country(client_ip)
    if country == "MY":
        status = "malaysia_timezone_conflict" if beijing else "malaysia_ip"
    elif country == "CN":
        status = "cn_timezone_and_ip" if beijing else "cn_ip_only"
    elif country:
        status = "other_country_timezone_conflict" if beijing else "other_country_ip"
    else:
        status = "beijing_timezone_only" if beijing else "unknown"
    return {
        "status": status,
        "beijing_timezone": beijing,
        "ip_country": country,
        "text_signals": {
            # 有限代表字啟發式，並非完整簡繁分類，也不計為地區證據。
            "simplified_sample": any(char in message for char in RULES["simplified_sample_chars"]),
            # 標點統一為一類；只列不同碼位，不依出現次數或種類累加地區權重。
            "punctuation": sorted({f"U+{ord(char):04X}" for char in RULES["punctuation_chars"] if char in message}),
            "phrases": [phrase for phrase in RULES["phrases"] if phrase in message],
        },
        "action": "observe_only",
    }
