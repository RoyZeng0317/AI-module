"""取得本輪公開國別資料；只下載資料庫，不傳送使用者 IP。"""
import gzip, hashlib, json, urllib.request
from pathlib import Path
import maxminddb

root = Path(__file__).resolve().parents[2]
target = root / "web/backend/data/dbip-country-lite.mmdb"
url = "https://download.db-ip.com/free/dbip-country-lite-2026-10.mmdb.gz"
request = urllib.request.Request(url, headers={"User-Agent": "AI-module/1.0 (local geolocation data setup)",
                                              "Referer": "https://db-ip.com/db/download/ip-to-country-lite"})
with urllib.request.urlopen(request, timeout=60) as response:
    compressed = response.read(20_000_001)
if len(compressed) > 20_000_000:
    raise ValueError("資料下載超過預期上限")
data = gzip.decompress(compressed)
if len(data) > 64_000_000:
    raise ValueError("解壓資料超過預期上限")
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(".download")
temporary.write_bytes(data)
with maxminddb.open_database(str(temporary)) as reader:
    metadata = reader.metadata()
    samples = {ip: reader.get(ip).get("country", {}).get("iso_code")
               for ip in ("114.114.114.114", "1.9.1.1", "168.95.1.1")}
temporary.replace(target)
record = {"source": url, "license": "CC BY 4.0", "attribution": "IP Geolocation by DB-IP (https://db-ip.com)",
          "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
          "database_type": metadata.database_type, "build_epoch": metadata.build_epoch, "samples": samples}
Path(__file__).with_name("database-download.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
