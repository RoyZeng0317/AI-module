"""本機時區與 Unicode 查碼練習；不傳送或儲存輸入內容。"""
import argparse, unicodedata
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="台灣 中国 😀 e\u0301", help="查詢文字（最多顯示 200 個碼位）")
    text = parser.parse_args().text
    utc = datetime.now(timezone.utc)
    local = utc.astimezone()
    minutes = int(local.utcoffset().total_seconds() / 60)
    hours, remainder = divmod(abs(minutes), 60)
    print(f"UTC：{utc.isoformat()}\n本地時間：{local.isoformat()}")
    print(f"系統時區標籤：{local.tzname()}（不保證是 IANA 名稱）")
    print(f"UTC 時差：UTC{'+' if minutes >= 0 else '-'}{hours:02}:{remainder:02}")
    print("時區是裝置設定；Unicode 是文字編碼，皆無法據此判定國籍或實際所在地。")
    print("每列為一個碼位；結合字與 emoji 可能含多個碼位。")
    for index, char in enumerate(text[:200], 1):
        # repr 讓空白、換行與控制字元不會改變輸出排列。
        print(f"{index}. {ascii(char)} U+{ord(char):04X} {unicodedata.name(char, '未命名')} [{unicodedata.category(char)}]")
    if len(text) > 200:
        print("輸入超過 200 個碼位，僅顯示前 200 個。")


if __name__ == "__main__":
    main()
