"""build_corpus.py — fetch a small, real starter text corpus for
transformer_chat.py's pretrain() from Chinese Wikipedia's public REST/Action
API (same data source tools.py's wikipedia_search() already calls — a plain
data API, not an AI model, so this stays inside Rule 06).

This is NOT a replacement for a real pretraining corpus (see
transformer_chat.py's "honest scope" docstring) — it's a few dozen articles,
maybe a few hundred KB of text, just enough to (a) exercise pretrain() on
something more real than synthetic placeholder sentences and (b) give a
first checkpoint some actual Chinese/English grammar and common-word
statistics to start from instead of nothing. A model good enough to
generalize confidently needs a corpus several orders of magnitude larger
than this; that is a separate, later step once you decide how much text to
gather (see CLAUDE.md to-do).

`variant=zh-tw` asks Wikipedia's built-in Simplified<->Traditional
converter for Taiwan-style Traditional Chinese output (電腦/軟體/網路, not
计算机/软件/网络), matching this project's Traditional Chinese convention.
`redirects=1` is required — without it, titles that are redirect pages
(e.g. "人工智慧" -> "人工智能") return the redirect NOTICE text instead of
the real article body.

Usage:
    python build_corpus.py --out ../data/corpus_zh_starter.txt
"""

import argparse
import time
from pathlib import Path

import requests

API_URL = "https://zh.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "sinco-AI-module/1.0 (local desktop app; no contact URL)"}
TIMEOUT = 15

# A small, deliberately diverse starter topic list (general knowledge +
# a few topics related to this project's own domain) -- diversity matters
# more than volume at this scale, since volume alone won't get anywhere
# near "real" corpus size regardless of how long this list is.
DEFAULT_TITLES = [
    "人工智慧", "機器學習", "類神經網路", "電腦程式設計", "Python",
    "網際網路", "微積分", "物理學", "數學", "天文學",
    "台灣", "台北市", "歷史", "地球", "音樂",
    "電影", "貓", "狗", "氣象學", "教育",
    "深度學習", "自然語言處理", "演算法", "資料庫", "半導體",
    "化學", "生物學", "太陽系", "相對論", "高雄市",
    "台灣歷史", "故宮博物院", "第二次世界大戰", "經濟學", "心理學",
    "哲學", "文學", "古典音樂", "籃球", "棒球",
    "智慧型手機", "資訊安全", "機器人學", "咖啡", "旅遊",
    # --- 2026-07-30 擴充：把語料量從 ~47萬字衝到 ~5倍(~237萬字)，
    # 主題盡量橫跨自然科學/數學/科技/歷史/地理/人文/運動/生活/醫學生物/動植物，
    # 求多樣性而非單一主題重複，理由同上方 DEFAULT_TITLES 原本的設計說明。
    "量子力學", "熱力學", "電磁學", "光學", "遺傳學",
    "演化論", "細胞", "DNA", "免疫系統", "地質學",
    "板塊構造論", "海洋學", "生態學", "環境保護", "氣候變遷",
    "核能", "能源", "太陽能", "元素週期表", "原子",
    "分子", "化學反應", "有機化學", "無機化學", "材料科學",
    "奈米科技", "統計學", "機率論", "幾何學", "代數",
    "三角函數", "線性代數", "數論", "拓撲學", "網頁開發",
    "JavaScript", "Java", "C語言", "作業系統", "Linux",
    "Microsoft Windows", "雲端運算", "大數據", "區塊鏈", "加密貨幣",
    "虛擬實境", "擴增實境", "物聯網", "5G", "積體電路",
    "電腦硬體", "電腦視覺", "語音辨識", "機器翻譯", "中國歷史",
    "世界歷史", "古埃及", "古羅馬", "古希臘", "文藝復興",
    "工業革命", "冷戰", "第一次世界大戰", "中華民國", "清朝",
    "明朝", "唐朝", "三國", "日本歷史", "美國歷史",
    "法國大革命", "亞洲", "歐洲", "非洲", "北美洲",
    "南美洲", "大洋洲", "南極洲", "太平洋", "大西洋",
    "喜馬拉雅山", "長江", "黃河", "日本", "韓國",
    "美國", "英國", "法國", "德國", "義大利",
    "印度", "中國", "新加坡", "香港", "澳門",
    "台中市", "台南市", "新竹市", "花蓮縣", "澎湖群島",
    "繪畫", "雕塑", "建築", "書法", "攝影",
    "戲劇", "舞蹈", "歌劇", "流行音樂", "搖滾樂",
    "爵士樂", "動漫", "漫畫", "電子遊戲", "小說",
    "詩", "神話", "佛教", "基督教", "伊斯蘭教",
    "道教", "儒家思想", "足球", "排球", "網球",
    "桌球", "羽毛球", "游泳", "田徑", "體操",
    "奧林匹克運動會", "世界盃足球賽", "高爾夫球", "拳擊", "柔道",
    "空手道", "滑雪", "馬拉松", "茶", "巧克力",
    "麵包", "壽司", "披薩", "紅酒", "啤酒",
    "汽車", "飛機", "火車", "腳踏車", "智慧型手錶",
    "相機", "電視", "廣播", "報紙", "圖書館",
    "醫院", "銀行", "股票市場", "保險", "房地產",
    "病毒", "細菌", "疫苗", "抗生素", "癌症",
    "心臟", "大腦", "神經系統", "消化系統", "呼吸系統",
    "骨骼", "肌肉", "血液", "荷爾蒙", "睡眠",
    "營養學", "鳥類", "魚類", "昆蟲", "爬蟲類",
    "兩棲類", "哺乳類", "恐龍", "大熊貓", "老虎",
    "獅子", "大象", "鯨魚", "鯊魚", "蝴蝶",
    "蜜蜂", "森林", "沙漠", "熱帶雨林", "珊瑚礁",
    # --- 2026-07-30 第二輪擴充：從 5 倍(262萬字)衝到 10 倍(~474萬字)，
    # 新增歷史人物/文學經典/科技公司/進階科學/世界城市地標/流行文化/飲食/
    # 運動/交通載具/現代議題/醫學/動植物/社會人文/音樂/節慶/更多國家與台灣
    # 主題/科學家發明，一樣求多樣性、避免跟前兩輪重複。
    "孔子", "老子", "牛頓", "愛因斯坦", "達爾文",
    "居禮夫人", "林肯", "拿破崙", "秦始皇", "毛澤東",
    "蔣中正", "孫中山", "莎士比亞", "達文西", "畢卡索",
    "貝多芬", "莫札特", "甘地", "邱吉爾", "亞里斯多德",
    "柏拉圖", "蘇格拉底", "紅樓夢", "西遊記", "三國演義",
    "水滸傳", "哈利波特", "魔戒", "唐詩", "宋詞",
    "論語", "道德經", "Google", "Apple", "Microsoft",
    "Facebook", "Amazon", "Tesla", "Nvidia", "Intel",
    "台積電", "鴻海", "黑洞", "宇宙大爆炸", "基因工程",
    "幹細胞", "量子電腦", "超導體", "雷射", "核融合",
    "暗物質", "弦理論", "紐約", "倫敦", "巴黎",
    "東京", "首爾", "雪梨", "莫斯科", "柏林",
    "羅馬", "開羅", "埃及金字塔", "萬里長城", "富士山",
    "阿爾卑斯山", "大峽谷", "尼加拉瀑布", "自由女神像", "艾菲爾鐵塔",
    "好萊塢", "迪士尼", "漫威", "星際大戰", "神奇寶貝",
    "任天堂", "電子競技", "火鍋", "拉麵", "牛肉麵",
    "珍珠奶茶", "威士忌", "壽喜燒", "泡菜", "F1賽車",
    "美式足球", "冰球", "板球", "鐵人三項", "攀岩",
    "高速公路", "捷運", "航空母艦", "潛水艇", "太空梭",
    "國際太空站", "高速鐵路", "智慧家庭", "無人機", "3D列印",
    "基因改造食品", "再生能源", "風力發電", "核電廠", "地震",
    "颱風", "火山", "糖尿病", "高血壓", "憂鬱症",
    "失眠", "過敏", "中醫", "針灸", "疫情",
    "企鵝", "無尾熊", "袋鼠", "長頸鹿", "河馬",
    "犀牛", "海豚", "水母", "螃蟹", "烏龜",
    "玫瑰", "櫻花", "竹子", "仙人掌", "松樹",
    "楓葉", "向日葵", "電磁波", "聲波", "溫室效應",
    "臭氧層", "生物多樣性", "微生物", "病毒學", "神經科學",
    "認知科學", "人類學", "社會學", "政治學", "法律",
    "憲法", "民主", "資本主義", "社會主義", "全球化",
    "嘻哈音樂", "電子音樂", "民謠", "藍調", "交響樂",
    "芭蕾舞", "農曆新年", "中秋節", "端午節", "聖誕節",
    "萬聖節", "感恩節", "加拿大", "澳洲", "巴西",
    "墨西哥", "俄羅斯", "西班牙", "荷蘭", "瑞士",
    "瑞典", "泰國", "越南", "馬來西亞", "印尼",
    "菲律賓", "台灣小吃", "夜市", "台灣原住民", "客家文化",
    "台灣高鐵", "阿里山", "日月潭", "墾丁", "尼古拉·特斯拉",
    "萊特兄弟", "電燈", "蒸汽機", "內燃機", "網際網路協定",
    "冰島", "埃及", "土耳其", "南非", "肯亞",
    "秘魯", "阿根廷", "紐西蘭", "芬蘭", "挪威",
]


def fetch_article(title: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        resp = requests.get(
            API_URL,
            params={
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": 1, "redirects": 1, "variant": "zh-tw", "titles": title,
            },
            headers=HEADERS, timeout=TIMEOUT,
        )
        if resp.status_code == 429:
            # anonymous requests get rate-limited fast without a pause
            # between calls -- back off and retry rather than give up.
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)
        if page is None or "extract" not in page:
            return None
        extract = page["extract"].strip()
        return extract or None
    return None


def build_corpus(titles: list[str], out_path: Path, delay: float = 1.5) -> dict[str, bool]:
    results: dict[str, bool] = {}
    chunks: list[str] = []
    for i, title in enumerate(titles):
        if i > 0:
            time.sleep(delay)  # be a polite anonymous API client, not just a retry-on-429 one
        try:
            extract = fetch_article(title)
        except requests.RequestException as exc:
            print(f"[skip] {title}: {exc}")
            results[title] = False
            continue
        if extract is None:
            print(f"[skip] {title}: no extract returned")
            results[title] = False
            continue
        chunks.append(extract)
        results[title] = True
        print(f"[ok]   {title}: {len(extract)} chars")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(chunks), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch a small starter text corpus from Chinese Wikipedia")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "corpus_zh_starter.txt")
    parser.add_argument("--titles", nargs="*", default=DEFAULT_TITLES)
    args = parser.parse_args()

    results = build_corpus(args.titles, args.out)
    ok = sum(results.values())
    print(f"\n{ok}/{len(results)} articles fetched -> {args.out}")


if __name__ == "__main__":
    main()
