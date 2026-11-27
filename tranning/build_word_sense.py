"""Rebuild the small, manually authored demonstration fixtures."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/word_sense'
LABELS = ['apple_phone', 'apple_fruit', 'mouse_device', 'mouse_animal',
          'virus_computer', 'virus_biological', 'insufficient_context']
REPLIES = ['指的是蘋果品牌手機。', '指的是可食用的蘋果水果。', '指的是操作電腦的滑鼠。',
           '指的是會活動的鼠類動物。', '指的是感染電腦的惡意程式。', '指的是感染生物的病毒。',
           '資訊不足，請補充用途、行為或其他語境。']
# Every family belongs to exactly one split; rows use the first occurrence.
FAMILIES = {
    'purchase': [
        '我買蘋果是為了換掉舊手機。', '我買蘋果準備榨汁喝。',
        '我訂購一隻鼠，用來控制電腦游標。', '我領養一隻鼠，準備餵牠飼料。',
        '我要移除病毒，讓筆電能正常開機。', '我要檢驗病毒，確認病人感染原因。',
        '我想買蘋果。', '他帶來了一隻鼠。', '我想了解病毒。'],
    'care_instruction': [
        '蘋果充電時請使用合適的充電線。', '蘋果切片前請先洗乾淨果皮。',
        '鼠的按鍵失靈，請重新安裝驅動程式。', '鼠的籠子髒了，請清理木屑和飲水器。',
        '病毒刪除了硬碟檔案，請隔離受感染的程式。', '病毒透過飛沫傳播，請保持室內通風。',
        '請處理這個蘋果。', '請幫我看一下鼠。', '請處理這種病毒。'],
    'incident_report': [
        '市場旁發現的蘋果不是水果，螢幕還顯示來電。', '通訊行桌上的蘋果不是手機，咬開後果汁流了出來。',
        '寵物店裡的鼠沒有生命，只是接在收銀電腦上的指標裝置。', '電腦教室出現的鼠不是周邊設備，牠正啃食餅乾。',
        '醫院發現的病毒沒有感染病人，而是加密了掛號電腦的檔案。', '資訊中心談的病毒不是惡意軟體，而是使員工發燒的病原。',
        '有人在車站提到蘋果，沒有說用途。', '報告只寫發現鼠，沒有描述特徵。', '公告只說有病毒，沒有說感染對象。'],
    'comparison_question': [
        '水果攤老闆說這台蘋果拍照清楚；他談的是哪類東西？', '手機店員說這顆蘋果煮成果醬很香；他談的是哪類東西？',
        '獸醫詢問這個鼠是否支援藍牙和滾輪；談的是哪類東西？', '工程師問這隻鼠為什麼會生幼崽；談的是哪類東西？',
        '診所說病毒使伺服器跳出勒索畫面；感染的是什麼？', '軟體公司說病毒使呼吸道發炎；感染的是什麼？',
        '同事說蘋果很好，能確定是水果或手機嗎？', '只聽到鼠這個詞，能確定是動物或裝置嗎？', '沒有其他描述時，病毒是病原或惡意程式？']}


def main():
    ROOT.mkdir(exist_ok=True)
    for split, families in {'train': ['purchase', 'care_instruction'],
                            'val': ['incident_report'], 'test': ['comparison_question']}.items():
        rows = []
        for family in families:
            for i, text in enumerate(FAMILIES[family]):
                term = ['蘋果', '蘋果', '鼠', '鼠', '病毒', '病毒', '蘋果', '鼠', '病毒'][i]
                start = text.index(term)
                rows.append(dict(id=f'{family}_{i}', family=family, term=term, text=text,
                                 span=[start, start + len(term)], label=LABELS[min(i, 6)],
                                 reply=REPLIES[min(i, 6)]))
        if split == 'train':
            for i, text in enumerate(['我想買一台蘋果手機。', '我想買一斤蘋果。']):
                start = text.index('蘋果')
                rows.append(dict(id=f'purchase_exact_{i}', family='purchase', term='蘋果', text=text,
                                 span=[start, start + 2], label=LABELS[i], reply=REPLIES[i]))
        pairs = [dict(prompt=f'請依語境說明「{r["term"]}」的意思：{r["text"]}', reply=r['reply']) for r in rows]
        for name, data in [(split, rows), (f'sft_{split}', pairs)]:
            (ROOT / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
