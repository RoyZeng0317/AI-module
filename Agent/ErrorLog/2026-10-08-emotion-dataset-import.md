# 情緒資料匯入檢查（2026-10-08）

1. CPED 的 comfort 標籤不能保證回覆同理：實際範例包含責備、敷衍與依賴劇情上下文的台詞。處理：保留作情緒辨識，不自動當作安慰回覆目標。
2. DailyDialog 論文舊官網已停放，舊 Hugging Face datasets 程式路徑亦不可取得。處理：用 ConvLab 發布的 data.zip，並核對原 DailyDialog dataset card 的 CC BY-NC-SA 4.0；兩者版本分別固定。
3. 來源含重複文字，不能把每個來源各自切分後直接接在一起。處理：去重優先 test > val > train，來源對話群組跨 split 直接報錯。實際完全相同正規化文字及對話 ID 的跨 split 重疊皆為 0。
4. CPED／GoEmotions 的類別／多標籤不能直接當成原 action.py 的強度分數。處理：標籤保持原義、分類資料獨立存放，明確紀錄另需分類 loader／head；沒有偽造 0~5 強度標籤。

新回覆集已能被既有 ChatSFTDataset／DataLoader 載入，無 512-token 超長句對。匯入測試10項通過。尚未執行模型微調或宣稱真實情緒表現改善。
