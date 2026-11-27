const fs=require('fs'),path=require('path');
const {Document,Packer,Paragraph,TextRun,Header,Footer,PageNumber,AlignmentType,HeadingLevel,Table,TableRow,TableCell,WidthType,ImageRun}=require('C:/Users/roy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/docx');
const root=path.resolve(__dirname,'../..'),run=path.join(root,'tranning/speech_pilot_runs/alignment_v1');
const read=n=>JSON.parse(fs.readFileSync(path.join(run,n),'utf8'));
const h=read('history.json'),s=read('status.json'),rows=read('raw_compare.json').rows;
const font={ascii:'Times New Roman',hAnsi:'Times New Roman',eastAsia:'標楷體',cs:'Times New Roman'};
const r=(text,bold=false)=>new TextRun({text,font,size:24,bold,color:'000000'});
const p=text=>new Paragraph({alignment:AlignmentType.JUSTIFIED,spacing:{after:90,line:270},children:[r(text)]});
const heading=text=>new Paragraph({heading:HeadingLevel.HEADING_1,keepNext:true,spacing:{before:160,after:100},children:[r(text,true)]});
function table(data,widths){return new Table({width:{size:widths.reduce((a,b)=>a+b),type:WidthType.DXA},columnWidths:widths,rows:data.map((row,i)=>new TableRow({tableHeader:i===0,children:row.map((v,j)=>new TableCell({width:{size:widths[j],type:WidthType.DXA},children:[new Paragraph({spacing:{after:30,line:250},children:[r(String(v),i===0)]})]}))}))});}
const children=[
new Paragraph({heading:HeadingLevel.TITLE,alignment:AlignmentType.CENTER,keepNext:true,children:[r('語音識別對齊修復首輪訓練結果',true)]}),
p('2026年11月3日｜alignment_v1｜本機 RTX 4060 8GB｜Claude 編寫、Grok 審查、Codex 原樣套用與驗證'),
heading('一、結論：訓練完成，辨識品質未達標'),
p('本輪真實完成 10 epochs、640 次參數更新，訓練階段約 97.03 秒，整個程序約 104 秒；停止原因為 max_updates。沒有 OOM 或非有限數值停止，正式 speech_runs 的權重、字表、設定與歷史紀錄經前後雜湊核對均未變動。候選權重另存，沒有切換正式模型，也沒有自動啟動第二輪。'),
p('修復後基準的 train loss 由 6.219806 降到 3.980749，dev loss 由 6.807816 降到 6.595831；但 global CER 僅由 96.5657% 降到 96.1616%，少了 2 個編輯錯誤（478→476，參考共495字）。公開片段「用衛星與圖資盤點」從「我街」變成「民街」，仍然錯誤。這不是可靠聽寫能力的提升證據，候選不適合直接取代現有模型。'),
heading('二、訓練設定與計分母體'),
p('沿用專案自建 CRNN 舊權重與舊字表微調，不使用外部模型或雲端 AI API。seed 42，從原 train／val 清單固定抽 train 256、dev 64；batch 4、learning rate 0.0001、AdamW、dropout 0.2，BatchNorm 固定使用既有統計值，不做資料增強。最長音訊上限10秒，本輪沒有因超長或對齊不足而排除樣本。上限10 epochs／640 updates／600秒，於batch邊界檢查；本次達更新數上限，未碰到時間上限。'),
p('train loss 使用256筆、1995個參考字；dev loss僅使用字表可編碼的60筆、445個參考字。其餘4筆有未知字，不刪除標註，也不勉強計CTC loss，仍全部納入dev CER的64筆、495字。loss為CTC損失總和除以參考字數，CER為總編輯距離除以總參考字數。兩者母體不同，不能當作相同樣本上的兩個等價品質量尺；CER也不能直接換算為一般辨識準確率。'),
heading('三、每個 epoch 的實測紀錄'),
table([['Epoch','Train loss','Dev loss','Dev CER','Updates'],...h.map(x=>[x.epoch,x.train_loss.toFixed(6),x.val_loss.toFixed(6),(x.val_cer*100).toFixed(4)+'%',x.updates])],[1050,2200,2200,2250,2000]),
p('Epoch 0 是修復後完整音訊與packed前向、舊權重尚未更新的基準，不是舊裁切CLI整套流程的同一評估。舊history的740筆val CER約93%與本次抽樣64筆、完整音訊、global CER不可直接比較。第3輪dev loss最低6.369894；第6輪CER最低95.7576%。只有最終第10輪權重被保存，不能聲稱已保存或可直接取回第3／6輪最佳模型。'),
new Paragraph({alignment:AlignmentType.CENTER,children:[new ImageRun({data:fs.readFileSync(path.join(__dirname,'training_curves.png')),type:'png',transformation:{width:620,height:223}})]}),
heading('四、原始輸出對照'),
p('下表直接使用固定dev前8筆與公開回歸片段，兩次均用完整有效音訊與相同貪婪解碼；不使用remove_fillers、不交聊天模型補字。回歸片段從train/dev抽樣中排除，不作未見測試。這9筆最終輸出均非整句正確；它們只是小批診斷例，不能代替完整語言／口音測試。'),
table([['組別／參考逐字稿','舊權重原始輸出','第10輪原始輸出'],...rows.map(x=>[(x.set==='regression'?'公開回歸：':'Dev：')+x.reference,x.baseline_output||'（空白）',x.final_output||'（空白）'])],[4500,2600,2600]),
heading('五、過擬合、欠擬合與工程修復的解讀'),
p('train loss持續下降，dev loss在第3輪後整體回升，到第10輪仍高於最低點，呈現過擬合警訊；但這不能證明所有失敗都只因過擬合。train辨識正確率尚未單獨計算，兩者loss也不是可直接對照的不同任務標準。整體CER仍接近96%，短而錯誤的輸出顯示辨識尚未可用；單靠增加epochs不能合理保證改善，也不能僅憑loss數值就判定欠擬合的唯一根因。'),
p('本輪修掉固定1.6秒裁音卻保留全文標註、有效CTC長度、BiLSTM補零影響及未知字靜默刪除等流程問題；採逐筆有效CNN、packed LSTM，並檢查相鄰重複字所需CTC步數。這些工程修復有獨立測試，但本輪同時固定BatchNorm、取消增強及沿用舊權重，因此不是只改單一變因的因果消融。原始訓練可能已學到不良對齊；是否應重建聲學基準，必須另擬試驗，不能由本輪結果直接斷言。'),
p('manifest沒有speaker_id／session_id，原train／val未發現同路徑或相同全文逐字稿，不等於已隔離說話者、近重複音訊或錄音場次。這輪沒有獨立test，也沒有台灣台語／其他閩南語及中國大陸口音的分組實測；所有結果只描述本輪既有資料切分的華語診斷，不能外推到方言理解、語音合成或完整語音對話能力。'),
heading('六、執行時長、顯存與下一步'),
p('更新第2–6步平均約0.123秒／次，當時估計剩餘更新約78秒，外加評估；最終訓練階段約97秒。PyTorch峰值allocated顯存為642,600,960 bytes，約612.8 MiB；這不含所有CUDA保留記憶體、桌面或其他程序，不能當作整機顯存占用。本輪不涉及每分鐘USD5的對外扣款，該價格是另案的思考與回應時間計費草案。'),
p('下一步先核聽一小批音訊與逐字稿、按長度與字元覆蓋整理錯誤，再以固定小樣本檢驗模型能否學到完整短句，並比對舊權重初始化與從零聲學基準的必要性。補說話者來源資訊與獨立測試後，才討論一般化能力。這是後續提案，尚未執行，沒有自動重訓或用公開案例反覆調參。'),
heading('七、協作與資料來源'),
p('Claude交付speech_ctc_pilot_v2.zip，Grok先審完整v1原碼與測試，提出history每輪持久化及loss/CER母體兩項修訂，再審v2差異回覆PASS。本機14項工程測試通過後原樣套用兩檔，沒有由Codex代寫修復。工程測試使用合成音與假checkpoint，不當作真實準確度。兩方對本輪真實數據的結果審閱另存本工作目錄，來源是本機紀錄而非兩方另跑訓練。'),
p('直接證據（皆位於 C:/Users/roy/Documents/GitHub/AI-module/）：tranning/speech_pilot_runs/alignment_v1/history.json、status.json、data_manifest.json、raw_compare.json、train.log、pilot_model.pt；本報告表格與曲線由這些JSON產生。原始資料：tranning/commonvoice_zhTW/train.json、val.json及clips/。'),
p('修復與審查來源：tranning/speech_ctc_pilot.py、test_speech_ctc_pilot.py；team-run/speech-recognition/ALIGNMENT_REVIEW.md、ALIGNMENT_V1_PROPOSAL.md、formal_before_alignment_v1.json、public_regression_before.json。v2 runner SHA-256：25946415e2e483dce9bb9c21e0cfb5230ddc1111c08720d5d2ca700cac9250f8。本報告沒有引用外部資料集成績作為本專案成果。'),
];
const doc=new Document({styles:{default:{document:{run:{font,size:24,color:'000000'},paragraph:{spacing:{line:270}}}}},sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1000,bottom:1000,left:1000,right:1000,header:450,footer:450}}},headers:{default:new Header({children:[p('AI-module｜語音識別對齊修復首輪實測報告')]})},footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[r('Codex｜2026-11-03　｜　第 '),new TextRun({font,size:24,children:[PageNumber.CURRENT]}),r(' 頁')]})]})},children}]});
Packer.toBuffer(doc).then(b=>fs.writeFileSync(path.join(__dirname,'(Codex-語音識別首輪訓練結果).docx'),b));
