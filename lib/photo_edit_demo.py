"""獨立照片示範前端；訓練與影像邏輯位於 tranning.photo_edit。"""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tranning.photo_edit import run_demo, train_orientation, train_editor


def main():
    parser = argparse.ArgumentParser(description='自行訓練方向／灰階文字編輯與照片示範')
    parser.add_argument('--input', default=str(Path(__file__).resolve().parent / 'opencv2/penguins.jpg'))
    parser.add_argument('--output', default='output/photo_edit_demo')
    parser.add_argument('--instruction', help='電路／麵包板預設不修改；照片預設亮度 15、銳化 0.5')
    parser.add_argument('--mode', choices=['photo', 'circuit', 'breadboard'], default='photo')
    parser.add_argument('--edge-low', type=float, default=60)
    parser.add_argument('--edge-high', type=float, default=160)
    parser.add_argument('--orientation-checkpoint')
    parser.add_argument('--editor-checkpoint')
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--angle', type=float, help='逆時針角度；明確指定優先於模型')
    parser.add_argument('--font', help='添加文字時使用的字型檔')
    training = parser.add_mutually_exclusive_group()
    training.add_argument('--train-orientation', metavar='UPRIGHT_PHOTO_FOLDER')
    training.add_argument('--train-editor', metavar='PAIR_FOLDER')
    parser.add_argument('--save', help='訓練權重輸出路徑（不允許覆寫）')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    try:
        if args.train_orientation or args.train_editor:
            if not args.save or Path(args.save).exists():
                parser.error('訓練需 --save 指定不存在的權重路徑。')
            train = train_orientation if args.train_orientation else train_editor
            report = train(args.train_orientation or args.train_editor, args.save, args.epochs)
        else:
            report = run_demo(args.input, args.output, args.instruction, args.orientation_checkpoint,
                              args.threshold, args.angle, args.font, args.editor_checkpoint,
                              mode=args.mode, edge_low=args.edge_low, edge_high=args.edge_high)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f'錯誤：{error}\n')


if __name__ == '__main__':
    main()
