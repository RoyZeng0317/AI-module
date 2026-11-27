"""麵包板照片／影片唯讀匯入 CLI。"""
from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tranning.breadboard_dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description='只讀照片與影片，產生待人工標註的資料清單')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames-per-video', type=int, default=3)
    parser.add_argument('--hash-videos', action='store_true', help='額外串流計算大型影片 SHA256')
    args = parser.parse_args()
    try:
        manifest = prepare_dataset(args.source, args.output, args.frames_per_video, args.hash_videos)
        print(f"來源 {len(manifest['sources'])}，照片／影格 {len(manifest['assets'])}，錯誤 {len(manifest['errors'])}")
        if manifest['errors']:
            parser.exit(1, '部分來源無法處理，請查看輸出 manifest.json 的 errors。\n')
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f'錯誤：{error}\n')


if __name__ == '__main__':
    main()
