import argparse, sys, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tranning.photo_edit import extract_edges, breadboard_color_features


def main():
    parser = argparse.ArgumentParser(description='照片／電路圖／麵包板實拍的三組邊緣門檻比較')
    parser.add_argument('--input', default='lib/opencv2/penguins.jpg')
    parser.add_argument('--mode', choices=['photo', 'circuit', 'breadboard'], default='photo')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    image = cv2.imread(args.input)
    if image is None:
        parser.error('讀不到圖片，請確認輸入路徑。電路圖需先匯出 PNG／JPEG 等圖片。')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 照片先減少紋理雜訊；電路模式由 extract_edges 使用較輕的 3x3 去噪。
    prepared = cv2.GaussianBlur(gray, (5, 5), 0) if args.mode == 'photo' else gray
    output = args.output or Path(f'output/{args.mode}_edges_practice')
    if output.exists():
        parser.error('輸出資料夾已存在，請用 --output 指定新資料夾。')
    output.mkdir(parents=True)

    def save(name, pixels):
        if not cv2.imwrite(str(output / name), pixels):
            raise OSError(f'無法儲存圖片：{name}')

    save('gray.png', gray)
    if args.mode == 'breadboard':
        save('original_color.png', image)
    for low, high in [(30, 90), (60, 160), (100, 220)]:
        edges, ink = extract_edges(prepared, args.mode, low, high)
        save(f'edges_{low}_{high}.png', edges)
        if ink is not None and low == 60:
            save('circuit_ink.png', ink)
        if args.mode == 'breadboard':
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask, colors, overlay = breadboard_color_features(rgb, edges)
            save(f'edge_overlay_{low}_{high}.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if low == 60:
                save('color_mask.png', mask)
                save('color_regions.png', cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))
        print(f'{low}/{high}: {output / f"edges_{low}_{high}.png"}')


if __name__ == '__main__':
    main()
