"""從零方向分類、灰階特徵及有限文字照片編輯；沒有預訓練權重。"""
from pathlib import Path
import json, math, random, re
import cv2, numpy as np, torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torch import nn

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
DEFAULT_INSTRUCTION = '亮度 15；銳化 0.5'


def read_photo(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def gray_tensor(image, size=64):
    array = np.asarray(image.convert('L').resize((size, size)), dtype=np.float32).copy()
    return torch.from_numpy(array)[None] / 255


class OrientationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(24 * 16 * 16, 4))

    def forward(self, image):
        return self.net(image)


def split_photos(folder, seed=7):
    paths = sorted(p for p in Path(folder).rglob('*') if p.suffix.lower() in EXTENSIONS)
    # 同一內容的重複照片只留一份，避免原圖複本跨越訓練與驗證。
    import hashlib
    unique = {}
    for path in paths:
        photo = read_photo(path)
        identity = hashlib.sha256(str(photo.size).encode() + photo.tobytes()).hexdigest()
        unique.setdefault(identity, path)
    paths = list(unique.values())
    if len(paths) < 2:
        raise ValueError('方向訓練至少需要兩張不同、方向正確的照片。')
    random.Random(seed).shuffle(paths)
    count = max(1, round(len(paths) * .2))
    return paths[count:], paths[:count]


def orientation_samples(paths):
    images, labels = [], []
    for path in paths:
        photo = read_photo(path)
        for label in range(4):
            images.append(gray_tensor(photo.rotate(90 * label, expand=True)))
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def train_orientation(folder, checkpoint, epochs=10, batch_size=16):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs 與 batch_size 必須大於零。')
    train, valid = split_photos(folder)
    torch.manual_seed(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OrientationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, y = orientation_samples(train)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    vx, vy = orientation_samples(valid)
    with torch.no_grad():
        predictions = torch.cat([model(batch.to(device)).cpu() for batch in vx.split(batch_size)])
    metrics = {'validation_accuracy': float((predictions.argmax(1) == vy).float().mean()),
               'train_originals': [str(p) for p in train], 'validation_originals': [str(p) for p in valid],
               'epochs': epochs, 'note': '合成直角旋轉驗證，不能代表任意照片語意或小角度準確率'}
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kind': 'orientation-v1', 'state': model.cpu().state_dict(), 'metrics': metrics}, checkpoint)
    return metrics


def load_weights(path, kind, model):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('kind') != kind:
        raise ValueError('checkpoint 類型不正確。')
    model.load_state_dict(data['state'])
    return model.eval(), data


def choose_rotation(photo, checkpoint=None, threshold=.8):
    if not .25 < threshold <= 1:
        raise ValueError('信心門檻必須大於 0.25 且不超過 1。')
    if not checkpoint:
        return 0, {'status': 'untrained', 'reason': '未提供方向權重，保留原方向。'}
    model, data = load_weights(checkpoint, 'orientation-v1', OrientationNet())
    with torch.no_grad():
        scores = model(gray_tensor(photo)[None]).softmax(1)[0]
    label = int(scores.argmax())
    confidence = float(scores[label])
    angle = (-label * 90) % 360
    accepted = confidence >= threshold
    return angle if accepted else 0, {'status': 'accepted' if accepted else 'uncertain',
        'confidence': confidence, 'threshold': threshold, 'proposed_ccw_degrees': angle,
        'input_rotation_probabilities': scores.tolist(), 'training': data['metrics']}


def parse_instructions(text):
    actions = []
    for sentence in re.split('[;；\n]', text):
        sentence = sentence.strip()
        if not sentence:
            continue
        match = re.fullmatch(r'(亮度|對比|銳化)\s*([+-]?\d+(?:\.\d+)?)', sentence)
        if match:
            name, value = match[1], float(match[2])
            low, high = (-100, 100) if name == '亮度' else (0, 3)
            if not low <= value <= high:
                raise ValueError(f'{name} 範圍為 {low} 至 {high}。')
            actions.append((name, value))
            continue
        match = re.fullmatch(r'添加圓形\s+(\d+)\s+(\d+)\s+(\d+)\s+(#[0-9a-fA-F]{6})', sentence)
        if match:
            actions.append(('circle', int(match[1]), int(match[2]), int(match[3]), match[4]))
            continue
        match = re.fullmatch(r'添加文字\s+(\d+)\s+(\d+)\s+(\d+)\s+(#[0-9a-fA-F]{6})\s+(.+)', sentence)
        if match:
            if not 1 <= int(match[3]) <= 256:
                raise ValueError('文字大小必須為 1 至 256。')
            actions.append(('text', int(match[1]), int(match[2]), int(match[3]), match[4], match[5]))
            continue
        raise ValueError(f'不支援的指令：{sentence}。請參考 README 的精確語法。')
    return actions


def merge_luminance(photo, luminance):
    ycbcr = np.asarray(photo.convert('YCbCr')).copy()
    ycbcr[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8)
    return Image.fromarray(ycbcr, 'YCbCr').convert('RGB')


def edit_photo(photo, actions, font=None):
    if not actions or all(a[0] in {'circle', 'text'} for a in actions):
        gray = None
    else:
        gray = np.asarray(photo.convert('YCbCr'))[:, :, 0].astype(np.float32)
    # 先依輸入順序處理亮度類操作，再疊加明確指定的彩色標記。
    for action in actions:
        name, value = action[:2]
        if name == '亮度':
            gray = np.clip(gray + value, 0, 255)
        elif name == '對比':
            gray = np.clip((gray - 127.5) * value + 127.5, 0, 255)
        elif name == '銳化':
            gray = np.clip(gray + value * (gray - cv2.GaussianBlur(gray, (0, 0), 1)), 0, 255)
    result = photo.copy() if gray is None else merge_luminance(photo, gray)
    draw = ImageDraw.Draw(result)
    for action in actions:
        if action[0] not in {'circle', 'text'}:
            continue
        _, x, y, size, color, *content = action
        if x >= photo.width or y >= photo.height:
            raise ValueError('添加座標必須位於旋轉後照片內。')
        if action[0] == 'circle':
            if size < 1:
                raise ValueError('圓形半徑必須大於零。')
            draw.ellipse((x-size, y-size, x+size, y+size), fill=color)
        else:
            if not font:
                raise ValueError('添加文字需要 --font 指定支援該文字的字型檔。')
            draw.text((x, y), content[0], fill=color, font=ImageFont.truetype(str(font), size))
    return result


class GrayTextEditor(nn.Module):
    """字元平均嵌入 + 小型灰階殘差 CNN；需真實成對資料自行訓練。"""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.net = nn.Sequential(nn.Conv2d(17, 24, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 1, 3, padding=1))

    def forward(self, image, tokens):
        mask = (tokens != 0)[..., None]
        condition = (self.embedding(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1)
        condition = condition[..., None, None].expand(-1, -1, *image.shape[-2:])
        return (image + .5 * self.net(torch.cat((image, condition), 1)).tanh()).clamp(0, 1)


def encode_text(text, vocab):
    if not text.strip() or len(text) > 256:
        raise ValueError('學習式指令長度必須為 1 至 256 字元。')
    unknown = set(text) - vocab.keys()
    if unknown:
        raise ValueError('指令包含訓練詞表沒有的字元：' + ''.join(sorted(unknown)))
    return torch.tensor([[vocab[c] for c in text]])


def train_editor(folder, checkpoint, epochs=10):
    if epochs < 1:
        raise ValueError('epochs 必須大於零。')
    pairs = []
    for directory in sorted(Path(folder).iterdir()):
        if directory.is_dir() and (directory / 'instruction.txt').is_file():
            source, target = read_photo(directory / 'source.png'), read_photo(directory / 'target.png')
            if source.size != target.size:
                raise ValueError('成對來源與目標尺寸必須相同。')
            text = (directory / 'instruction.txt').read_text(encoding='utf-8').strip()
            if not text or len(text) > 256:
                raise ValueError('配對指令必須為 1 至 256 字元。')
            pairs.append((source, target, text))
    if not pairs:
        raise ValueError('找不到 source.png、target.png、instruction.txt 配對。')
    vocab = {c: i + 1 for i, c in enumerate(sorted(set(''.join(p[2] for p in pairs))))}
    torch.manual_seed(7)
    model = GrayTextEditor(len(vocab) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(epochs):
        for source, target, text in pairs:
            optimizer.zero_grad()
            loss = nn.functional.l1_loss(model(gray_tensor(source)[None], encode_text(text, vocab)), gray_tensor(target)[None])
            loss.backward()
            optimizer.step()
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kind': 'gray-editor-v1', 'state': model.state_dict(), 'vocab': vocab,
                'metrics': {'pairs': len(pairs), 'training_l1': float(loss.detach()), 'validation': '未提供獨立驗證集'}}, checkpoint)
    return {'pairs': len(pairs), 'training_l1': float(loss.detach())}


def learned_edit(photo, text, checkpoint):
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if data.get('kind') != 'gray-editor-v1':
        raise ValueError('需要灰階文字編輯 checkpoint。')
    model = GrayTextEditor(len(data['vocab']) + 1)
    model.load_state_dict(data['state'])
    model.eval()
    with torch.no_grad():
        prediction = model(gray_tensor(photo)[None], encode_text(text, data['vocab']))[0, 0].numpy()
    original_small = np.asarray(photo.convert('L').resize((64, 64)), dtype=np.float32) / 255
    residual = cv2.resize(prediction - original_small, photo.size, interpolation=cv2.INTER_LINEAR)
    original = np.asarray(photo.convert('YCbCr'))[:, :, 0].astype(np.float32)
    return merge_luminance(photo, original + residual * 255)


def extract_edges(gray, mode='photo', low=60, high=160):
    """電路圖只做輕微去噪；保留線寬的二值圖另存，不自動補線。"""
    if mode not in {'photo', 'circuit', 'breadboard'}:
        raise ValueError('mode 必須為 photo、circuit 或 breadboard。')
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
        raise ValueError('邊緣門檻需為有限數字，且 0 <= low < high。')
    prepared = cv2.GaussianBlur(gray, (3, 3), 0) if mode == 'circuit' else gray
    if mode == 'breadboard':
        prepared = cv2.bilateralFilter(gray, 5, 35, 35)
    edges = cv2.Canny(prepared, low, high)
    ink = None
    if mode == 'circuit':
        # 白底深色線的前景遮罩；不做膨脹／閉運算，以免接上原本斷開的線。
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return edges, ink


def breadboard_color_features(rgb, edges):
    """僅擷取飽和色區域，不代表導線／電阻辨識；黑白物件仍保留在原圖。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 60, 40), (179, 255, 255))
    colors = cv2.bitwise_and(rgb, rgb, mask=mask)
    overlay = rgb.copy()
    overlay[edges > 0] = (0, 255, 0)
    return mask, colors, overlay


def run_demo(source, output, instruction=None,
             checkpoint=None, threshold=.8, angle=None, font=None, editor_checkpoint=None,
             mode='photo', edge_low=60, edge_high=160):
    source, output = Path(source).resolve(), Path(output).resolve()
    # 輸出採新目錄，既有內容與原圖一律不覆寫。
    if output.exists():
        raise ValueError('輸出資料夾已存在，請指定新的資料夾。')
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION if mode == 'photo' else ''
    actions = None if editor_checkpoint else parse_instructions(instruction)
    photo = read_photo(source)
    if angle is not None:
        if not math.isfinite(angle):
            raise ValueError('旋轉角度必須是有限數字。')
        rotation, decision = angle, {'status': 'manual', 'ccw_degrees': angle}
    else:
        rotation, decision = choose_rotation(photo, checkpoint, threshold)
    rotated = photo.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    result = learned_edit(rotated, instruction, editor_checkpoint) if editor_checkpoint else edit_photo(rotated, actions, font)
    gray = np.asarray(rotated.convert('L'))
    edges, ink = extract_edges(gray, mode, edge_low, edge_high)
    corners = cv2.goodFeaturesToTrack(gray, 150, .02, 8)
    features = rotated.copy()
    draw = ImageDraw.Draw(features)
    for x, y in ([] if corners is None else corners[:, 0]):
        draw.ellipse((x-3, y-3, x+3, y+3), outline='#00ff00', width=2)
    output.mkdir(parents=True)
    photo.save(output / '01_original.png')
    Image.fromarray(gray).save(output / '02_gray.png')
    Image.fromarray(edges).save(output / '03_edges.png')
    features.save(output / '04_features.png')
    rotated.save(output / '05_rotated.png')
    result.save(output / '06_edited_color.png')
    if ink is not None:
        Image.fromarray(ink).save(output / '07_circuit_ink.png')
    color_pixels = None
    if mode == 'breadboard':
        mask, colors, overlay = breadboard_color_features(np.asarray(rotated), edges)
        Image.fromarray(mask).save(output / '07_color_mask.png')
        Image.fromarray(colors).save(output / '08_color_regions.png')
        Image.fromarray(overlay).save(output / '09_edge_overlay.png')
        color_pixels = int(np.count_nonzero(mask))
    panels = [('Original', photo), ('Gray luminance', Image.fromarray(gray)),
              ('Handcrafted edges', Image.fromarray(edges)), ('Handcrafted corners', features),
              ('Rotated / abstained', rotated), ('Edited color', result)]
    if mode == 'breadboard':
        panels[3] = ('Color regions (not wire detection)', Image.fromarray(colors))
        panels[4] = ('Original color / rotated', rotated)
        panels[5] = ('Edges on original color', Image.fromarray(overlay))
    sheet = Image.new('RGB', (960, 540), '#202020')
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x, y = (index % 3) * 320, (index // 3) * 270
        thumbnail = ImageOps.contain(panel.convert('RGB'), (310, 235))
        sheet.paste(thumbnail, (x + (320-thumbnail.width)//2, y+28))
        sheet_draw.text((x+10, y+8), label, fill='white')
    sheet.save(output / 'preview.jpg')
    for angle_candidate in (0, 90, 180, 270):
        photo.rotate(angle_candidate, expand=True).save(output / f'candidate_ccw_{angle_candidate}.png')
    report = {'source': str(source), 'mode': mode, 'rotation': decision, 'applied_ccw_degrees': rotation,
              'instruction': instruction, 'editor': 'trained_gray_residual' if editor_checkpoint else 'strict_operations',
              'features': {'edge_pixels': int(np.count_nonzero(edges)), 'corners': 0 if corners is None else len(corners),
                           'edge_thresholds': [edge_low, edge_high],
                           'color_region_pixels': color_pixels,
                           'circuit_ink_pixels': None if ink is None else int(np.count_nonzero(ink))},
              'limitations': '邊緣與角點不是語意理解；保留來源色度合成。學習式編輯只預測灰階低解析殘差，未保證任意文字編輯。'}
    if mode == 'breadboard':
        report['breadboard_analysis'] = {'stage': 'preprocessing_only',
            'component_detection': 'not_implemented', 'connection_reasoning': 'not_implemented',
            'note': '彩色區域包含線材、LED、色環及其他物體，不是元件遮罩；黑白線不會出現在飽和色遮罩，請同時參考完整彩圖與邊緣。'}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
