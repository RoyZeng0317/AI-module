"""只讀照片／影片整理；不產生元件或連線真值，也不訓練模型。"""
from pathlib import Path
import csv, hashlib, json, math
import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEOS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
COLUMNS = ['asset_id', 'status', 'component_id', 'component_type', 'terminal_id',
           'endpoint_x', 'endpoint_y', 'board_id', 'hole_id', 'visibility', 'verification']
CONNECTION_COLUMNS = ['connection_id', 'asset_id', 'endpoint_a', 'endpoint_b',
                      'connection_kind', 'verification', 'evidence']
HOLE_COLUMNS = ['board_id', 'asset_id', 'hole_id', 'internal_net_id', 'verification', 'evidence']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_number(value):
    return value if math.isfinite(value) and value >= 0 else None


def image_info(path):
    with Image.open(path) as image:
        orientation = image.getexif().get(274)
        original_size = list(image.size)
        normalized = ImageOps.exif_transpose(image).convert('RGB')
        return normalized, {'stored_size': original_size, 'normalized_size': list(normalized.size),
                            'exif_orientation': orientation, 'stored_mode': image.mode}


def sample_video(path, output, source_id, count):
    capture = cv2.VideoCapture(str(path))
    frames, failures = [], []
    try:
        if not capture.isOpened():
            raise ValueError('無法開啟影片，請檢查 OpenCV 解碼器與檔案。')
        total = safe_number(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = safe_number(capture.get(cv2.CAP_PROP_FPS))
        total = int(total) if total else None
        fps = fps if fps and fps > 0 else None
        if not total:
            raise ValueError('影片沒有可用 frame count，無法可靠抽樣。')
        fractions = [.5] if count == 1 else [.1 + .8 * i / (count - 1) for i in range(count)]
        indices = list(dict.fromkeys(round((total - 1) * fraction) for fraction in fractions))
        for index in indices:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                failures.append({'requested_frame_index': index, 'error': '解碼器拒絕 frame seek'})
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                failures.append({'requested_frame_index': index, 'error': '讀取影格失敗'})
                continue
            next_position = safe_number(capture.get(cv2.CAP_PROP_POS_FRAMES))
            actual_index = int(round(next_position)) - 1 if next_position and next_position >= 1 else None
            reported_msec = safe_number(capture.get(cv2.CAP_PROP_POS_MSEC))
            asset_id = f'{source_id}_f{index:08d}'
            relative = f'frames/{asset_id}.jpg'
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.save(output / relative, quality=95)
            frames.append({'asset_id': asset_id, 'source_id': source_id, 'kind': 'sampled_frame',
                'path': relative, 'path_basis': 'output_relative', 'size': list(image.size),
                'requested_frame_index': index, 'decoder_frame_index': actual_index,
                'estimated_seconds_from_index': index / fps if fps else None,
                'decoder_reported_msec': reported_msec,
                'timestamp_basis': 'frame index / reported fps estimate; decoder positions are unverified, not exact PTS',
                'default_group_id': source_id, 'group_verified': False,
                'status': 'unreviewed', 'sha256': sha256(output / relative)})
        return {'frame_count_reported': total, 'fps_reported': fps,
                'duration_seconds_estimated': total / fps if fps else None,
                'width_reported': safe_number(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height_reported': safe_number(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'requested_samples': count, 'decoded_samples': len(frames),
                'sampling': '10% to 90% inclusive; one sample uses 50%; duplicate indices removed',
                'failures': failures}, frames
    finally:
        capture.release()


def make_contact_sheet(assets, output, source):
    errors = []
    columns, width, height = 4, 260, 220
    for start in range(0, len(assets), 32):
        page = assets[start:start + 32]
        sheet = Image.new('RGB', (columns * width, math.ceil(len(page) / columns) * height), '#202020')
        draw = ImageDraw.Draw(sheet)
        for i, asset in enumerate(page):
            path = output / asset['path'] if asset['path_basis'] == 'output_relative' else source / asset['path']
            x, y = (i % columns) * width, (i // columns) * height
            try:
                image, _ = image_info(path)
                thumbnail = ImageOps.contain(image, (width - 10, height - 35))
                sheet.paste(thumbnail, (x + (width-thumbnail.width)//2, y + 28))
            except (OSError, ValueError) as error:
                errors.append({'asset_id': asset['asset_id'], 'stage': 'contact_thumbnail', 'error': str(error)})
                draw.text((x + 5, y + 50), 'Thumbnail unavailable', fill='#ff8080')
            draw.text((x + 5, y + 8), asset['asset_id'], fill='white')
        sheet.save(output / f'contact_{start // 32 + 1:03d}.jpg', quality=90)
    return errors


def prepare_dataset(source, output, frames_per_video=3, hash_videos=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError('來源必須是存在的資料夾。')
    if type(frames_per_video) is not int or not 1 <= frames_per_video <= 32:
        raise ValueError('每部影片抽樣數必須為 1 至 32。')
    if not output.is_relative_to(ROOT):
        raise ValueError('輸出必須位於 AI-module 專案內。')
    if output == source or output.is_relative_to(source):
        raise ValueError('輸出不得位於唯讀來源資料夾內。')
    if output.exists():
        raise ValueError('輸出已存在，請使用新的資料夾。')
    paths = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMAGES | VIDEOS)
    if not paths:
        raise ValueError('來源沒有支援的照片或影片。')
    output.mkdir(parents=True)
    (output / 'frames').mkdir()
    manifest = {'version': 1, 'source_root': str(source), 'sources': [], 'assets': [], 'errors': [],
                'ground_truth': 'none; blank annotations mean unknown, never negative',
                'split_policy': 'Do not split until humans confirm circuit groups across files; frames from one video stay together.',
                'training_performed': False}
    for path in paths:
        relative = path.relative_to(source).as_posix()
        source_id = 's' + hashlib.sha256(relative.encode('utf-8')).hexdigest()[:12]
        kind = 'image' if path.suffix.lower() in IMAGES else 'video'
        record = {'source_id': source_id, 'absolute_path': str(path), 'relative_path': relative,
                  'kind': kind, 'bytes': None, 'sha256': None, 'hash_status': 'not_computed',
                  'default_group_id': source_id, 'group_verified': False,
                  'cross_file_circuit_group': None, 'status': 'unreviewed'}
        manifest['sources'].append(record)
        try:
            record['bytes'] = path.stat().st_size
            if kind == 'image' or hash_videos:
                record['sha256'] = sha256(path)
                record['hash_status'] = 'computed'
            else:
                record['hash_status'] = 'skipped_video_by_default'
            if kind == 'image':
                image, metadata = image_info(path)
                record.update(metadata)
                manifest['assets'].append({'asset_id': source_id + '_image', 'source_id': source_id,
                    'kind': 'image', 'path': relative, 'path_basis': 'source_relative',
                    'size': list(image.size), 'default_group_id': source_id,
                    'group_verified': False, 'status': 'unreviewed'})
            else:
                metadata, assets = sample_video(path, output, source_id, frames_per_video)
                record['video'] = metadata
                manifest['assets'].extend(assets)
                if metadata['failures']:
                    manifest['errors'].append({'source_id': source_id, 'error': '部分影格抽樣失敗',
                                               'details': metadata['failures']})
                if not assets:
                    record['status'] = 'error'
                    manifest['errors'].append({'source_id': source_id, 'error': '無可用抽樣影格'})
        except (OSError, ValueError, cv2.error) as error:
            record['status'] = 'error'
            manifest['errors'].append({'source_id': source_id, 'error': str(error)})
    with (output / 'annotations.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({'asset_id': asset['asset_id'], 'status': 'unreviewed'} for asset in manifest['assets'])
    with (output / 'groups.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_id', 'default_group_id', 'confirmed_circuit_group', 'verification'])
        writer.writeheader()
        writer.writerows({'source_id': item['source_id'], 'default_group_id': item['source_id']} for item in manifest['sources'])
    for filename, columns in [('connections.csv', CONNECTION_COLUMNS), ('board_holes.csv', HOLE_COLUMNS)]:
        with (output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            csv.DictWriter(stream, fieldnames=columns).writeheader()
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        manifest['errors'].extend(make_contact_sheet(manifest['assets'], output, source))
    except (OSError, ValueError) as error:
        manifest['errors'].append({'stage': 'contact_sheet', 'error': str(error)})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
