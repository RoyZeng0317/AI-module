from pathlib import Path
import csv, json
import cv2, numpy as np, pytest
from PIL import Image
from tranning.breadboard_dataset import prepare_dataset, sha256


def test_photo_intake_annotations_and_source_immutable(tmp_path):
    source = tmp_path / '電路原圖'
    source.mkdir()
    photo = source / '實拍.jpg'
    image = Image.new('RGB', (60, 40), 'green')
    exif = image.getexif()
    exif[274] = 6
    image.save(photo, exif=exif)
    before = sha256(photo)
    output = tmp_path / 'prepared'
    manifest = prepare_dataset(source, output)
    assert sha256(photo) == before
    record = manifest['sources'][0]
    assert record['sha256'] == before and record['normalized_size'] == [40, 60]
    assert record['exif_orientation'] == 6
    assert record['group_verified'] is False
    assert manifest['training_performed'] is False and not manifest['errors']
    assert manifest['assets'][0]['path_basis'] == 'source_relative'
    assert not list((output / 'frames').iterdir())
    with (output / 'annotations.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['status'] == 'unreviewed'
    assert all(rows[0][field] == '' for field in rows[0] if field not in {'asset_id', 'status'})
    assert 'unknown' in manifest['ground_truth']
    templates = {'connections.csv': ['connection_id', 'asset_id', 'endpoint_a', 'endpoint_b',
                                     'connection_kind', 'verification', 'evidence'],
                 'board_holes.csv': ['board_id', 'asset_id', 'hole_id', 'internal_net_id', 'verification', 'evidence']}
    for filename, fields in templates.items():
        with (output / filename).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == fields and list(reader) == []
    assert (output / 'contact_001.jpg').is_file()
    assert len(list(source.iterdir())) == 1
    assert json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['sources'][0]['absolute_path'] == str(photo.resolve())


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (48, 32))
    if not writer.isOpened():
        pytest.skip('此環境無 MJPG VideoWriter，影片測試無法執行')
    try:
        for index in range(20):
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            frame[:, :, 1] = index * 10
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    ok, _ = capture.read()
    capture.release()
    if not ok:
        pytest.skip('此環境無 MJPG 解碼器')


def test_video_sampling_integrity_and_grouping(tmp_path):
    source = tmp_path / '影片'
    source.mkdir()
    video = source / '掃描.avi'
    make_video(video)
    before = sha256(video)
    output = tmp_path / 'video_prepared'
    manifest = prepare_dataset(source, output, hash_videos=True)
    assert not manifest['errors'] and sha256(video) == before
    record = manifest['sources'][0]
    assert record['sha256'] == before
    assert record['video']['frame_count_reported'] == 20
    assert record['video']['fps_reported'] == pytest.approx(10)
    assert len(manifest['assets']) == 3
    assert [a['requested_frame_index'] for a in manifest['assets']] == [2, 10, 17]
    assert len({a['default_group_id'] for a in manifest['assets']}) == 1
    for asset in manifest['assets']:
        assert asset['group_verified'] is False and asset['status'] == 'unreviewed'
        assert 'estimate' in asset['timestamp_basis']
        assert Image.open(output / asset['path']).size == (48, 32)
        assert sha256(output / asset['path']) == asset['sha256']
    assert len(list(source.iterdir())) == 1


def test_invalid_empty_overwrite_and_corrupt_sources(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='沒有支援'):
        prepare_dataset(empty, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()
    with pytest.raises(ValueError, match='來源'):
        prepare_dataset(tmp_path / 'missing', tmp_path / 'unused')
    with pytest.raises(ValueError, match='抽樣數'):
        prepare_dataset(empty, tmp_path / 'unused', frames_per_video=0)
    with pytest.raises(ValueError, match='唯讀'):
        prepare_dataset(empty, empty / 'nested')
    (empty / 'bad.png').write_bytes(b'not an image')
    output = tmp_path / 'corrupt'
    manifest = prepare_dataset(empty, output)
    assert len(manifest['errors']) == 1 and not manifest['assets']
    assert manifest['sources'][0]['status'] == 'error'
    with pytest.raises(ValueError, match='已存在'):
        prepare_dataset(empty, output)

@pytest.mark.parametrize('count', [3.5, True])
def test_sampling_requires_integer_before_output_creation(tmp_path, count):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    with pytest.raises(ValueError, match='抽樣數'):
        prepare_dataset(source, output, count)
    assert not output.exists()


def test_unreadable_source_is_recorded_and_other_sources_continue(tmp_path, monkeypatch):
    import tranning.breadboard_dataset as module
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    for name in ['a_locked.png', 'b_readable.png']:
        Image.new('RGB', (12, 16), 'blue').save(source / name)
    original = module.sha256
    def fail_one(path):
        if Path(path).name == 'a_locked.png':
            raise PermissionError('synthetic permission denied')
        return original(path)
    monkeypatch.setattr(module, 'sha256', fail_one)
    manifest = prepare_dataset(source, output)
    assert len(manifest['errors']) == 1 and len(manifest['assets']) == 1
    failed = manifest['sources'][0]
    assert failed['status'] == 'error' and failed['hash_status'] == 'not_computed'
    assert failed['sha256'] is None
    saved = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['errors'] == manifest['errors']


def test_thumbnail_failure_is_explicit_and_manifest_persisted(tmp_path, monkeypatch):
    import tranning.breadboard_dataset as module
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    Image.new('RGB', (12, 16), 'blue').save(source / 'photo.png')
    original = module.image_info
    calls = 0
    def fail_thumbnail(path):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise FileNotFoundError('synthetic disappeared after processing')
        return original(path)
    monkeypatch.setattr(module, 'image_info', fail_thumbnail)
    manifest = prepare_dataset(source, output)
    assert manifest['errors'][0]['stage'] == 'contact_thumbnail'
    assert (output / 'contact_001.jpg').is_file()
    saved = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['errors'] == manifest['errors']
