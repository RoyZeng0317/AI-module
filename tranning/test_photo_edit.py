from pathlib import Path
import hashlib
import numpy as np, pytest, torch
from PIL import Image
from tranning.photo_edit import (parse_instructions, edit_photo, run_demo, choose_rotation,
    split_photos, train_orientation, train_editor, learned_edit)

torch.set_num_threads(1)


def test_strict_commands_and_color():
    photo = Image.new('RGB', (80, 50), (90, 100, 110))
    result = edit_photo(photo, parse_instructions('亮度 10；添加圓形 20 20 5 #ff0000'))
    assert result.getpixel((20, 20)) == (255, 0, 0)
    assert result.getpixel((60, 40))[0] > photo.getpixel((60, 40))[0]
    for text in ['不要變亮', '亮度 20 其他', '移除人物', '亮度 101']:
        with pytest.raises(ValueError):
            parse_instructions(text)


def test_demo_preserves_source_and_abstains(tmp_path):
    source = tmp_path / '照片.png'
    Image.new('RGB', (80, 50), 'blue').save(source)
    before = hashlib.sha256(source.read_bytes()).digest()
    output = tmp_path / 'result'
    report = run_demo(source, output, '亮度 0')
    assert report['rotation']['status'] == 'untrained'
    assert report['applied_ccw_degrees'] == 0
    assert hashlib.sha256(source.read_bytes()).digest() == before
    assert Image.open(output / 'candidate_ccw_90.png').size == (50, 80)
    with pytest.raises(ValueError):
        run_demo(source, output)
    manual = run_demo(source, tmp_path / 'manual', '亮度 0', angle=90)
    assert manual['applied_ccw_degrees'] == 90
    assert Image.open(tmp_path / 'manual/05_rotated.png').size == (50, 80)


def test_rotation_checkpoint_sign_and_confidence(tmp_path):
    from tranning.photo_edit import OrientationNet
    model = OrientationNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias[1] = 10
    checkpoint = tmp_path / 'orientation.pt'
    torch.save({'kind': 'orientation-v1', 'state': model.state_dict(), 'metrics': {}}, checkpoint)
    angle, decision = choose_rotation(Image.new('RGB', (20, 40)), checkpoint)
    assert angle == 270 and decision['status'] == 'accepted'
    with torch.no_grad():
        model.net[-1].bias.zero_()
    torch.save({'kind': 'orientation-v1', 'state': model.state_dict(), 'metrics': {}}, checkpoint)
    assert choose_rotation(Image.new('RGB', (20, 40)), checkpoint)[1]['status'] == 'uncertain'


def test_training_smoke_and_identity_split(tmp_path):
    photos = tmp_path / 'upright'
    photos.mkdir()
    for i in range(3):
        image = Image.new('RGB', (30 + i, 40), (50 + i * 30, 80, 120))
        image.save(photos / f'{i}.png')
    Image.open(photos / '0.png').save(photos / 'duplicate.png')
    train, valid = split_photos(photos)
    assert len(train) + len(valid) == 3
    assert not set(train) & set(valid)
    metrics = train_orientation(photos, tmp_path / 'orientation.pt', epochs=1)
    assert 0 <= metrics['validation_accuracy'] <= 1
    pair = tmp_path / 'pairs/example'
    pair.mkdir(parents=True)
    Image.new('RGB', (40, 60), (70, 80, 100)).save(pair / 'source.png')
    Image.new('RGB', (40, 60), (90, 100, 120)).save(pair / 'target.png')
    (pair / 'instruction.txt').write_text('變亮', encoding='utf-8')
    editor = tmp_path / 'editor.pt'
    assert train_editor(pair.parent, editor, 1)['pairs'] == 1
    from tranning.photo_edit import GrayTextEditor
    torch.manual_seed(7)
    baseline = GrayTextEditor(3).state_dict()
    trained = torch.load(editor, weights_only=True)['state']
    assert any(not torch.equal(value, trained[name]) for name, value in baseline.items())
    assert learned_edit(Image.open(pair / 'source.png'), '變亮', editor).size == (40, 60)
    with pytest.raises(ValueError):
        learned_edit(Image.open(pair / 'source.png'), '未知', editor)


@pytest.mark.parametrize('size', [(1, 1), (12, 30)])
def test_tiny_photo_cli_defaults(tmp_path, monkeypatch, capsys, size):
    from lib.photo_edit_demo import main
    source, output = tmp_path / 'tiny.png', tmp_path / 'tiny_demo'
    Image.new('RGB', size, 'blue').save(source)
    monkeypatch.setattr('sys.argv', ['photo_edit_demo.py', '--input', str(source), '--output', str(output)])
    main()
    assert Image.open(output / '06_edited_color.png').size == size
    assert 'strict_operations' in capsys.readouterr().out
    with pytest.raises(ValueError, match='座標'):
        edit_photo(Image.new('RGB', size), parse_instructions('添加圓形 40 40 15 #ff5500'))


def test_circuit_ink_preserves_thin_wire_and_gap(tmp_path):
    from tranning.photo_edit import extract_edges
    gray = np.full((40, 60), 255, dtype=np.uint8)
    gray[20, 5:55] = 0
    gray[20, 30] = 255  # 刻意保留的單像素斷線，不應自動接通。
    edges, ink = extract_edges(gray, 'circuit')
    assert np.array_equal(ink, np.where(gray == 0, 255, 0).astype(np.uint8))
    assert ink[20, 29] == 255 and ink[20, 30] == 0 and ink[20, 31] == 255
    assert np.count_nonzero(edges) > 0
    source = tmp_path / 'circuit.png'
    Image.fromarray(gray).save(source)
    report = run_demo(source, tmp_path / 'out', mode='circuit', edge_low=30, edge_high=90)
    assert report['instruction'] == '' and report['mode'] == 'circuit'
    assert report['features']['edge_thresholds'] == [30, 90]
    assert (tmp_path / 'out/07_circuit_ink.png').exists()
    assert np.array_equal(np.asarray(Image.open(tmp_path / 'out/06_edited_color.png')), np.asarray(Image.open(source).convert('RGB')))
    for low, high in [(90, 30), (-1, 60), (float('nan'), 160)]:
        with pytest.raises(ValueError):
            extract_edges(gray, 'circuit', low, high)


def test_circuit_practice_cli_saves_comparisons(tmp_path, monkeypatch):
    from lib.img_edge import main
    gray = np.full((50, 80), 255, dtype=np.uint8)
    gray[20, 5:70] = 0
    source, output = tmp_path / 'diagram.png', tmp_path / 'practice'
    Image.fromarray(gray).save(source)
    monkeypatch.setattr('sys.argv', ['img_edge.py', '--mode', 'circuit', '--input', str(source), '--output', str(output)])
    main()
    assert len(list(output.glob('edges_*.png'))) == 3
    assert np.count_nonzero(np.asarray(Image.open(output / 'circuit_ink.png'))) == 65
    with pytest.raises(SystemExit):
        main()  # 輸出已存在時不能覆寫。


def test_breadboard_preserves_color_and_separates_analysis(tmp_path):
    from tranning.photo_edit import breadboard_color_features, extract_edges
    rgb = np.full((50, 80, 3), 210, dtype=np.uint8)
    rgb[10:15, 5:60] = (255, 0, 0)
    rgb[25:30, 5:60] = (0, 0, 0)  # 黑線不在飽和色遮罩，仍須保存於彩圖及邊緣。
    source, output = tmp_path / 'board.png', tmp_path / 'board_output'
    Image.fromarray(rgb).save(source)
    source_bytes = source.read_bytes()
    edges, ink = extract_edges(np.asarray(Image.fromarray(rgb).convert('L')), 'breadboard')
    assert ink is None
    mask, colors, overlay = breadboard_color_features(rgb, edges)
    assert mask[12, 20] == 255 and mask[27, 20] == 0
    assert tuple(colors[12, 20]) == (255, 0, 0)
    assert np.array_equal(overlay[edges == 0], rgb[edges == 0])
    report = run_demo(source, output, mode='breadboard')
    assert source.read_bytes() == source_bytes
    assert report['instruction'] == ''
    assert report['breadboard_analysis']['component_detection'] == 'not_implemented'
    assert np.array_equal(np.asarray(Image.open(output / '06_edited_color.png')), rgb)
    assert (output / '09_edge_overlay.png').is_file()
    assert not (output / '07_circuit_ink.png').exists()


def test_breadboard_practice_cli_saves_color(tmp_path, monkeypatch):
    from lib.img_edge import main
    source, output = tmp_path / 'breadboard.png', tmp_path / 'practice'
    Image.new('RGB', (40, 30), (255, 0, 0)).save(source)
    monkeypatch.setattr('sys.argv', ['img_edge.py', '--mode', 'breadboard', '--input', str(source), '--output', str(output)])
    main()
    assert Image.open(output / 'original_color.png').getpixel((10, 10)) == (255, 0, 0)
    assert Image.open(output / 'color_regions.png').getpixel((10, 10)) == (255, 0, 0)
    assert len(list(output.glob('edge_overlay_*.png'))) == 3
