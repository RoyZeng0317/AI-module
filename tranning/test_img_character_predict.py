from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import img_character_predict as predictor


def test_prepare_chinese_path_keeps_color_and_source(tmp_path):
    path = tmp_path / '人物照片.png'
    Image.new('RGB', (16, 12), (255, 0, 0)).save(path)
    before = path.read_bytes()
    image = predictor.prepare_image(path)
    assert image.shape == (12, 16, 3) and image.flags.writeable
    assert image[0, 0].tolist() == [0, 0, 255]
    assert path.read_bytes() == before


def test_transparent_background_is_white(tmp_path):
    path = tmp_path / 'transparent.png'
    Image.new('RGBA', (8, 8), (0, 0, 0, 0)).save(path)
    assert np.all(predictor.prepare_image(path) == 255)


def test_prediction_uses_model_names_and_threshold(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'person.png'
    Image.new('RGB', (20, 20)).save(path)
    observed = {}
    class Model:
        def predict(self, image, **kwargs):
            observed.update(kwargs)
            print('model diagnostic noise')
            return [SimpleNamespace(names={0: '人物甲', 1: '人物乙'}, boxes=[SimpleNamespace(cls=[1]), SimpleNamespace(cls=[0])])]
    monkeypatch.setattr(predictor, 'get_model', lambda: Model())
    predictor.main(path, .15, gui=False)
    assert capsys.readouterr().out == '人物乙\n人物甲\n'
    assert observed['conf'] == .15 and observed['verbose'] is False


def test_no_detection_and_cancel(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'person.png'
    Image.new('RGB', (20, 20)).save(path)
    monkeypatch.setattr(predictor, 'get_model', lambda: SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=[], names={})]))
    predictor.main(path, gui=False)
    assert capsys.readouterr().out == '未偵測到人物\n'
    monkeypatch.setattr(predictor, 'test_img_dir', None)
    monkeypatch.setattr(predictor, 'choose_image', lambda: '')
    monkeypatch.setattr(predictor, 'get_model', lambda: pytest.fail('取消不應載入模型'))
    predictor.main(gui=False)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('conf', [0, 1, float('nan')])
def test_invalid_threshold_rejected_before_reading(conf):
    with pytest.raises(ValueError, match='門檻'):
        predictor.predict_image('missing.png', conf)


def test_missing_image_and_empty_weights(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        predictor.prepare_image(tmp_path / 'missing.png')
    path = tmp_path / 'characters.pt'; path.write_bytes(b'')
    monkeypatch.setattr(predictor, 'model_path', path)
    with pytest.raises(ValueError, match='空檔'):
        predictor.get_model()


def test_tk_window_shows_boxes_names_empty_and_error(tmp_path, monkeypatch):
    import time, tkinter as tk, torch
    from tkinter import filedialog, messagebox
    class Boxes:
        xyxy = torch.tensor([[2., 3., 18., 19.]])
        cls = torch.tensor([0.]); conf = torch.tensor([.9])
        def __len__(self): return len(self.cls)
        def __iter__(self):
            for i in range(len(self)):
                yield SimpleNamespace(cls=self.cls[i:i+1], conf=self.conf[i:i+1])
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    result = SimpleNamespace(names={0: '人物甲'}, boxes=Boxes())
    observed, errors = [], []
    def detect(path, conf):
        observed.append(conf)
        return image, result
    monkeypatch.setattr(predictor, 'detect_image', detect)
    monkeypatch.setattr(messagebox, 'showerror', lambda *args, **kwargs: errors.append(args[1]))
    root = tk.Tk(); root.withdraw()
    app = predictor.CharacterImageWindow(root)
    def wait():
        deadline = time.monotonic()+5
        while app.busy and time.monotonic() < deadline:
            root.update(); time.sleep(.01)
        assert not app.busy
    try:
        app.conf.set('.15'); app.start(tmp_path / '人物.png'); wait()
        assert observed == [.15]
        assert '人物甲' in app.result.get() and '90.0%' in app.result.get()
        assert app.canvas.find_all() and app.photo is not None
        result.boxes.cls = torch.empty(0); result.boxes.conf = torch.empty(0); result.boxes.xyxy = torch.empty((0, 4))
        app.start(tmp_path / 'empty.png'); wait()
        assert app.result.get() == '未偵測到人物'
        monkeypatch.setattr(filedialog, 'askopenfilename', lambda **kwargs: '')
        app.open_image(); assert not app.busy
        def fail(*args): raise ValueError('測試讀圖錯誤')
        monkeypatch.setattr(predictor, 'detect_image', fail)
        app.start(tmp_path / 'broken.png'); wait()
        assert errors[-1] == '測試讀圖錯誤' and app.preview is None
        assert app.result.get() == '辨識失敗'
        app.conf.set('nan'); app.start(tmp_path / 'x.png')
        assert not app.busy and '門檻' in errors[-1]
        app.close()
    finally:
        try: root.destroy()
        except tk.TclError: pass
