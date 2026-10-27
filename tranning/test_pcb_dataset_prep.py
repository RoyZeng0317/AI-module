import cv2, numpy as np
from pcb_dataset_prep import prep, prep_yolo


def test_prep_crops_by_class_and_skips_bad(tmp_path):
    (tmp_path / "img").mkdir(); (tmp_path / "xml").mkdir()
    cv2.imwrite(str(tmp_path / "img" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    obj = lambda n, b: f"<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
    (tmp_path / "xml" / "a.xml").write_text(f"<annotation>{obj('short',(10,10,40,40))}{obj('spur',(50,50,52,52))}</annotation>")
    (tmp_path / "xml" / "missing.xml").write_text(f"<annotation>{obj('short',(1,1,30,30))}</annotation>")
    out = tmp_path / "out"
    assert prep(tmp_path / "img", tmp_path / "xml", out, pad=0) == {"short": 1}
    assert len(list((out / "short").glob("*.jpg"))) == 1 and not (out / "spur").exists()


def test_prep_yolo_writes_images_labels_and_data_yaml(tmp_path):
    (tmp_path / "img").mkdir(); (tmp_path / "xml").mkdir()
    obj = lambda n, b: f"<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
    for i in range(6):
        cv2.imwrite(str(tmp_path / "img" / f"a{i}.jpg"), np.zeros((100, 100, 3), np.uint8))
        (tmp_path / "xml" / f"a{i}.xml").write_text(
            f"<annotation>{obj('short', (10, 10, 40, 40))}{obj('spur', (50, 50, 80, 80))}</annotation>"
        )
    out = tmp_path / "yolo_out"
    counts = prep_yolo(tmp_path / "img", tmp_path / "xml", out, val_ratio=0.5, seed=1)

    assert counts == {"short": 6, "spur": 6}
    data_yaml = (out / "data.yaml").read_text()
    assert "names:" in data_yaml and "short" in data_yaml and "spur" in data_yaml
    train_imgs = list((out / "images" / "train").glob("*.jpg"))
    val_imgs = list((out / "images" / "val").glob("*.jpg"))
    assert len(train_imgs) + len(val_imgs) == 6
    for img in train_imgs + val_imgs:
        label = out / "labels" / img.parent.name / f"{img.stem}.txt"
        lines = label.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            c, *box = line.split()
            assert int(c) in (0, 1) and all(0 <= float(v) <= 1 for v in box)


def test_prep_yolo_respects_explicit_classes(tmp_path):
    (tmp_path / "img").mkdir(); (tmp_path / "xml").mkdir()
    cv2.imwrite(str(tmp_path / "img" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    (tmp_path / "xml" / "a.xml").write_text(
        "<annotation><object><name>spur</name><bndbox><xmin>10</xmin><ymin>10</ymin>"
        "<xmax>40</xmax><ymax>40</ymax></bndbox></object></annotation>"
    )
    out = tmp_path / "yolo_out"
    prep_yolo(tmp_path / "img", tmp_path / "xml", out, classes=["short", "spur"], val_ratio=0.0)
    label = next((out / "labels" / "train").glob("*.txt"))
    assert label.read_text().startswith("1 ")  # "spur" 是 classes 清單裡的第 2 個 (index 1)
