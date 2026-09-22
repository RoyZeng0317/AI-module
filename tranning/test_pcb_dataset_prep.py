import cv2, numpy as np
from pcb_dataset_prep import prep


def test_prep_crops_by_class_and_skips_bad(tmp_path):
    (tmp_path / "img").mkdir(); (tmp_path / "xml").mkdir()
    cv2.imwrite(str(tmp_path / "img" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    obj = lambda n, b: f"<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
    (tmp_path / "xml" / "a.xml").write_text(f"<annotation>{obj('short',(10,10,40,40))}{obj('spur',(50,50,52,52))}</annotation>")
    (tmp_path / "xml" / "missing.xml").write_text(f"<annotation>{obj('short',(1,1,30,30))}</annotation>")
    out = tmp_path / "out"
    assert prep(tmp_path / "img", tmp_path / "xml", out, pad=0) == {"short": 1}
    assert len(list((out / "short").glob("*.jpg"))) == 1 and not (out / "spur").exists()
