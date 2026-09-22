"""驗證猜拳遊戲的判斷/計分邏輯，以及用真實模型跑一次合成假圖片確認推論管線本身能跑完，
不代表真實手勢辨識準確度。"""
import numpy as np

from main import (
    load_labels, normalize_choice, judge, predict, load_interpreter, LABELS_PATH,
)


def test_load_labels_parses_index_and_name():
    labels = load_labels(LABELS_PATH)
    assert labels[0] == "Scissors"
    assert labels[1] == "Rock"


def test_normalize_choice_handles_known_typo_in_labels_file():
    assert normalize_choice("Rock") == "rock"
    assert normalize_choice("Scissors") == "scissors"
    assert normalize_choice("Papaer") == "paper"  # labels.txt 本身就拼錯，靠子字串比對容錯
    assert normalize_choice("unknown") is None


def test_judge_rock_paper_scissors_and_tie():
    assert judge("rock", "scissors") == "你贏"
    assert judge("scissors", "paper") == "你贏"
    assert judge("paper", "rock") == "你贏"
    assert judge("rock", "paper") == "電腦贏"
    assert judge("rock", "rock") == "平手"


def test_predict_pipeline_runs_end_to_end_on_synthetic_image():
    interpreter, input_details, output_details = load_interpreter()
    labels = load_labels(LABELS_PATH)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    raw_label, confidence = predict(interpreter, input_details, output_details, labels, frame)
    assert raw_label in labels.values()
    assert 0.0 <= confidence <= 1.0
