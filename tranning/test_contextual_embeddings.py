"""Engineering tests use random local weights, never a semantic quality claim."""
import pytest, torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from tranning.contextual_embeddings import ContextualEncoder


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_context_encoder")
    (path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "蘋", "果", "水", "手", "機", "店", "買"]), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=False)
    # Transformers 5 uses vocab=; retain compatibility with older releases.
    if tokenizer.vocab_size != 12:
        tokenizer = BertTokenizerFast(vocab=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    BertModel(BertConfig(vocab_size=12, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=24, max_position_embeddings=32)).save_pretrained(path)
    return path


@pytest.fixture
def encoder(model_dir):
    return ContextualEncoder(model_dir, max_length=16, batch_size=2)


def test_sentence_cls_and_batch_padding(encoder):
    texts = ["蘋果", "水果店買蘋果"]
    actual = encoder.encode_sentences(texts)
    batch = encoder.tokenizer(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        expected = torch.nn.functional.normalize(encoder.model(**batch).last_hidden_state[:, 0], dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0], encoder.encode_sentences(texts[:1])[0])
    assert actual.shape == (2, 16) and not actual.requires_grad


def test_explicit_repeated_span_and_mean(encoder):
    text = "蘋果店買蘋果"
    actual = encoder.encode_spans([text, text], [(0, 2), (4, 6)])
    batch = encoder.tokenizer([text], return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder.model(**batch).last_hidden_state
        expected = torch.nn.functional.normalize(torch.stack([hidden[0, 1:3].mean(0), hidden[0, 5:7].mean(0)]), dim=1)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0], actual[1])


@pytest.mark.parametrize("span", [(-1, 1), (0, 0), (2, 1), (0, 10), (True, 2), (0,), "蘋果"])
def test_bad_spans(encoder, span):
    with pytest.raises(ValueError):
        encoder.encode_spans(["蘋果"], [span])


def test_empty_and_invalid_texts(encoder):
    assert encoder.encode_sentences([]).shape == (0, 16)
    assert encoder.encode_spans([], []).shape == (0, 16)
    for texts in ([""], [" "], [None], "蘋果"):
        with pytest.raises(ValueError):
            encoder.encode_sentences(texts)
    with pytest.raises(ValueError):
        encoder.encode_spans(["蘋果"], [])


def test_truncation_rejects_partial_and_full(model_dir):
    encoder = ContextualEncoder(model_dir, max_length=5)
    for span in [(2, 4), (4, 6)]:
        with pytest.raises(ValueError, match="truncated"):
            encoder.encode_spans(["水果店買蘋果"], [span])
    assert encoder.encode_spans(["水果店買蘋果"], [(0, 2)]).shape == (1, 16)


@pytest.mark.parametrize("kwargs", [{"max_length": 2}, {"max_length": True}, {"max_length": 33}, {"batch_size": 0}, {"device": "meta"}, {"device": "invalid"}, {"device": "cpu:1"}])
def test_bad_configuration(model_dir, kwargs):
    with pytest.raises(ValueError):
        ContextualEncoder(model_dir, **kwargs)


def test_all_attention_masked(encoder):
    original = encoder.tokenizer

    class MaskedTokenizer:
        def __call__(self, *args, **kwargs):
            result = original(*args, **kwargs)
            result["attention_mask"].zero_()
            return result

    encoder.tokenizer = MaskedTokenizer()
    with pytest.raises(ValueError, match="no real"):
        encoder.encode_sentences(["??"])


@pytest.mark.parametrize("control", ["[MASK]", "[CLS]", "[PAD]"])
def test_literal_controls_are_not_content(encoder, control):
    with pytest.raises(ValueError, match="no real"):
        encoder.encode_sentences([control])
    with pytest.raises(ValueError, match="no real"):
        encoder.encode_spans([control], [(0, len(control))])
    with pytest.raises(ValueError, match="no real"):
        encoder.encode_spans([control + "??"], [(0, len(control))])


def test_mixed_span_excludes_controls_but_keeps_unknown(encoder):
    text = "?[MASK]?"
    batch = encoder.tokenizer([text], return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder.model(**batch).last_hidden_state
        expected = torch.nn.functional.normalize(hidden[:, [1, 3]].mean(1), dim=1)
    torch.testing.assert_close(encoder.encode_spans([text], [(0, len(text))]), expected)
    # A real out-of-vocabulary character maps to UNK and remains usable.
    assert encoder.encode_spans(["?"], [(0, 1)]).shape == (1, 16)
