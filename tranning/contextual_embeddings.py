"""Optional offline contextual encoder; independent of the self-trained chat models."""
from pathlib import Path


class ContextualEncoder:
    """Local Hugging Face encoder. Sentence CLS pooling; explicit character spans.

    Spans use Python string offsets [start, end), one per input sentence.
    Outputs are detached, L2-normalized CPU torch tensors.
    """

    def __init__(self, model_dir, device="cpu", max_length=512, batch_size=8):
        import torch
        from transformers import AutoModel, AutoTokenizer

        for name, value, minimum in (("max_length", max_length, 3), ("batch_size", batch_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        try:
            resolved = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("device must be cpu or an available cuda device") from exc
        if resolved.type not in ("cpu", "cuda") or (resolved.type == "cpu" and resolved.index is not None):
            raise ValueError("device must be cpu or an available cuda device")
        if resolved.type == "cuda":
            index = resolved.index if resolved.index is not None else torch.cuda.current_device() if torch.cuda.is_available() else 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError("requested CUDA device is unavailable")
            torch.cuda.set_per_process_memory_fraction(0.85, index)
        path = Path(model_dir).resolve()
        if not path.is_dir():
            raise ValueError(f"local model directory does not exist: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("a fast tokenizer with offset mapping is required")
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and max_length > limit:
            raise ValueError(f"max_length exceeds model position limit {limit}")
        if self.tokenizer.cls_token_id is None:
            raise ValueError("sentence pooling requires a CLS token")
        # Unknown text is content; control tokens are never span content, even
        # when typed literally and reported with nonzero character offsets.
        self.control_ids = set(self.tokenizer.all_special_ids) - {self.tokenizer.unk_token_id}
        self.device, self.max_length, self.batch_size = resolved, max_length, batch_size
        self.dimension = self.model.config.hidden_size
        self.model.to(resolved).eval()

    @staticmethod
    def _texts(texts):
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence of nonempty strings")
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("each text must be a nonempty string")
        return texts

    def encode_sentences(self, texts):
        return self._encode(self._texts(texts))

    def encode_spans(self, texts, spans):
        texts, spans = self._texts(texts), list(spans)
        if len(texts) != len(spans):
            raise ValueError("provide exactly one span per text")
        for text, span in zip(texts, spans):
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                raise ValueError("each span must be a (start, end) pair")
            start, end = span
            if any(isinstance(x, bool) or not isinstance(x, int) for x in span) or not 0 <= start < end <= len(text):
                raise ValueError("span must satisfy 0 <= start < end <= len(text)")
        return self._encode(texts, spans)

    def _encode(self, texts, spans=None):
        import torch
        import torch.nn.functional as F

        if not texts:
            return torch.empty((0, self.dimension))
        output = []
        with torch.inference_mode():
            for first in range(0, len(texts), self.batch_size):
                chunk = texts[first:first + self.batch_size]
                encoded = self.tokenizer(chunk, padding=True, truncation=True, max_length=self.max_length,
                                         return_offsets_mapping=True, return_tensors="pt")
                offsets = encoded.pop("offset_mapping")
                mask = encoded["attention_mask"].bool()
                real = mask & (offsets[:, :, 1] > offsets[:, :, 0])
                for token_id in self.control_ids:
                    real &= encoded["input_ids"] != token_id
                if not real.any(dim=1).all():
                    raise ValueError("text has no real, unmasked tokens")
                selected = []
                if spans is not None:
                    # Compare untruncated offsets so a partly retained span cannot pass.
                    full = self.tokenizer(chunk, truncation=False, return_offsets_mapping=True)
                    for row, (start, end) in enumerate(spans[first:first + len(chunk)]):
                        expected = [(a, b) for (a, b), token_id in zip(full["offset_mapping"][row], full["input_ids"][row])
                                    if token_id not in self.control_ids and b > a and a < end and b > start]
                        keep = real[row] & (offsets[row, :, 0] < end) & (offsets[row, :, 1] > start)
                        actual = [tuple(pair) for pair in offsets[row][keep].tolist()]
                        if not expected:
                            raise ValueError("span overlaps no real tokens")
                        if actual != expected:
                            raise ValueError("span is partially or fully truncated; shorten text or increase max_length")
                        selected.append(keep.to(self.device))
                inputs = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self.model(**inputs).last_hidden_state
                if spans is None:
                    if not (encoded["input_ids"][:, 0] == self.tokenizer.cls_token_id).all() or not mask[:, 0].all():
                        raise ValueError("sentence pooling requires an unmasked CLS at position zero")
                    pooled = hidden[:, 0]
                else:
                    pooled = torch.stack([hidden[row, keep].mean(0) for row, keep in enumerate(selected)])
                if not torch.isfinite(pooled).all() or (pooled.norm(dim=1) == 0).any():
                    raise ValueError("encoder produced non-finite or zero vectors")
                output.append(F.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(output)
