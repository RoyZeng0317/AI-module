"""Explicit download only; inference never calls this script."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'BAAI/bge-small-zh-v1.5'


def main():
    # Keep all download artifacts inside this project; do not read .env.
    os.environ['HF_HOME'] = str(ROOT / 'models/.hf-cache')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from huggingface_hub import HfApi, snapshot_download

    destination = ROOT / 'models/bge-small-zh-v1.5'
    revision = HfApi(token=False).model_info(MODEL_ID).sha
    snapshot_download(MODEL_ID, revision=revision, local_dir=destination, token=False,
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                      'tokenizer_config.json', 'special_tokens_map.json',
                                      'vocab.txt', 'README.md', 'LICENSE*'])
    required = ['config.json', 'model.safetensors', 'tokenizer.json']
    if any(not (destination / name).is_file() for name in required):
        raise RuntimeError('Model download is incomplete; do not use this directory yet.')
    metadata = {'model_id': MODEL_ID, 'revision': revision,
                'source': f'https://huggingface.co/{MODEL_ID}',
                'license': 'MIT', 'pooling': 'CLS + L2',
                'purpose': 'local contextual encoder; not GPT weights'}
    (destination / 'download_source.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
