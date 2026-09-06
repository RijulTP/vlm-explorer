.PHONY: install download run clean

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

download:
	. .venv/bin/activate && pip install hf_transfer
	. .venv/bin/activate && hf download openai/clip-vit-base-patch32 --local-dir ./model_cache/clip-vit-base-patch32

run:
	. .venv/bin/activate && TOKENIZERS_PARALLELISM=false python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .venv model_cache __pycache__ backend/__pycache__
