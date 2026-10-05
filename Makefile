PYTHON ?= python
SRC := src scripts tests

.PHONY: install data train serve test lint replay replay-drift drift bench up

install:
	$(PYTHON) -m pip install -e ".[train,monitor,dev]"

data:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) -m cardshield.train

serve:
	$(PYTHON) -m uvicorn cardshield.serve.app:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check $(SRC)
	$(PYTHON) -m ruff format --check $(SRC)
	$(PYTHON) -m mypy

replay:
	@echo "make replay: not implemented yet (Sprint 4)"

replay-drift:
	@echo "make replay-drift: not implemented yet (Sprint 4)"

drift:
	@echo "make drift: not implemented yet (Sprint 4)"

bench:
	@echo "make bench: not implemented yet (Sprint 4)"

up:
	@echo "make up: not implemented yet (Sprint 3)"
