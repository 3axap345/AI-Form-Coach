PYTHON ?= python

.PHONY: setup test lint format clean lock

setup:
	$(PYTHON) -m pip install --require-hashes -r requirements-dev.txt

test:
	$(PYTHON) -m unittest discover -s collector/tests -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path) for path in Path('.').rglob('__pycache__')]; [shutil.rmtree(path) for path in (Path('.pytest_cache'), Path('.ruff_cache')) if path.exists()]; [path.unlink() for path in Path('.').rglob('*.pyc')]"

lock:
	$(PYTHON) -m piptools compile --generate-hashes --allow-unsafe --output-file=requirements.txt requirements.in
	$(PYTHON) -m piptools compile --generate-hashes --allow-unsafe --output-file=requirements-dev.txt requirements-dev.in
