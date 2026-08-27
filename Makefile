PYTHON ?= python

.PHONY: setup test lint coverage typecheck build format clean lock

setup:
	$(PYTHON) -m pip install --require-hashes -r requirements-dev.txt

test:
	$(PYTHON) -m unittest discover -s collector/tests -v

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m unittest discover -s collector/tests -v
	$(PYTHON) -m coverage report --show-missing

typecheck:
	$(PYTHON) -m mypy

build:
	$(PYTHON) scripts/build_source_bundle.py

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path) for path in Path('.').rglob('__pycache__')]; [shutil.rmtree(path) for path in (Path('.pytest_cache'), Path('.ruff_cache'), Path('.mypy_cache'), Path('htmlcov')) if path.exists()]; [path.unlink() for path in Path('.').rglob('*.pyc')]; [path.unlink() for path in Path('dist').glob('ai-form-coach-*.zip')] if Path('dist').exists() else None; Path('.coverage').unlink(missing_ok=True)"

lock:
	$(PYTHON) -m piptools compile --generate-hashes --allow-unsafe --index-url https://pypi.org/simple --output-file=requirements.txt requirements.in
	$(PYTHON) -m piptools compile --generate-hashes --allow-unsafe --index-url https://pypi.org/simple --output-file=requirements-dev.txt requirements-dev.in
