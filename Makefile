OS := $(shell uname 2>/dev/null || echo Windows)

.PHONY: install install-dev lint format type-check test clean run dashboard

install:
	poetry install --no-dev

install-dev:
	poetry install

lint:
	poetry run ruff check src tests
	poetry run black --check src tests

format:
	poetry run ruff check --fix src tests
	poetry run black src tests

type-check:
	poetry run mypy src

test:
	poetry run pytest

clean:
ifeq ($(OS), Windows)
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force"
	powershell -Command "Get-ChildItem -Recurse -Include *.pyc,*.pyo,*.pyd,.coverage | Remove-Item -Force"
	powershell -Command "Get-ChildItem -Recurse -Directory -Include *.egg-info,.pytest_cache,.mypy_cache,.ruff_cache,htmlcov | Remove-Item -Recurse -Force"
else
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
endif

check: format lint type-check test

run:
	poetry run python -m src.main

dashboard:
	poetry run streamlit run src/streamlit_app.py

