.PHONY: install dev-install run test lint clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

run:
	uvicorn qa_intelli_media_comparator.main:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	python -m py_compile qa_intelli_media_comparator/**/*.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage
