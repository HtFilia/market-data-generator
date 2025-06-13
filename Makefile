.PHONY: install test lint format clean gui

install:
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

test:
	pytest tests/ --cov=. --cov-report=term-missing

lint:
	mypy .
	black . --check

format:
	black .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +

gui:
	streamlit run visualization/visualizer.py 