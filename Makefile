clean:
	find ./src -type d -name "__pycache__" -exec rm -rf {} +
	find ./src -type f -name "*.pyc" -exec rm -f {} +
	find ./src -type d -name ".pytest_cache" -exec rm -rf {} +
