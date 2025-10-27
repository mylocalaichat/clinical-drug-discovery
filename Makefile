.PHONY: help run install mlflow dagster dev test clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

mlflow:  ## Start MLflow server on port 5000
	@echo "Starting MLflow server..."
	@echo "Navigate to: http://localhost:5000"
	uv run mlflow server --host 127.0.0.1 --port 5000

dagster:  ## Start Dagster web server
	@echo "Starting Dagster web server..."
	@echo "Navigate to: http://localhost:3000"
	uv run dagster dev

# Alias for backwards compatibility
run: dagster  ## Alias for dagster command

test:  ## Run basic validation tests
	@echo "Testing imports..."
	uv run python -c "from dagster_definitions import defs; print('✓ Dagster definitions loaded')"
	uv run python -c "import mlflow; print('✓ MLflow available')"
	uv run python -c "from neo4j import GraphDatabase; print('✓ Neo4j driver available')"
	@echo "✓ All tests passed!"

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned up Python cache files"
