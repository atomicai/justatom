fix-format:
	@PY_FILES="$$(git ls-files '*.py')"; \
	if [ -z "$$PY_FILES" ]; then \
		echo "No tracked Python files found."; \
		exit 0; \
	fi; \
	isort $$PY_FILES; \
	black $$PY_FILES

format-check:
	@PY_FILES="$$(git ls-files '*.py')"; \
	if [ -z "$$PY_FILES" ]; then \
		echo "No tracked Python files found."; \
		exit 0; \
	fi; \
	black --check --diff $$PY_FILES; \
	isort --check-only --diff $$PY_FILES

fix-format-noqa:
	@$(MAKE) fix-format
	-ruff check --add-noqa
	-ruff check --add-noqa
