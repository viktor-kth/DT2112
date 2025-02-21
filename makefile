VENV = venv
PYTHON = $(VENV)/bin/python
ACTIVATE = $(VENV)/bin/activate


install_dev:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	source $(ACTIVATE) && pre-commit install

clean:
	find . -type d -name .pytest_cache | xargs --no-run-if-empty -t rm -r
	find . -type d -name __pycache__ | xargs --no-run-if-empty -t rm -r
	find . -type d -name dist | xargs --no-run-if-empty -t rm -r
	find . -type d -name build | xargs --no-run-if-empty -t rm -r
	find . -type d -regex ".*\.egg.*"  | xargs --no-run-if-empty -t rm -r
	rm -rf $(VENV)

.PHONY: install_dev clean