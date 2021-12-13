project_name := xport
SHELL := /bin/bash
.SHELLFLAGS := -c

help:					# Display all make targets with inline comments
	@grep -E '^[a-z/.\-]+:.*#' Makefile
.PHONY: help


########################################################################
# GNU Standard Targets
# https://www.gnu.org/prep/standards/html_node/Standard-Targets.html

all: dist install
	@echo '----'
	@echo 'To test this installation, run ``make check``'
	@echo '----'

install: 				# Install the Python package
	python -m pip install .

install-html: html 			# Build the documentation website
	@echo 'Docs website available in docs/_build/html'
install-dvi: dvi
install-pdf: pdf
install-ps: ps

uninstall:
	python -m pip uninstall -y $(project_name)

clean:					# Remove all build and test artifacts
	rm -rf dist
	rm -rf build
	rm -rf docs/_build
	rm -rf .eggs
	rm -rf src/*.egg-info
	rm -rf src/*/__pycache__
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache
	rm -rf test/__pycache__
	rm -rf test/*/__pycache__
	rm -rf test/**/__pycache__

html:
	$(MAKE) -C docs html
dvi:
pdf:
ps:

dist:		 			# Build a "wheel" for distribution
	python setup.py sdist bdist_wheel

check: 					# Verify that everything is working
	python setup.py test


########################################################################
# Miscellaneous

git-hooks:				# Instal Git hooks
	git config core.hookspath .githooks

pypi: clean check dist			# Upload to PyPI
	twine upload --repository pypi --config-file ~/.pypirc --verbose dist/*

