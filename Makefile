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

install: conda-create git-hooks 	# Install into a Conda virtual environment
	@echo '----'
	@echo 'Run ``conda activate $(venv_name)`` to enable the project environment'
	@echo '----'

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

dist: conda-update 			# Build a "wheel" for distribution
	$(conda_activate) && python setup.py sdist bdist_wheel

check: conda-update 			# Verify that everything is working
	$(conda_activate) && python setup.py test


########################################################################
# Miscellaneous

git-hooks:
	git config core.hookspath .githooks

pypi: clean dist			# Upload to PyPI
	$(MAKE) clean
	$(MAKE) dist
	twine upload --repository pypi --config-file ~/.pypirc dist/*


########################################################################
# Miniconda
# For virtual environments and package management.  Handles both Python
# and non-Python project dependencies.  The OS package manager may have
# better maintenance for less popular non-Python packages, but using the
# OS package manager will install globally instead of in an environment.
# https://docs.conda.io/en/latest/miniconda.html

venv_name = $(shell grep 'name:' environment.yml | cut -d" " -f2)

ifdef CONDA_EXE
miniconda_install_path := $(shell echo ${CONDA_EXE} | sed 's/\/bin\/conda$$//')
conda_exe := $(CONDA_EXE)
else
miniconda_install_path = ~/lib/miniconda
conda_exe := $(miniconda_install_path)/bin/conda
endif

ifeq ($(shell uname), Darwin)
conda_os = MacOSX
else ifeq ($(shell uname), Linux)
conda_os = Linux
else
$(error Could not identify operating system)
endif

miniconda_installer_baseurl = https://repo.anaconda.com/miniconda
miniconda_installer_filename = Miniconda3-latest-$(conda_os)-x86_64.sh
miniconda_installer_url = $(miniconda_installer_baseurl)/$(miniconda_installer_filename)
miniconda_env_path = $(miniconda_install_path)/envs/$(venv_name)
conda_activate = source $(miniconda_install_path)/bin/activate $(venv_name)

$(conda_exe):
	mkdir -vp ~/.conda
	mkdir -vp $(miniconda_install_path)
	curl --url $(miniconda_installer_url) --output ~/miniconda.sh
	bash ~/miniconda.sh -bup $(miniconda_install_path)
	rm ~/miniconda.sh
	$(conda_exe) init bash zsh fish

$(miniconda_env_path): $(conda_exe) miniconda-update
	$(conda_exe) create --name $(venv_name) --yes

miniconda-update: $(conda_exe)
	$(conda_exe) update -n base conda -c defaults --yes

# TODO: Stop using PIP_SRC variable when Conda bug #5861 is fixed.
#       https://github.com/conda/conda/issues/5861

# TODO: When ``update --prune`` is fixed, use that; it's faster.
#       https://github.com/conda/conda/issues/7279
#       $(conda_exe) env update --name $(venv_name) --file environment.yml --prune

conda-create: $(miniconda_env_path) miniconda-update
	PIP_SRC=$(miniconda_env_path)/src \
		$(conda_exe) env create --name $(venv_name) --file environment.yml --force
	@$(conda_activate) && echo "Done, installed at `which python`"

conda-update: miniconda-update 		# Update the Conda virtual environment
	# BUG: This does not prune unused packages.
	#      Use ``make install`` to recreate the env from scratch.
	#      https://github.com/conda/conda/issues/7279
	PIP_SRC=$(miniconda_env_path)/src \
		$(conda_exe) env update --name $(venv_name) --file environment.yml --prune

conda-export:				# Show the current Conda environment specification
	@$(conda_exe) env export --no-builds | grep -Ev '^prefix:| - $(venv_name)=='
