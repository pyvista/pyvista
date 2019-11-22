# Simple makefile to simplify repetitive build env management tasks under posix

CODESPELL_DIRS ?= "docs pyvista examples"
CODESPELL_SKIP ?= "*.pyc,*.txt,*.gif,*.png,*.jpg,*.ply,*.vtk,*.vti"
CODESPELL_IGNORE ?= "ignore_words.txt"

all: codespell

codespell:
	codespell $CODESPELL_DIRS -S $CODESPELL_SKIP -I $CODESPELL_IGNORE
