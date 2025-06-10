# Define variables
VENV_NAME = env
PYTHON = python # Adjust this if you need to use python3 specifically
PIP = $(VENV_NAME)\Scripts\pip.exe  # Correct path for Windows

# List of packages to install
REQUIREMENTS = requirements.txt

.PHONY: all create_venv install activate clean

# Default target
all: create_venv install activate

# Create the virtual environment
create_venv:
	$(PYTHON) -m venv $(VENV_NAME)

# Install packages from requirements.txt
install:
	$(PIP) install -r $(REQUIREMENTS)

# Activate the virtual environment
activate:
	@echo "To activate the virtual environment, run the following command:"
	@echo "env\Scripts\activate" 

# Clean up the virtual environment
clean:
	rmdir /S /Q $(VENV_NAME) 
