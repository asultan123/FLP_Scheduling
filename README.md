# FLP_Scheduling

A benchmark for testing multiapplication scheduling techniques on heterogeneous platforms like: https://doi.org/10.1007/s11265-015-1058-5. 
Simulation of benchmarked techniques available at: https://github.com/neu-ece-esl/FLP_Scheduling

## To install poetry
(Assuming linux like enviornment)

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

## To install virtual enviornment in current directory (useful if vscode isn't picking up on venv)
poetry config virtualenvs.in-project true

## To install dependencies
poetry install

## To start virtual enviornment
poetry shell

