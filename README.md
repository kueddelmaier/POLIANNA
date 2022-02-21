# Spark NLP Project

This repository contains the code for the NLP Spark Project [Link to preprint paper]. 

[Short description of the project]

This README is not intended to be self-explanatory, but rather should be read together with the original paper. Below we give an overview of the corpus, the annotation procedure and 

## Project


## Installing dependencies

[I still need to figure out a way to create one environment that works for every funtion. For the moment, there is a requirements.text]

### Virtual Environment

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Conda

```bash
conda create --name <env> --file requirements.txt
```

## Data

The annotated corpus is publicly available under [insert link where data is published]

For the moment, the data is avaiable on the Polybox. Download this folder and store the unzipt folder directory in the RAW_DATA_PATH variable in the ``definitions.py`` file. 

Process the raw data by running the ``src/d01_data/load_data.py`` script.

## Running the 

