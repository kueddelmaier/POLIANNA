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

Process the raw data by running the ``src/d01_data/load_data.py`` script. The cleaned corpus can then be found under data/d02_processed_to_dataframe as a pkl and csv format.

## Corpus Class
The Corpus class in ``src/d02_corpus_statistics/corpus.py`` contains all the relevant functions to calculate the relevant corpus statistics such as annotation counts and tag frequencies. A working example of all the functions can be found in the Jupyter Notebook ``notebooks/Corpus.ipynb``


## Inter Annotator Agreement Class
The Inter Annotator Agreement class in ``src/d03_inter_annotator_agreement`` contains all the relevant functions to calculate the relevant inter annotator agreement scores. A working example of all the functions can be found in the Jupyter Notebook ``notebooks/Inter_Annotator_Agreement.ipynb``

## Tests and more examples
For the moment, test notebooks and more examples (such as sentence wise implementation) can be found under notebooks/test_notebooks

