# POLIcy design ANNotAtions (POLIANNA)

This repository contains the code for the manuscript entitled "Towards understanding policy design through text-as-data approaches: The POLIcy design ANNotAtions (POLIANNA) dataset" [Link to follow]. 

This README is not intended to be self-explanatory, but rather should be read together with the original paper. Below we give an overview of the project and how to use this repository. 

## Paper abstract
Despite the importance of ambitious policy action for addressing climate change, large and systematic assessments of public policies and their design are lacking as analysing text manually is labour-intensive and costly. POLIANNA is a dataset of policy texts from the European Union (EU) that are annotated based on theoretical concepts of policy design, which can be used to develop supervised machine learning approaches for scaling policy analysis. The dataset consists of 20,577 annotated spans, drawn from 18 EU climate change mitigation and renewable energy policies. We developed a novel coding scheme translating existing taxonomies of policy design elements to a method for annotating text spans that consist of one or several words. Here, we provide the coding scheme, a description of the annotated corpus, and an analysis of inter-annotator agreement, and discuss potential applications. As understanding policy texts is still difficult for current text-processing algorithms, we envision this database to be used for building tools that help with manual coding of policy texts by automatically proposing paragraphs containing relevant information.

## Installing dependencies

### Conda

```bash
conda env create -f environment.yml
```
Then manually install the pygamma package[^1] over pip:


```bash
conda activate POLIANNA
pip install git+https://github.com/bootphon/pygamma-agreement.git
```

## Data

The annotated corpus is publicly available at https://doi.org/10.5281/zenodo.8284380. This folder contains a file with the data preprocessed as a dataframe as it is needed for working with this repository. If you wish to process data from scratch, for example to add your own annotatations, we also provide a way to process data exported from Inception[^2].

### Processed data
Download this folder and store the subfolder 02_processed_to_dataframe under the directory ``/data/02_processed_to_dataframe``.

### Processing from Inception
Store your raw data folder directory in the RAW_DATA_PATH variable in the ``definitions.py`` file. Note that the data published at the link above is not the raw data in the Inception format, which is not anonymous, but converted in a generally usable format.

Process the raw data by running the ``src/d01_data/load_data.py`` script. The cleaned corpus can then be found under data/d02_processed_to_dataframe as a pkl and csv format. The script allows to anonymize the data with the following flag:

```bash
python load_data.py --anonymous_annotators=1
```

## Corpus Class
The Corpus class in ``src/d02_corpus_statistics/corpus.py`` contains all the relevant functions to calculate the relevant corpus statistics such as annotation counts and tag frequencies. A working example of all the functions can be found in the Jupyter Notebook ``notebooks/Tutorial.ipynb``.

## Inter-Annotator Agreement Class
The Inter Annotator Agreement class in ``src/d03_inter_annotator_agreement`` contains all the relevant functions to calculate the relevant inter annotator agreement scores. A working example of all the functions can be found in the Jupyter Notebook ``notebooks/Inter_Annotator_Agreement.ipynb``

## Coding scheme
A file containing the coding scheme in JSON format can be found in the provided dataset at ``01_policy_info/Coding_Scheme.json``. Store this under the ``data`` directory as well.

## Getting started
The Jupyter Notebook ``notebooks/Tutorial.ipynb`` introduces the classes and simple examples how to work with the data.

## Analysis
The notebook ``notebooks/Analysis.ipynb`` contains the descriptive analysis of the data and figures used in the paper.

## Tests and more examples
Test notebooks can be found under ``notebooks/test_notebooks`` and more examples (such as sentence wise implementation) can be found under ``notebooks/other``.

## Other scripts to process your own data
For the labeled data, we have split the EU laws into articles. We provide the scripts to download EU laws as ``text_processing/download_searches.py`` and to split those into articles as ``text_processing/process_text.ipynb``.

## References
[^1]: https://pygamma-agreement.readthedocs.io/en/latest/
[^2]: https://github.com/inception-project