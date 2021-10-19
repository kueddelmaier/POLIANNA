#%%
import sys
import pandas as pd 
import os
sys.path.append('/home/jkuettel/NLP_spark/src')
sys.path.append('/home/jkuettel/NLP_spark')
import src.experiment_utils
from src.experiment_utils.helper_classes import token, tag, repository
from definitions import ROOT_DIR



dataframe_dir = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe.pkl')
stat_df = pd.read_pickle(dataframe_dir)

#%%
test_tags = stat_df.loc['EU_32009L0028_Title_0_Chapter_0_Section_0_Article_07']
# %%
from pygamma_agreement import Continuum
from pyannote.core import Segment


annotators = ['Fabian', 'Alisha', 'Onerva']

continuum = Continuum()
for annotator in annotators:
    for tag in test_tags[annotator]:
        continuum.add(annotator, Segment(tag.start, tag.stop), tag.tag_)

from pygamma_agreement import CombinedCategoricalDissimilarity

dissim = CombinedCategoricalDissimilarity(list(continuum.categories))
gamma_results = continuum.compute_gamma(dissim)
print(f"The gamma for that annotation is f{gamma_results.gamma}")
# %%
from itertools import chain
from itertools import groupby
# %%
hello = chain.from_iterable(stat_df)

# %%
