import os
import pandas as pd
import sys
import time
import swifter
print(sys.path)
sys.path.append('/home/jkuettel/NLP_spark/src')
sys.path.append('/home/jkuettel/NLP_spark')
from src.experiment_utils.helper_classes import token, span, repository

from definitions import ROOT_DIR
from src.d03_inter_annotator_agreement.inter_annotator_agremment import Inter_Annotator_Agreement

DEBUG = True

def main():

    dataframe_dir = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe.pkl')    
    stat_df = pd.read_pickle(dataframe_dir)

    Evaluator = Inter_Annotator_Agreement(stat_df, DEBUG = DEBUG)

    
    scoring_metrics = ['f1_exact', 'f1_tokenwise']

    Evaluator.get_total_score(scoring_metrics, tuple_algos, append_to_df = True, weight_by_tokens = True)

    print(Evaluator.df.head())
    #print('---------------normal------------------')
    #start_time = time.time()
    #Evaluator.get_total_score(scoring_method = 'unified_gamma', append_to_df = True)
    #print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()