from itertools import chain
import multiprocessing
import numpy as np
import pandas as pd
from src.d03_inter_annotator_agreement.scoring_functions import unified_gamma, create_scoring_matrix, check_symmetric 
from src.experiment_utils.helper_classes import token, tag, repository
from definitions import df_annotation_marker

from pandarallel import pandarallel

from multiprocessing import Pool

score = 0
total_tokens = 0

category_list, cat_dissimilarity_matrix = create_scoring_matrix('/home/jkuettel/NLP_spark/src/experiment_utils/tag_set.json',  soft_layer_dissimilarity = True)

cat_dissimilarity_matrix_test = np.ones((len(category_list), len(category_list)))
np.fill_diagonal(cat_dissimilarity_matrix_test,0)
scoring_method = 'unified_gamma'

def keep_valid_anotations(tag_series):
  
    
    len_list = [len(x) for x in tag_series if type(x) == list]  
    quantil = np.quantile(len_list, 0.7)
    valid_annotations = [x for x in tag_series if type(x) == list and len(x) > 0.3*quantil]
    return valid_annotations


def keep_valid_anotations_simple(tag_series):
      
    valid_annotations = [x for x in tag_series if type(x) == list and len(x) > 2]
    return valid_annotations

def apply_scoring_function(row, scoring_method, df_annotation_marker, category_list, cat_dissimilarity_matrix):

    
    annotations = row[df_annotation_marker:]
    valid_annotations = keep_valid_anotations(annotations)
    
    score = np.nan
       
    if len(valid_annotations) < 2:
        return ('not complete annotation')
    valid_annotations_flat = list(chain.from_iterable(valid_annotations))

    if scoring_method == 'unified_gamma':
        alpha = 1
        beta = 1
        try:
            gamma_results = unified_gamma(valid_annotations_flat, category_list, alpha=alpha, beta=beta, cat_dissimilarity_matrix = cat_dissimilarity_matrix)
            score = gamma_results.gamma 
            #gamma_results = unified_gamma(valid_annotations_flat, category_list, alpha=alpha, beta=beta, cat_dissimilarity_matrix = cat_dissimilarity_matrix_test)
        except Exception as e: 
            print(e)
    
    if scoring_method == 'f1':
        
    
        pass

            

    return score


#def apply_func(df):
#    df['score'] = df.apply(lambda row: apply_unified_gamma(row, scoring_method, df_annotation_marker, category_list, cat_dissimilarity_matrix), axis=1)

        

class Inter_Annotator_Agreement:
   
    def __init__(self, df, DEBUG):
        
        if DEBUG:
            self.df = df[0:200]
        else:
            self.df = df
        
    def get_total_score(self, scoring_method, append_to_df = False, weight_by_tokens = True):
        score = 0
        total_tokens = 0
        df_annotation_marker = 4
        category_list, cat_dissimilarity_matrix = create_scoring_matrix('/home/jkuettel/NLP_spark/src/experiment_utils/tag_set.json',  soft_layer_dissimilarity = True)
  
        cat_dissimilarity_matrix_test = np.ones((len(category_list), len(category_list)))
        np.fill_diagonal(cat_dissimilarity_matrix_test,0)

        for index, row in self.df.iterrows():
            annotations = row[df_annotation_marker:]
            valid_annotations = keep_valid_anotations(annotations)
            
            if len(valid_annotations) < 2:
                continue
            valid_annotations_flat = list(chain.from_iterable(valid_annotations))

            if scoring_method == 'unified_gamma':
                alpha = 1
                beta = 1
                try:
                    gamma_results = unified_gamma(valid_annotations_flat, category_list, alpha=alpha, beta=beta, cat_dissimilarity_matrix = cat_dissimilarity_matrix) 
                    #gamma_results = unified_gamma(valid_annotations_flat, category_list, alpha=alpha, beta=beta, cat_dissimilarity_matrix = cat_dissimilarity_matrix_test)
                except Exception as e: 
                    print(e)
                    continue

                    
                    
                N_tokens = len(row['Tokens'])
                score += (gamma_results.gamma * N_tokens)
                total_tokens += N_tokens
                print(index,gamma_results.gamma)



    
        print('----------------------------------------')
        print('calculation finished')
        print('score is: ',score/total_tokens)
    # to do: 
    # 1. implement function to get scores of specific annotator
    # 2. funtion to get score for specific article
    # 3. create different score functions
    # 4.    


    

    def inter_annotator_agremment(self, scoring_metrics, append_to_df = False, weight_by_tokens = True):

        score = 0
        total_tokens = 0
        

        category_list, cat_dissimilarity_matrix = create_scoring_matrix('/home/jkuettel/NLP_spark/src/experiment_utils/tag_set.json',  soft_layer_dissimilarity = True)

        cat_dissimilarity_matrix_test = np.ones((len(category_list), len(category_list)))
        np.fill_diagonal(cat_dissimilarity_matrix_test,0)
        N_tokens = sum(self.df['Tokens'].apply(len))
        pandarallel.initialize()
        for scoring_metric in scoring_metrics:
    
            self.df[scoring_metrics] = self.df.parallel_apply(lambda row: apply_scoring_function(row, scoring_metric, df_annotation_marker, category_list, cat_dissimilarity_matrix), axis=1)


        