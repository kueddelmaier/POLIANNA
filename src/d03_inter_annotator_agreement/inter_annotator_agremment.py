import multiprocessing
from itertools import chain, combinations
from multiprocessing import Pool, Value

import numpy as np
import pandas as pd
from definitions import df_annotation_marker
from pandarallel import pandarallel
from tqdm import tqdm
from src.d02_corpus_statistics.corpus import Corpus, Sent_Corpus
from src.d03_inter_annotator_agreement.scoring_functions import (
    check_symmetric, create_scoring_matrix, scoring_metrics, unified_gamma, f1_heuristic)
from src.d03_inter_annotator_agreement.span_matching import matching_methods
from src.experiment_utils.helper_classes import repository, span, token

score = 0
total_tokens = 0

category_list, cat_dissimilarity_matrix = create_scoring_matrix('/home/jkuettel/NLP_spark/src/experiment_utils/tag_set.json',  soft_layer_dissimilarity = True)

cat_dissimilarity_matrix_test = np.ones((len(category_list), len(category_list)))
np.fill_diagonal(cat_dissimilarity_matrix_test,0)
scoring_method = 'unified_gamma'

def is_valid_annotation(span_list):
    return type(span_list) == list and len(span_list) >= 2

def keep_valid_anotations(span_series):
  
    
    len_list = [len(x) for x in span_series if type(x) == list]  
    quantil = np.quantile(len_list, 0.7)
    valid_annotations = [x for x in span_series if type(x) == list and len(x) > 0.3*quantil]
    return valid_annotations


def keep_valid_anotations_simple(span_series):
      
    #valid_annotations = [x for x in span_series if type(x) == list and len(x) > 2]
    valid_annotations = [x for x in span_series if type(x) == list]
    return valid_annotations

def row_to_span_list(row):
    annotations = row[row['Finished_Annotators']]
    annotations_flat = list(chain.from_iterable(annotations))
    return annotations_flat



def _get_score_article(span_list,  scoring_metric, finished_annotators, **optional_tuple_properties):
    """
    Calculates scoring metric based on tuple algo of spanlist of a single article. Optional tuple properties related to tuple matching, e.g gamma

    """
    if not isinstance(scoring_metric, str):
        raise ValueError('scoring metric must be a string')

    #annotators = set([span_.annotator for span_ in span_list])
    annotators = finished_annotators

    if len(annotators) < 2:
        raise ValueError('not two finished annotators found')


    if scoring_metric == 'pygamma':

        unique_annotators = len(set([span_.annotator for span_ in span_list])) # pygamma requires that both annotators have at least one span

        if unique_annotators < 2:
            score = 1 - unique_annotators #if either is no answer, the score is 1 if they agree, 0 otherwise
        else:
            score = unified_gamma(span_list, **optional_tuple_properties)
        return score


    #create tuples:
    else:
        if scoring_metric not in scoring_metrics and scoring_metric != 'f1_heuristic':
            raise ValueError('This metric: ', scoring_metric, ' does not exist')

        score = 0

        for annotator_pair in combinations(annotators,2):

            span_list_annotator_pair = [span_ for span_ in span_list if span_.annotator in annotator_pair]

            # the pygamma matching method runs into problem if one annotator or both have no spans
            # therefore handle this case here
            
            spans_a1 = [span_ for span_ in span_list_annotator_pair if span_.annotator == annotator_pair[0]]
            spans_a2 = [span_ for span_ in span_list_annotator_pair if span_.annotator == annotator_pair[1]]


            if len(spans_a1) == 0 or len(spans_a2) == 0: 
                #if either is no answer, the score is 1 if they agree, 0 otherwise
                score += int(len(spans_a1) == len(spans_a2))


            elif scoring_metric == 'f1_heuristic':
                score += f1_heuristic(span_list_annotator_pair, annotator_pair)
            
            else:
                #span_tuples = matching_methods[tuple_algo](span_list_annotator_pair, **optional_tuple_properties)
                span_tuples = matching_methods['pygamma'](span_list_annotator_pair, **optional_tuple_properties)
                score += scoring_metrics[scoring_metric] (span_tuples)
        
        return score/len(list(combinations(annotators,2)))



class Inter_Annotator_Agreement(Corpus):
   
    def __init__(self, df, DEBUG = False, front_and_whereas = False):

        super().__init__(df, front_and_whereas)
        
        if DEBUG:
            self.df = self.df[0:10]
        
        self.annotators = list(self.df.columns[df_annotation_marker:])




    def append_total_score_per_article_parallel(self, scoring_metrics, append_to_df = False, weight_by_tokens = True, **optional_tuple_properties):

        pandarallel.initialize(progress_bar = True)

        for scoring_metric in scoring_metrics:
            
            column_name = '_'.join([scoring_metric, 'score'])
            if column_name in self.df.columns:
                continue

            self.df[column_name] = self.df.parallel_apply(lambda row: _get_score_article(row_to_span_list(row), scoring_metric, **optional_tuple_properties), axis=1)


    def append_total_score_per_article(self, scoring_metrics, append_to_df = False, weight_by_tokens = True, **optional_tuple_properties):
        

        tqdm.pandas()

        for scoring_metric in scoring_metrics:
            
            column_name = '_'.join([scoring_metric, 'score'])
            if column_name in self.df.columns:
                continue

            self.df[column_name] = self.df.progress_apply(lambda row: _get_score_article(row_to_span_list(row), scoring_metric, row['Finished_Annotators'],  **optional_tuple_properties), axis=1)


    def get_total_score_df(self, columns = 'all', annotator = 'all', weight_by = 'Tokens'):
        
        df_columns = self.df.columns
        

        if columns == 'all':
            columns = [column for column in df_columns if 'score' in column]

        elif isinstance(columns, str):
            if columns in df_columns:
                columns = [columns]
            else:
                raise ValueError('This score does not exist')

        elif isinstance(columns, list):
            non_valid_scores = [x for x in columns if x not in df_columns]
            if len(non_valid_scores) != 0:
                raise ValueError(non_valid_scores, 'do not exist')
        else:
            raise ValueError('Enter columns as a single string or as a list of strings')
        
        if annotator in self.annotators:
            df_annotator = self.df[self.df.apply(lambda x: annotator in x['Finished_Annotators'],axis=1)]
        
        elif annotator == 'all':
            df_annotator = self.df

        else:
            raise ValueError('This annotator does not exist! Enter a valid annotator')


        score_dict = {}
        

        if weight_by == 'no_weighting':

            for score_col in columns:
        
                score_dict[score_col] = df_annotator[score_col].mean()
            
            return score_dict

        

        if weight_by == 'Tokens':
            total_n_tokens = len(list(chain.from_iterable(df_annotator['Tokens'])))

            for score_col in columns:
                score = df_annotator.apply(lambda x: len(x['Tokens']) * x[score_col] / total_n_tokens, axis=1).sum()
                score_dict[score_col] = score
            
            return score_dict
        
        elif weight_by == 'Spans':

            for score_col in columns:
                score = df_annotator.apply(lambda x: len(list(chain.from_iterable(x['Finished_Annotators']))) * x[score_col], axis=1).sum()
                total_n_spans = df_annotator.apply(lambda x: len(list(chain.from_iterable(x['Finished_Annotators']))), axis=1).sum()
                score_dict[score_col] = score/total_n_spans
            
            return score_dict
        
        elif weight_by == 'Span_Tokens':
    
            raise NotImplementedError
        
        else:
            raise ValueError('This weighting method is not valid!')




    def get_score_annotator(self, annotator, columns = 'all', weight_by_tokens = True):
        df_columns = self.df.columns

        if columns == 'all':
            columns = [column for column in df_columns if 'score' in column]
            if len(columns) == 0:
                raise ValueError('no score calculated yet, first calculate scores and append to dataframe ')

        elif isinstance(columns, str):
            if columns in df_columns:
                columns = [columns]
            else:
                raise ValueError('This score does not exist')

        elif isinstance(columns, list):
            non_valid_scores = [x for x in columns if x not in df_columns]
            if len(non_valid_scores) != 0:
                raise ValueError(non_valid_scores, 'do not exist')
        else:
            raise ValueError('Enter columns as a single string or as a list of strings')

        #keep only the columns where the annotator has valid annotations
        df_annotator = self.df[self.df.apply(lambda x: annotator in x['Finished_Annotators'],axis=1)]

   
        
        #get total tokens
        total_n_tokens = len(list(chain.from_iterable(df_annotator['Tokens'])))
        
        score_dict = {}

        for score_column in columns:
            score_dict[score_column] = df_annotator.apply(lambda x: len(x['Tokens']) * x[score_column] / total_n_tokens, axis=1).sum()
        #return the mean of scores weighted by the number of tokens
        return score_dict
        
                
    def get_score_spanlist(self, span_list, scoring_metric, weight_by_tokens = True, **optional_tuple_properties):
        """
        Extends get_score_spanlist for spanlist with multiple articles

        """

        repos = list(set([span_.rep for span_ in span_list]))
        total_score = 0
        total_tokens = 0
        
        for repo in repos:
            n_tokens = self.get_token_count_from_repository(repo)
            if n_tokens == 0:
                continue


            span_list_repo = [span_ for span_ in span_list if span_.rep == repo]
            annotators_span_list_repo = set([span_.annotator for span_ in span_list_repo])

            if len(annotators_span_list_repo) < 2:
                continue
          #  if len(annotators_span_list_repo) < 2 or len():
          #      continue

            if weight_by_tokens:
                n_tokens = n_tokens

            else: n_tokens = 1

            total_score += _get_score_article(span_list_repo , scoring_metric, annotators_span_list_repo, **optional_tuple_properties) * n_tokens
            total_tokens += n_tokens
          
        return total_score / total_tokens
    
    

class Sent_Inter_Annotator_Agreement(Sent_Corpus, Inter_Annotator_Agreement):
    pass





            
          
            
    


