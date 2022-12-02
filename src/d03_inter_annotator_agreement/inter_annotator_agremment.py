import multiprocessing
from itertools import chain, combinations
from multiprocessing import Pool, Value

import numpy as np
import pandas as pd
from definitions import df_annotation_marker
from pandarallel import pandarallel
from tqdm import tqdm
import collections
from src.d02_corpus_statistics.corpus import Corpus, Sent_Corpus
from src.d03_inter_annotator_agreement.scoring_functions import (
    check_symmetric, create_scoring_matrix, scoring_metrics, unified_gamma)
from src.d03_inter_annotator_agreement.span_matching import matching_methods
from src.experiment_utils.helper_classes import repository, span, token


def keep_valid_anotations(span_series):

    """
    only for testing purposes

    """

    len_list = [len(x) for x in span_series if type(x) == list]  
    quantil = np.quantile(len_list, 0.7)
    valid_annotations = [x for x in span_series if type(x) == list and len(x) > 0.3*quantil]
    return valid_annotations



def row_to_span_list(row):

    """
    return a list of spans of all the finished annotators for a specific row

    """
    annotations = row[row['Finished_Annotators']]
    annotations_flat = list(chain.from_iterable(annotations))
    return annotations_flat

def cur_annotator_to_span_list(row, annotator):

    """
    return a list of spans for the specific annotator and the Curation

    """
    annotations = row[['Curation', annotator]]
    annotations_flat = list(chain.from_iterable(annotations))
    return annotations_flat


def _get_curation_annotator_score(row, scoring_metrics_to_calc, total_curation_scores, annotator,  **optional_tuple_properties):

    """
    Calculates curation-annotator score for all the scores defined in "scoring_metrics_to_calc" for the "annotator". 
    The "total_curation_scores" are the curation-annotator-scores that have already been calculated, including the new scores. 
    Since we create a row for each annotator-to-curation and store the different IAA scores as tuples, we need to know at which index to store the scores 
    Optional tuple properties related to tuple matching, e.g gamma

    """

    spanlist = cur_annotator_to_span_list(row, annotator) #get of all curation spans and annotator spans

    if collections.Counter(scoring_metrics_to_calc) == collections.Counter(total_curation_scores): #the scores to calculate == the scores of the no score has been calculated yet
        new_score_array = np.zeros(len(scoring_metrics_to_calc)) #create empty score array
        new_score_indexes = np.arange(len(scoring_metrics_to_calc)) #get the indexes on where to insert the newly calculated scores
    else:
        assert len(scoring_metrics_to_calc) < len(total_curation_scores), f"scoring metrics to calculate are longer than total scoring metrics to calc: {scoring_metrics_to_calc} total: {total_curation_scores}" 
        old_score_array = row['_'.join([annotator, 'to_curation'])] #retrieve the already calculated scores

        new_score_array = np.zeros(len(total_curation_scores))
        new_score_indexes = np.array([total_curation_scores.index(score) for score in scoring_metrics_to_calc]) #get the indexes on where to insert the newly calculated scores (alphabetically sorted)

        old_score_array_indexes = np.array([total_curation_scores.index(score) for score in total_curation_scores if score not in scoring_metrics_to_calc]) #get indexes of already calculated scores
        new_score_array[old_score_array_indexes] = old_score_array #insert already calculated scores in array (alphabetically sorted)
    
    finished_annotators = ['Curation', annotator] 

    for i, scoring_metric in zip(new_score_indexes, scoring_metrics_to_calc): #iterate over scores to calculate

        new_score_array[i] = _get_score_article(spanlist, scoring_metric, finished_annotators,  **optional_tuple_properties) # insert newly calculated scores in array
    return new_score_array


def to_curation_colums(row):

    """
    Returns all the annotator-to-curation scores of the finished annotators of a specific row.

    """
    return row[['{}_to_curation'.format(annotator) for annotator in row['Finished_Annotators']]]

def _get_span_count_in_row(row, cols):

    """
    Returns the sum of items of a row present in the specified columns.

    """
    return len(list(chain.from_iterable(row[cols])))



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
        if scoring_metric not in scoring_metrics:
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

            else:
                score += scoring_metrics[scoring_metric] (span_list_annotator_pair, annotator_pair, **optional_tuple_properties)

        
        return score/len(list(combinations(annotators,2)))



class Inter_Annotator_Agreement(Corpus):
   
    def __init__(self, df, DEBUG = False, front_and_whereas = False):
        """
        
        Inter Annotator Agreement
        Parameters
        ----------
        df : Dataframe
            Dataframe where each column represents an article
        
        DEBUG: bool
            If True, only the first 10 rows of df are kept. For debug purposes.

        front_and_whereas: bool
            If True, Front and whereas articles are included in the corpus
        

        """

        super().__init__(df, front_and_whereas)
        
        if DEBUG:
            self.df = self.df[0:10]
        
        self.calculated_iaa_scores = []
        self.calculated_curation_scores = []
        
        

    def append_total_score_per_article_parallel(self, scoring_metrics, append_to_df = False, weight_by_tokens = True, **optional_tuple_properties):


        pandarallel.initialize(progress_bar = True)
        #what if score is singular, so tring but not list?

        for scoring_metric in scoring_metrics:
            
            column_name = '_'.join([scoring_metric, 'score'])
            if column_name in self.df.columns:
                continue

            self.df[column_name] = self.df.parallel_apply(lambda row: _get_score_article(row_to_span_list(row), scoring_metric, row['Finished_Annotators'],  **optional_tuple_properties), axis=1)


    def append_total_score_per_article(self, scoring_metrics, parallel = False, **optional_tuple_properties):

        """
        For each IAA score in scoring_metrics, appends a new row that calculates IAA-score for each article.

        Parameters
        ----------

        scoring metrics: str or list of str
             IAA-scores to calculate, either a single score or a list of scores. Score must be defined in scoring_functions.py

        parallel: bool
            If True, pandarallel library is used

        
        optional_tuple_properties: optional
            Properties related to pygamma such as category_list or cat_dissimilarity_matrix for either span matching or pygamma score.
            For example use case see Inter_Annotator_Agreement Notebook

        """

        if isinstance(scoring_metrics, str):
            scoring_metrics = [scoring_metrics]

        scoring_metrics_to_calc = [metric for metric in scoring_metrics if metric not in self.calculated_iaa_scores]


        if parallel:
            pandarallel.initialize(progress_bar = True)
            for scoring_metric in scoring_metrics:
                
                column_name = '_'.join([scoring_metric, 'score'])
                if column_name in self.df.columns:
                    continue

                self.df[column_name] = self.df.parallel_apply(lambda row: _get_score_article(row_to_span_list(row), scoring_metric, row['Finished_Annotators'],  **optional_tuple_properties), axis=1)


        else:
            tqdm.pandas()
            for scoring_metric in scoring_metrics:
                
                column_name = '_'.join([scoring_metric, 'score'])
                if column_name in self.df.columns:
                    continue

                self.df[column_name] = self.df.progress_apply(lambda row: _get_score_article(row_to_span_list(row), scoring_metric, row['Finished_Annotators'],  **optional_tuple_properties), axis=1)
            

        self.calculated_iaa_scores = self.calculated_iaa_scores + scoring_metrics_to_calc

    
    def append_score_to_curation(self, scoring_metrics, parallel = False, **optional_tuple_properties):
        # add if score alrewasdy in df then dont calculate

        """
        For each IAA score in scoring_metrics, appends a new row that calculates to-curation score for each article and for each score defined in scoring_metrics.
        Scores are stored as touples.

        Parameters
        ----------

        scoring metrics: str or list of str
             IAA-scores to calculate, either a single score or a list of scores. Score must be defined in scoring_functions.py

        parallel: bool
            If True, pandarallel library is used

        
        optional_tuple_properties: optional
            Properties related to pygamma such as category_list or cat_dissimilarity_matrix for either span matching or pygamma score.
            For example use case see Inter_Annotator_Agreement Notebook

        """
        if isinstance(scoring_metrics, str):
            scoring_metrics = [scoring_metrics]

        scoring_metrics_to_calc = [metric for metric in scoring_metrics if metric not in self.calculated_curation_scores] # All the scores that are not yet calculated

        self.calculated_curation_scores = (self.calculated_curation_scores + scoring_metrics_to_calc) # Append the new calcualted scores to the list 
        self.calculated_curation_scores = sorted(self.calculated_curation_scores) #order alphabetically
        
        if parallel:
            pandarallel.initialize(progress_bar = True)
            for annotator in self.finished_annotators:
                column_name = '_'.join([annotator, 'to_curation'])

                self.df[column_name] = self.df.parallel_apply(lambda row: _get_curation_annotator_score(row, sorted(scoring_metrics_to_calc), self.calculated_curation_scores, annotator, **optional_tuple_properties) if annotator in row['Finished_Annotators'] else '', axis=1)

        else:
            tqdm.pandas()
            for annotator in self.finished_annotators:
                column_name = '_'.join([annotator, 'to_curation'])

                self.df[column_name] = self.df.progress_apply(lambda row: _get_curation_annotator_score(row, sorted(scoring_metrics_to_calc), self.calculated_curation_scores, annotator, **optional_tuple_properties) if annotator in row['Finished_Annotators'] else '', axis=1)

        

    def get_total_score_df(self, scoring_metrics = 'all', annotator = 'all', weight_by = 'Tokens'):

        """
        Calculates a weighted average of the score columns in the DataFrame. Only possible if the score column exists in the DataFrame. 
        If not, first append score with "append_total_score_per_article". By default, all the scores are calculated, but can be specified with "scoring_metrics".

        Parameters
        ----------

        scoring metrics: str or list of str
             IAA-scores to calculate, either a single score or a list of scores. Score must be a column of the DataFrame.

        annotator: str
            Either 'all' or a specific annotator. Default is 'all', calulates average over all articles. 
            If specific annotator, calculates weighted IAA average over subset of DataFrame where the specific annotator in "Finished Annotators".

        
        weight_by: {'no_weighting', 'Tokens', 'Spans'}
            no_weighting: Calculates simple average over all articles
            Tokens: Weights each article by number of Tokens
            Spans: Weights each article by number of spans 

        Returns
        ----------
        out : dict
            Score dict containing the weighted averages for all the scores defined in "scoring_metrics"
            """

        if scoring_metrics == 'all':
            scoring_metrics = ['_'.join([col, 'score']) for col in self.calculated_iaa_scores]

        elif isinstance(scoring_metrics, str):
            if scoring_metrics in self.calculated_iaa_scores:

                scoring_metrics = ['_'.join([scoring_metrics, 'score'])]
                    
            else:
                raise ValueError('This score does not exist')


        elif isinstance(scoring_metrics, list):

            not_calculated_scoring_metrics = [metric for metric in scoring_metrics if metric not in self.calculated_iaa_scores]

            if len(not_calculated_scoring_metrics) != 0:
                raise ValueError(f"The scoring metrics {not_calculated_scoring_metrics} are not calculated yet or don't exist" )
            
            scoring_metrics = ['_'.join([metric, 'score']) for metric in scoring_metrics]
            

        else:
            raise ValueError('Enter scores as a single string or as a list of strings')


        
        if annotator in self.finished_annotators:
            df_annotator = self.df[self.df.apply(lambda x: annotator in x['Finished_Annotators'],axis=1)]
        
        elif annotator == 'all':
            df_annotator = self.df

        else:
            raise ValueError('This annotator does not exist! Enter a valid annotator')


        score_dict = {}
        

        if weight_by == 'no_weighting':
            for score_col in scoring_metrics:
                score_dict[score_col] = df_annotator[score_col].mean()
            return score_dict

        
        elif weight_by == 'Tokens':
            total_n_tokens = len(list(chain.from_iterable(df_annotator['Tokens'])))

            for score_col in scoring_metrics:
                score = df_annotator.apply(lambda row: len(row['Tokens']) * row[score_col] / total_n_tokens, axis=1).sum()
                score_dict[score_col] = score
            
            return score_dict
        
        elif weight_by == 'Spans':
            total_n_spans = df_annotator.apply(lambda row: _get_span_count_in_row(row, cols = row['Finished_Annotators']), axis=1).sum() # sum up all the spans of finished annotators

            for score_col in scoring_metrics:
                score = df_annotator.apply(lambda row:  _get_span_count_in_row(row, cols = row['Finished_Annotators']) * row[score_col], axis=1).sum() #multiply len of spans by score
                score_dict[score_col] = score/total_n_spans
            
            return score_dict
        
        elif weight_by == 'Span_Tokens':
    
            raise NotImplementedError
        
        else:
            raise ValueError('This weighting method is not valid!')



    def get_to_curation_score(self, weight_by = 'Tokens'):

        """
        Calculates a weighted average of the all the to-curation scores in the DataFrame for each annotator. 
        I.e for each annotator and each score we calculate a weighted to-curation score.
        Note that scores first need to be appended to DataFrame with "append_score_to_curation". 

        Parameters
        ----------
        
        weight_by: {'no_weighting', 'Tokens', 'Spans'}
            no_weighting: Calculates simple average over all articles
            Tokens: Weights each article by number of Tokens
            Spans: Weights each article by number of spans 
        

        Returns
        ----------
        out : dict
            Named dict containing a score-dict for each annotator.
            For each annotor get a dict with weighted average to-curation score for all to-curation scores in DataFrame.
            
        """
        score_dict = {}
        for annotator in self.finished_annotators:
            row_name = '{}_to_curation'.format(annotator)
        
            df_annotator = self.df[self.df.apply(lambda row: annotator in row['Finished_Annotators'],axis=1)]

            if weight_by == 'no_weighting':
                scores = df_annotator[row_name].mean()
                score_dict[annotator] = dict(zip(self.calculated_curation_scores, scores.T))

            elif weight_by == 'Tokens':
                total_n_tokens = len(list(chain.from_iterable(df_annotator['Tokens'])))
                scores = df_annotator.apply(lambda row: len(row['Tokens']) * row[row_name] / total_n_tokens, axis=1).sum()
                score_dict[annotator] = dict(zip(self.calculated_curation_scores, scores.T))

            elif weight_by == 'Spans':
                total_n_spans = df_annotator.apply(lambda row: _get_span_count_in_row(row, cols = [annotator, 'Curation']), axis=1).sum()
                scores = df_annotator.apply(lambda row:  _get_span_count_in_row(row, cols = [annotator, 'Curation']) * row[row_name] / total_n_spans, axis=1).sum()
                score_dict[annotator] = dict(zip(self.calculated_curation_scores, scores.T))
            else:
                raise ValueError('This weighting method is not valid!')
        
        
        return score_dict


    def get_to_curation_score_total(self, weight_by = 'Tokens'):

        """
        Calculates a weighted average of the all the to-curation scores in the DataFrame, averaged over all annotators. Only possible if the score column exists in the DataFrame. 
        If not, first append score with "append_score_to_curation". 

        Parameters
        ----------
        
        weight_by: {'no_weighting', 'Tokens', 'Spans'}
            no_weighting: Calculates simple average over all articles
            Tokens: Weights each article by number of Tokens
            Spans: Weights each article by number of spans 
        

        Returns
        ----------
        out : dict
            Score dict containing the weighted averages for all the to-curation scores defined in "scoring_metrics", weighted over all annotators.
            
        """
        score_dict = {}
        df_annotator = self.df


        if weight_by == 'no_weighting':
            scores = df_annotator.apply(lambda row: to_curation_colums(row).mean(), axis=1).mean()
            return dict(zip(self.calculated_curation_scores, scores.T))

        elif weight_by == 'Tokens':
            total_n_tokens = len(list(chain.from_iterable(df_annotator['Tokens'])))
            scores = df_annotator.apply(lambda row: len(row['Tokens']) * to_curation_colums(row).mean() / total_n_tokens, axis=1).sum()
            return dict(zip(self.calculated_curation_scores, scores.T))

        elif weight_by == 'Spans':
            total_n_spans = df_annotator.apply(lambda row:  _get_span_count_in_row(row, cols = row['Finished_Annotators'] + ['Curation'] * len(row['Finished_Annotators'])), axis=1).sum() #
            scores = df_annotator.apply(lambda row: _get_span_count_in_row(row, cols = row['Finished_Annotators'] + ['Curation']* len(row['Finished_Annotators'])) * to_curation_colums(row).mean() / total_n_spans, axis=1).sum()
            return dict(zip(self.calculated_curation_scores, scores.T))            

        return score_dict

                
    def get_score_spanlist(self, conditional_rep, annotators , scoring_metric, item = None, value = None, weight_by = 'Spans', **optional_tuple_properties):

        """
        Calculates IAA score based on a spanlist. For each article present in the spanlist, the IAA score is calculated if two annotators are found. 
        If only one annotator is found, the article is skipped. The final score is a weighted average {'no_weighting', 'Tokens', 'Spans'} of all the individual article scores. 


        Parameters
        ----------
        conditional_rep: repository

        annotators: {list, string}
            A single annotator or list of annotators that we want to generate a spanlist from

        scoring_metric: string
            Which IAA score to calculate

        item: {'layer', 'type', 'tag'}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        weight_by: {'no_weighting', 'Tokens', 'Spans'}

        optional_tuple_properties: Optional tuple properties for pygamma, see pygamma


        Returns
        ----------
        out: tuple 
            tuple[0]: list, spanlist of all the spans that are matched by the arguments, similar to get_span_list.
            tuple[1]: float, the IAA score for the respective spanlist.


        """

        span_list = self.get_span_list(conditional_rep, annotators, item , value)
 
        if annotators == 'all':
            annotators = self.annotators + ['Curation']
        
        if annotators == 'annotators':
            annotators = self.annotators

        
        if len(span_list) == 0:
            return ValueError('This spanlist is empty')

        repos = list(set([span_.rep for span_ in span_list]))
        total_score = 0
        normalization_count = 0
        
        two_finished_annos = []


        for repo in repos:

            # only continue if there are at least two finnished annotators for this article

            # we can not only just continue if there are two or more annotators for the specific article
            # if we would do so, we would miss a case where one annotator would annotate a part, the other would not, hence the score would be zero
            # but in this case, we would only have one unique annotator

            repo_row = self.df.loc[repo.index_name]

            finished_annotators_repo = repo_row['Finished_Annotators'] # Get the finsihed annotators for the specif repo

            if repo_row['Article_State'] == 'CURATION_FINISHED':
                finished_annotators_repo = finished_annotators_repo + ['Curation'] # if the curation is finished, we can append the annotor 'Curation' to the list of finsihed annotators

            finished_annotators_span_list_repo = [ann for ann in finished_annotators_repo if ann in annotators] # the get all the annotators for this specific rep that are part of the spanlist and a finsihed annotator for this specific rep
  
            if len(finished_annotators_span_list_repo) < 2: # for this repo there are less than two finished annotators in the spanlist
                two_finished_annos.append(False)
                continue
            else:
                two_finished_annos.append(True)


            span_list_repo = [span_ for span_ in span_list if span_.rep == repo]
            

            n_tokens = len([span_.tokens for span_ in span_list_repo])

            
            annos = list(set([spn.annotator for spn in span_list]))
            
            
            if n_tokens == 0:
                raise ValueError(f"zero tokens found for spanlist {span_list} and repo {repo}")

            if weight_by == 'no_weighting':

                total_score += _get_score_article(span_list_repo , scoring_metric, finished_annotators_span_list_repo, **optional_tuple_properties) 
                normalization_count +=1

            elif weight_by == 'Tokens':
                # this is not really correct. We are weighting by the total tokens from the spanlist, but we should weight by all the tokens that are present in this repo
                # difficult to do, either way mistakes are made

                total_score += _get_score_article(span_list_repo , scoring_metric, finished_annotators_span_list_repo, **optional_tuple_properties) * n_tokens
                normalization_count += n_tokens

               
            elif weight_by == 'Spans':
  
                n_spans = len(span_list_repo)
                total_score += _get_score_article(span_list_repo , scoring_metric, finished_annotators_span_list_repo, **optional_tuple_properties) * n_spans
                normalization_count += n_spans


            else: 
                raise ValueError('This weighting method is not valid!')

            if max(two_finished_annos) == False: # In none of the articles that are part of the spanlist contained two finished annotators
                return ValueError(f"No articles with two finished annotators found")

            if normalization_count == 0 and weight_by == 'Spans': # Even though at least one article contained two finsihed annos, the normalization count is zero. 
                return 1                                          # That means that for all articles, both have not annotated a single span, even though there where in the finished annotators. So the score is 1

        return span_list, total_score / normalization_count
    
    

class Sent_Inter_Annotator_Agreement(Sent_Corpus, Inter_Annotator_Agreement):
    pass






