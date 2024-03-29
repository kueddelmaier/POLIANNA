from multiprocessing import Value
import numpy as np
import json
from sortedcontainers import SortedSet
import logging


from pygamma_agreement import Continuum
from pyannote.core import Segment
from pygamma_agreement import CombinedCategoricalDissimilarity, CategoricalDissimilarity 

from itertools import chain
from collections import Counter
from src.d03_inter_annotator_agreement.span_matching import create_tuples_pygamma


# Supress GLPK warnings coming from pygamma
logger = logging.getLogger("cvxpy")
logger.setLevel(logging.CRITICAL)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    
    """
    
    Check if a matrix is symmetric

    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def create_scoring_matrix(tagset_path, soft_dissimilarity_penality = 0.5, soft_layer_dissimilarity = False, soft_tagset_dissimilarity = False):
    """
    TODO: why not traverse json directly, why create a hirarchical list?


    Creates a scoring matrix for the pygamma agreement based on a tagset. The tagset needs to be a json with a hirarchical structure. 
    For an example, see "tag_set.json"

    Missmatches between the same category are penalized with the soft_dissimilarity_penality, all other missmatches are penalized with 1

    """
    assert 0 <= soft_dissimilarity_penality <= 1, "soft_dissimilarity_penality should be a value between 0 and 1"

    if soft_layer_dissimilarity == soft_tagset_dissimilarity == True:
        raise ValueError('Soft_layer_dissimilarity and soft_tagset_dissimilarity can not both be True!')
   
    with open(tagset_path) as json_file:
        try:
            data = json.load(json_file)
        except IOError as e:
            raise e

    #create a hirarchical list structure of all the tags

    matrix_list = []


    if soft_layer_dissimilarity: #if soft_layer_dissimilarity = True, all the missmatches within the same layer are penalized with the soft_dissimilarity_penality
        for layer in data['layers']:
            layer_tags = []
            for tagset in layer['tagsets']:
                for tag in tagset['tags']:
                    layer_tags.append(tag['tag_name'])

            matrix_list.append(layer_tags)

    elif soft_tagset_dissimilarity: #if soft_tagset_dissimilarity = True, all the missmatches within the same tagset are penalized with the soft_dissimilarity_penality
        for layer in data['layers']:
            for tagset in layer['tagsets']:
                tagset_tags = []
                for tag in tagset['tags']:
                    tagset_tags.append(tag['tag_name'])
                matrix_list.append(tagset_tags)

    else:
        for layer in data['layers']:
            for tagset in layer['tagsets']:
                for tag in tagset['tags']:
                    matrix_list.append([tag['tag_name']])
        

    matrix_flat = list(chain.from_iterable(matrix_list)) #transform to 1dim list
    matrix_flat.append('')    #handles empty annotations
    matrix_flat_ordered = SortedSet(matrix_flat) #transform to otrdered set since this is required in the new version of pygamma-agreement
    matrix_array = np.ones((len(matrix_flat_ordered), len(matrix_flat_ordered))) #by default, the penalty of a missmatch between two different tags is 1

    if soft_layer_dissimilarity or soft_tagset_dissimilarity:
        for tag in matrix_flat_ordered: # iterate over all the tags
            for sublist in matrix_list: 
                if tag in sublist:
                    matrix_array[matrix_flat_ordered.index(tag),[matrix_flat_ordered.index(sub_list_tag) for sub_list_tag in sublist]] = soft_dissimilarity_penality #all the missmatches for the same catergory (layer or tagset) get the soft penalty

    np.fill_diagonal(matrix_array,0) # the penalty for the the diagonal (no missmatch as true label = predicted label) is zero

    #check if matrix symmetric
    assert check_symmetric(matrix_array), "Matrix not symmetric"

    return matrix_flat_ordered, matrix_array


def create_scoring_matrix_old(tagset_path, soft_dissimilarity_penality = 0.5, soft_layer_dissimilarity = False, soft_tagset_dissimilarity = False):
    """
    TODO: why not traverse json directly, why create a hirarchical list?
    Creates a scoring matrix for the pygamma agreement based on a tagset. The tagset needs to be a json with a hirarchical structure. 
    For an example, see "tag_set.json"
    Missmatches between the same category are penalized with the soft_dissimilarity_penality, all other missmatches are penalized with 1
    """
    assert 0 <= soft_dissimilarity_penality <= 1, "soft_dissimilarity_penality should be a value between 0 and 1"

    if soft_layer_dissimilarity == soft_tagset_dissimilarity:
        raise ValueError('Soft_layer_dissimilarity and soft_tagset_dissimilarity need to be different!')
   
    with open(tagset_path) as json_file:
        try:
            data = json.load(json_file)
        except IOError as e:
            raise e

    #create a hirarchical list structure of all the tags

    matrix_list = []


    if soft_layer_dissimilarity: #if soft_layer_dissimilarity = True, all the missmatches within the same layer are penalized with the soft_dissimilarity_penality
        for layer in data['layers']:
            layer_tags = []
            for tagset in layer['tagsets']:
                for tag in tagset['tags']:
                    layer_tags.append(tag['tag_name'])

            matrix_list.append(layer_tags)

    if soft_tagset_dissimilarity: #if soft_layer_dissimilarity = True, all the missmatches within the same tagset are penalized with the soft_dissimilarity_penality
        for layer in data['layers']:
            for tagset in layer['tagsets']:
                tagset_tags = []
                for tag in tagset['tags']:
                    tagset_tags.append(tag['tag_name'])
                matrix_list.append(tagset_tags)
    

    matrix_flat = list(chain.from_iterable(matrix_list))
    matrix_flat.append('')    #handles empty annotations
    matrix_array = np.ones((len(matrix_flat), len(matrix_flat))) #by default, the penalty of a missmatch between two different tags is 1


    for tag in matrix_flat: # iterate over all the tags
        for sublist in matrix_list: 
            if tag in sublist:
                matrix_array[matrix_flat.index(tag),[matrix_flat.index(sub_list_tag) for sub_list_tag in sublist]] = soft_dissimilarity_penality #all the missmatches for the same catergory (layer or tagset) get the soft penalty

    np.fill_diagonal(matrix_array,0) # the penalty for the the diagonal (no missmatch as true label = predicted label) is zero


    return matrix_flat, matrix_array

    

def unified_gamma(span_list, **dissimilarity_properties):
    from pygamma_agreement import PrecomputedCategoricalDissimilarity, PositionalSporadicDissimilarity

    disable_warnings = True
    if disable_warnings:
        logging.getLogger('pygamma_agreement').setLevel(logging.CRITICAL)

    continuum = Continuum()
    for spanlist_span in span_list:
        continuum.add(spanlist_span.annotator, Segment(spanlist_span.start, spanlist_span.stop), spanlist_span.tag)
    cat_dissim = PrecomputedCategoricalDissimilarity(categories = dissimilarity_properties.get('category_list',continuum.categories), matrix = dissimilarity_properties.get('cat_dissimilarity_matrix', None))
    pos_dissim = PositionalSporadicDissimilarity()
    dissimilarity = CombinedCategoricalDissimilarity(cat_dissim=cat_dissim, pos_dissim = pos_dissim, alpha=dissimilarity_properties.get('alpha', 1), beta=dissimilarity_properties.get('beta', 1), delta_empty=1.0)
    try: 
     
        gamma_results = continuum.compute_gamma(dissimilarity, soft = dissimilarity_properties.get('soft_pygamma',False))
    except TypeError:

        #means that the linear solver couldn't find an alignment with minimal disorder
        #can be due to too many unitary aligments
        #in this case, split the span_list in half
        #make shure that no spans are overlapping the treshhold since in this case we don't know where this span belongs to

        treshhold = np.median([(span_.start + span_.stop)/ 2 for span_ in span_list])
        spans_overlapping_treshhold = [span_ for span_ in span_list if span_.start < treshhold and span_.stop > treshhold]

        i = 1
        while len(spans_overlapping_treshhold) != 0: # if there is an overlapping span, create a alternating sequence to modify treshhold until there is no overlapping span
            treshhold = treshhold + (i * ((-1) ** (i-1))) 
            spans_overlapping_treshhold = [span_ for span_ in span_list if span_.start <= treshhold and span_.stop >= treshhold]
            i +=1
        
        span_list_1 = [span_ for span_ in span_list if span_.stop < treshhold]
        span_list_2 = [span_ for span_ in span_list if span_.start > treshhold]

        if len(span_list_1) + len(span_list_2) != len(span_list):
            raise ValueError('The splittet spanlist have not the same length as the initial spanlist')
        
        gamma_score_1 = unified_gamma(span_list_1, **dissimilarity_properties)
        gamma_score_2 = unified_gamma(span_list_2, **dissimilarity_properties)

        return (gamma_score_1 * len(span_list_1) + gamma_score_2 * len(span_list_2)) / len(span_list)
        
    return gamma_results.gamma

def unified_gamma_old(span_list, **dissimilarity_properties):
    
    continuum = Continuum()
    for spanlist_span in span_list:
        continuum.add(spanlist_span.annotator, Segment(spanlist_span.start, spanlist_span.stop), spanlist_span.tag)
    dissim = CombinedCategoricalDissimilarity(categories = dissimilarity_properties.get('category_list',continuum.categories), alpha=dissimilarity_properties.get('alpha', 1),   beta=dissimilarity_properties.get('beta', 1), cat_dissimilarity_matrix = dissimilarity_properties.get('cat_dissimilarity_matrix', None))
    try: 
        gamma_results = continuum.compute_gamma(dissim)
    except TypeError:

        #means that the linear solver couldn't find an alignment with minimal disorder
        #can be due to too many unitary aligments
        #in this case, split the span_list in half
        #make shure that no spans are overlapping the treshhold since in this case we don't know where this span belongs to

        treshhold = np.median([(span_.start + span_.stop)/ 2 for span_ in span_list])
        spans_overlapping_treshhold = [span_ for span_ in span_list if span_.start < treshhold and span_.stop > treshhold]

        i = 1
        while len(spans_overlapping_treshhold) != 0: # if there is an overlapping span, create a alternating sequence to modify treshhold until there is no overlapping span
            treshhold = treshhold + (i * ((-1) ** (i-1))) 
            spans_overlapping_treshhold = [span_ for span_ in span_list if span_.start <= treshhold and span_.stop >= treshhold]
            i +=1
        
        span_list_1 = [span_ for span_ in span_list if span_.stop < treshhold]
        span_list_2 = [span_ for span_ in span_list if span_.start > treshhold]

        if len(span_list_1) + len(span_list_2) != len(span_list):
            raise ValueError('The splittet spanlist have not the same length as the initial spanlist')
        
        gamma_score_1 = unified_gamma(span_list_1, **dissimilarity_properties)
        gamma_score_2 = unified_gamma(span_list_2, **dissimilarity_properties)

        return (gamma_score_1 * len(span_list_1) + gamma_score_2 * len(span_list_2)) / len(span_list)
        
    return gamma_results.gamma
    
def f1(cor, act, pos):
    precision = cor/act
    recall = cor/pos
    if precision + recall == 0: # avoid dividing by 0
        return 0
    else:
        return (2*precision*recall)/(precision+recall)

def f1_exact_pygamma(span_list_annotator_pair, annotator_pair, **optional_tuple_properties):

    tuple_list = create_tuples_pygamma(span_list_annotator_pair, **optional_tuple_properties)
    # for a given tuple, treat tuple[0] as prediciton and tuple[1] as gold standart since the score is symmetric
    # note that all the spans of a certain annotator that arre missing in the counterpart are matched to a "None" Tag

    exact = sum([n_tuple[0].exact_match(n_tuple[1]) for n_tuple in tuple_list if n_tuple[0].tag != None and n_tuple[1].tag != None])
    act = len([n_tuple[0] for n_tuple in tuple_list if n_tuple[0].tag != None ]) # equal to all the spans of the prediction = tp + fp
    pos = len([n_tuple[1] for n_tuple in tuple_list if n_tuple[1].tag != None ]) # equal to all the spans of the gold standart = tp + fn
    return f1(exact, act, pos)

def f1_partial_pygamma(span_list_annotator_pair, annotator_pair, **optional_tuple_properties):

    tuple_list = create_tuples_pygamma(span_list_annotator_pair, **optional_tuple_properties)
    # for a given tuple, treat tuple[0] as prediciton and tuple[1] as gold standart since the score is symmetric
    # note that all the spans of a certain annotator that are missing in the counterpart are matched to a "None" Tag

    partial = sum([n_tuple[0].partial_match(n_tuple[1]) for n_tuple in tuple_list if n_tuple[0].tag != None and n_tuple[1].tag != None])
    act = len([n_tuple[0] for n_tuple in tuple_list if n_tuple[0].tag != None ]) # equal to all the spans of the prediction = tp + fp
    pos = len([n_tuple[1] for n_tuple in tuple_list if n_tuple[1].tag != None ]) # equal to all the spans of the gold standart = tp + fn

    return f1(partial, act, pos)

def f1_tokenwise_pygamma(span_list_annotator_pair, annotator_pair, **optional_tuple_properties):

    tuple_list = create_tuples_pygamma(span_list_annotator_pair, **optional_tuple_properties)

    # calculate the mean of all the individual tokenwise f1-scores of all spans

    return sum([n_tuple[0].tokenwise_f1_score(n_tuple[1]) for n_tuple in tuple_list])/len(tuple_list)




def f1_positional_article_tokenwise(span_list, annotator_pair, **dissimilarity_properties):


    curation_spans = [cur_span for cur_span in span_list if cur_span.annotator == annotator_pair[0]]
    annotator_spans = [ann_span for ann_span in span_list if ann_span.annotator == annotator_pair[1]]


    curation_tokens = list(chain.from_iterable([cur_span.tokens for cur_span in curation_spans]))
    annotator_tokens = list(chain.from_iterable([ann_span.tokens for ann_span in annotator_spans]))


    tp = len(list((Counter(annotator_tokens) & Counter(curation_tokens)).elements())) #calculates the intersection including doublicates (some tokens are labeled multiple times), which is the lowest count found in either list for each element when you take the intersection

    act = len(annotator_tokens) # equal to all the spans of the prediction = tp + fp
    pos = len(curation_tokens) # equal to all the spans of the gold standart = tp + fn

    #return f1(tp, act, pos)
    return f1(tp, act, pos)


def f1_article_tokenwise(span_list, annotator_pair, **dissimilarity_properties):
    
    f1_score = np.zeros(2)
    for i in range(0,2):
        curation_spans = [cur_span for cur_span in span_list if cur_span.annotator == annotator_pair[i]]
        annotator_spans = [ann_span for ann_span in span_list if ann_span.annotator == annotator_pair[(i+1)%2]]

        curation_tokens = list(chain.from_iterable([cur_span.tokens for cur_span in curation_spans]))
        annotator_tokens = list(chain.from_iterable([ann_span.tokens for ann_span in annotator_spans]))

        if len(curation_spans) + len(annotator_spans) != len(span_list):
            raise ValueError('Curation spans, annotations spans and span_list do not match in length')
            
        tp = 0
    
        for ann_span in annotator_spans:
            for ann_tok in ann_span.tokens:
                tok_matchings = [cur_tok for cur_tok in curation_tokens if ann_tok == cur_tok and ann_span.tag in cur_tok.get_token_tags(annotators = annotator_pair[i])]
                if len(tok_matchings) >= 1:
                    tp += 1

        act = len(annotator_tokens) # equal to all the spans of the prediction = tp + fp
        pos = len(curation_tokens) # equal to all the spans of the gold standart = tp + fn

        #return f1(tp, act, pos)
        f1_score[i] = f1(tp, act, pos)
    return np.mean(f1_score)


def f1_exact(span_list, annotator_pair, **dissimilarity_properties):

   
    # Could be written much more compact, since much of the code is redundant
    # But avoided for better understanding

    f1_score = np.zeros(2)

    for i in range(0,2):
        curation_spans = [cur_span for cur_span in span_list if cur_span.annotator == annotator_pair[i]]
        annotator_spans = [ann_span for ann_span in span_list if ann_span.annotator == annotator_pair[(i+1)%2]]

        if len(curation_spans) + len(annotator_spans) != len(span_list):
            raise ValueError('Curation spans, annotations spans and span_list do not match in length')

        tp = 0
        
        for ann_span in annotator_spans:
            # all the spans that overlap and have the same ## To Do : use f1 function as is f1_tokenwise_article ##tag
            span_matchings = [cur_span for cur_span in curation_spans if cur_span.start == ann_span.start and cur_span.stop == ann_span.stop and cur_span.tag == ann_span.tag]
            
            if len(span_matchings) >= 1:
                tp += 1
        act = len(annotator_spans) # equal to all the spans of the prediction = tp + fp
        pos = len(curation_spans) # equal to all the spans of the gold standart = tp + fn

        #return f1(tp, act, pos)
        f1_score[i] = f1(tp, act, pos)
    return np.mean(f1_score)




def f1_heuristic(span_list, annotator_pair, **dissimilarity_properties): 


    f1_score = np.zeros(2)

    for i in range(0,2):
        curation_spans = [cur_span for cur_span in span_list if cur_span.annotator == annotator_pair[i]]
        annotator_spans = [ann_span for ann_span in span_list if ann_span.annotator == annotator_pair[(i+1)%2]]

        if len(curation_spans) + len(annotator_spans) != len(span_list):
            raise ValueError('Curation spans, annotations spans and span_list do not match in length')
        
        tp = 0

        
        for ann_span in annotator_spans:
            # all the spans that overlap and have the same tag
            span_matchings = [cur_span for cur_span in curation_spans if cur_span.start < ann_span.stop and cur_span.stop > ann_span.start and cur_span.tag == ann_span.tag ]
            
         
            if len(span_matchings) >= 1:
                tp += 1
        act = len(annotator_spans) # equal to all the spans of the prediction = tp + fp
        pos = len(curation_spans) # equal to all the spans of the gold standart = tp + fn

        #return f1(tp, act, pos)
        f1_score[i] = f1(tp, act, pos)
    return np.mean(f1_score)



scoring_metrics = {
    'f1_heuristic': f1_heuristic,
    'f1_exact': f1_exact,
    'f1_tokenwise': f1_article_tokenwise,
    'f1_positional': f1_positional_article_tokenwise,
    'f1_exact_pygamma': f1_exact_pygamma,
    'f1_partial_pygamma': f1_partial_pygamma,
    'f1_tokenwise_pygamma': f1_tokenwise_pygamma,

}
