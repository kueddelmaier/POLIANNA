from multiprocessing import Value
import numpy as np
import json


from pygamma_agreement import Continuum
from pyannote.core import Segment
from pygamma_agreement import CombinedCategoricalDissimilarity
from itertools import chain


def create_scoring_matrix(tagset_path, soft_dissimilarity_penality = 0.5, soft_layer_dissimilarity = False, soft_tagset_dissimilarity = False):
    if soft_layer_dissimilarity == soft_tagset_dissimilarity:
        raise ValueError('Soft_layer_dissimilarity and soft_tagset_dissimilarity need to be different!')
   
    with open(tagset_path) as json_file:
        try:
            data = json.load(json_file)
        except IOError as e:
            raise e

    matrix_list = []


    if soft_layer_dissimilarity:
        for layer in data['layers']:
            layer_tags = []
            for tagset in layer['tagsets']:
                for tag in tagset['tags']:
                    layer_tags.append(tag['tag_name'])

            matrix_list.append(layer_tags)

    if soft_tagset_dissimilarity:
        for layer in data['layers']:
            for tagset in layer['tagsets']:
                tagset_tags = []
                for tag in tagset['tags']:
                    tagset_tags.append(tag['tag_name'])
                matrix_list.append(tagset_tags)
    

    matrix_flat = list(chain.from_iterable(matrix_list))
    matrix_flat.append('')    #handles empty anotations
    matrix_array = np.ones((len(matrix_flat), len(matrix_flat)))
    

    for tag in matrix_flat:
        for sublist in matrix_list:
            if tag in sublist:
                matrix_array[matrix_flat.index(tag),[matrix_flat.index(sub_list_tag) for sub_list_tag in sublist]] = soft_dissimilarity_penality

    np.fill_diagonal(matrix_array,0)

    return matrix_flat, matrix_array

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
    

def unified_gamma(span_list, **dissimilarity_properties):

    continuum = Continuum()
    for spanlist_span in span_list:
        continuum.add(spanlist_span.annotator, Segment(spanlist_span.start, spanlist_span.stop), spanlist_span.tag_)
    dissim = CombinedCategoricalDissimilarity(categories = dissimilarity_properties.get('category_list',continuum.categories), alpha=dissimilarity_properties.get('alpha', 1),   beta=dissimilarity_properties.get('beta', 1), cat_dissimilarity_matrix = dissimilarity_properties.get('cat_dissimilarity_matrix', None))
    try: 
        gamma_results = continuum.compute_gamma(dissim)
    except TypeError:

        #means that the linear solver couldn't find an alignment with minimal disorder
        #can be due to too many unitary aligments
        #in this case, split the span_list in half

        treshhold = np.median([(span_.start + span_.stop)/ 2 for span_ in span_list])
        spans_overlapping_treshhold = [span_ for span_ in span_list if span_.start < treshhold and span_.stop > treshhold]

        i = 1
        while len(spans_overlapping_treshhold) != 0:
            treshhold = treshhold + (i * (-1) ** (i-1))
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
    if precision + recall == 0:
        return 0
    else:
        return (2*precision*recall)/(precision+recall)

def f1_exact(tuple_list):
    #for a given tuple, treat tuple[0] as prediciton and tuple[1] as gold standart
    exact = sum([n_tuple[0].exact_match(n_tuple[1]) for n_tuple in tuple_list if n_tuple[0].tag_ != None and n_tuple[1].tag_ != None])
    act = len([n_tuple[0] for n_tuple in tuple_list if n_tuple[0].tag_ != None ])
    pos = len([n_tuple[1] for n_tuple in tuple_list if n_tuple[1].tag_ != None ])

    return f1(exact, act, pos)

def f1_partial(tuple_list):
    #for a given tuple, treat tuple[0] as prediciton and tuple[1] as gold standart
    partial = sum([n_tuple[0].partial_match(n_tuple[1]) for n_tuple in tuple_list if n_tuple[0].tag_ != None and n_tuple[1].tag_ != None])
    act = len([n_tuple[0] for n_tuple in tuple_list if n_tuple[0].tag_ != None ])
    pos = len([n_tuple[1] for n_tuple in tuple_list if n_tuple[1].tag_ != None ])

    return f1(partial, act, pos)

def f1_tokenwise(tuple_list):
    return sum([n_tuple[0].tokenwise_f1_score(n_tuple[1]) for n_tuple in tuple_list])/len(tuple_list)

def f1_heuristic(span_list, annotator_pair):

    curation_spans = [cur_span for cur_span in span_list if cur_span.annotator == annotator_pair[0]]
    annotator_spans = [ann_span for ann_span in span_list if ann_span.annotator == annotator_pair[1]]


    if len(curation_spans) + len(annotator_spans) != len(span_list):
        raise ValueError('Curation spans, annotations spans and span_list do not match in length')
        
    tp_curation = 0
    tp_annotator = 0


    for cur_span in curation_spans:
        span_matchings = [ann_span for ann_span in annotator_spans if ann_span.start < cur_span.stop and ann_span.stop > cur_span.start and ann_span.tag_ == cur_span.tag_]

        if len(span_matchings) >= 1:
            tp_curation += 1
    
    for ann_span in annotator_spans:
        span_matchings = [cur_span for cur_span in curation_spans if cur_span.start < ann_span.stop and cur_span.stop > ann_span.start and cur_span.tag_ == ann_span.tag_ ]
        
        if len(span_matchings) >= 1:
            tp_annotator += 1

    precision_cur = tp_curation / (tp_curation + len(annotator_spans) - tp_annotator)
    recall_cur = tp_curation / (tp_curation + len(curation_spans) - tp_curation)

    precision_ann = tp_annotator / (tp_annotator + len(curation_spans) - tp_curation)
    recall_ann = tp_annotator / (tp_annotator + len(annotator_spans) - tp_annotator)

   

    if precision_cur + recall_cur == 0:
        f1_cur = 0
    else:
        f1_cur = (2*precision_cur*recall_cur)/(precision_cur + recall_cur) 


    if precision_ann + recall_ann == 0:
        f1_ann = 0
    else:
        f1_ann = (2*precision_ann*recall_ann)/(precision_ann+recall_ann)

    return (f1_cur + f1_ann) / 2



scoring_metrics = {
    'f1_exact': f1_exact,
    'f1_partial': f1_partial,
    'f1_tokenwise': f1_tokenwise,
}





