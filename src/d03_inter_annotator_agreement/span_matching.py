from pygamma_agreement import Continuum, CombinedCategoricalDissimilarity
from src.experiment_utils.helper_classes import token, span, repository
from pyannote.core import Segment


def create_tuples_pygamma(span_list, **dissimilarity_properties):

    continuum = Continuum()
    for spanlist_span in span_list:
        continuum.add(spanlist_span.annotator, Segment(spanlist_span.start, spanlist_span.stop), spanlist_span.tag)
    dissim = CombinedCategoricalDissimilarity(categories = dissimilarity_properties.get('category_list_matching',continuum.categories), alpha=dissimilarity_properties.get('alpha', 1),   beta=dissimilarity_properties.get('beta', 1), cat_dissimilarity_matrix = dissimilarity_properties.get('cat_dissimilarity_matrix_matching', None))
    best_alignment = continuum.get_best_alignment(dissim)
    
    #now retrieve spantuples
    total_tuples = []

    for un in best_alignment.unitary_alignments: #loop over all the unitary aligments 
        list_tuple = []
        for n_tuple in un._n_tuple: #loop over all the units in unitary aligment          
            if n_tuple[1] == None: #in case the unit is a epty unit, create a None-span
                tuple_span = span(annotator = n_tuple[0])
                
            else: #in case the unit is not empty, find the span in spanlist that corresponds to this unit
                matches = [span_ for span_ in span_list if span_.annotator == n_tuple[0] and span_.tag == n_tuple[1].annotation and span_.start == n_tuple[1].segment.start and span_.stop == n_tuple[1].segment.end]
                if len(matches) != 1:
                    print('Matches:')
                    print(matches)
                    print('-----------')
                    print('Repository for those matches:')
                    print([t.rep for t in matches])
                    print('-----------')
                    print('Total Span List:')
                    print(span_list)
    
                    raise ValueError('More than 1 match found for the above matches in pygamma matching')
                    
                tuple_span = matches[0] #list 'matches' only contains one element
            list_tuple.append(tuple_span)
        total_tuples.append((tuple(list_tuple))) #create tuple out of spanlist and append to total tuples
    return total_tuples

matching_methods = {
    'pygamma': create_tuples_pygamma
}


        