from itertools import chain
import collections
from definitions import df_annotation_marker

class Corpus:
    
    def __init__(self, df):
        """
        
    Corpus Statistics
    Parameters
    ----------
    Dataframe :
        Dataframe where each column represents a article
    
        
        """
        self.df = df

    def _get_iterator_all(self, columns): 

        """
        returns a iterator of all the spans present in the dataframe

        """
        
        if isinstance(columns, str):
    
            return chain.from_iterable(self.df[columns])
            
        else:
            return chain.from_iterable(self.df[columns].stack())
        
        
    def _get_iterator_conditional_rep(self, conditional_rep, columns): 


        """
        returns a iterator of a certain column based on a conditional repository

        """
        
        if isinstance(columns, str):
        
            iterator =  chain.from_iterable(self.df[columns])
           
        else:
            iterator = chain.from_iterable(self.df[columns].stack())

        return [x for x in iterator if type(x)!= str and x.rep.match(conditional_rep) == True]
        
        
    def get_span_list(self, conditional_rep, columns = 'all', item = None, value = None):
        """
        Returns a list of spans based on a conditional repository and can be filtered by a item and corresponding value.

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'type', 'tag'}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """

        if columns == 'all':
            columns = [col for col in self.df.columns[df_annotation_marker-1 :] if 'score' not in col]
        

        if columns == 'annotators':
            columns = [col for col in self.df.columns[df_annotation_marker-1 :] if 'score' not in col and col != 'Curation']
        

        iterator = self._get_iterator_conditional_rep(conditional_rep, columns)
        
        if item == None and value == None:
            return [x for x in iterator if type(x)!= str]
        if item == 'layer':
            return [x for x in iterator if type(x)!= str and x.layer_ == value]
        if item == 'type':
            return [x for x in iterator if type(x)!= str and x.type_ == value]
        if item == 'tag':
            return [x for x in iterator if type(x)!= str and x.tag_ == value]

            
    
    def get_span_count(self, conditional_rep, columns = 'Curation', item = None, value = None):
                   
        """
        Returns count of spans based on a conditional repository and can be filtered by a item and corresponding value.

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'type', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the tags matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.


        """
        return len(self.get_span_list(conditional_rep, columns, item, value))    

    def get_span_distro(self, conditional_rep, columns = 'Curation', item = None, value = None, return_format = 'dict', level = 'character'):
                   
        """
        Returns a distribution of span lenghts based on a span list. 
        The distribution is calculated on token or character level and can be returned as a dict or as a list.

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'type', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        return_format: {'dict', 'list'}

        level: {'character', 'token'}

        """
        
        span_list = self.get_span_list(conditional_rep, columns, item, value)
        
        if level == 'character':
            len_list = [(x.stop - x.start) for x in span_list]
        if level == 'token':
            len_list = [len(x.tokens) for x in span_list]
            
        len_dict = collections.Counter(len_list)
        
        if return_format == 'dict':
            return dict(sorted(len_dict.items(), key=lambda item: item[1], reverse=True))
            
        
        if return_format == 'list':
            distro_list = []
            for i in range(1,max(len_list)+1):
                distro_list.append(len_list.count(i))
            return distro_list
                        

    def get_token_list_from_repository(self, conditional_rep):
                      
        """
        Returns a token list based on conditional repository.

        Parameters
        ----------
        conditional_rep: repository

        """
        token_iterator = self._get_iterator_conditional_rep(conditional_rep, 'Tokens')
        return list(token_iterator)

    def get_token_count_from_repository(self, conditional_rep):

        """
        Returns token count based on conditional repository.

        Parameters
        ----------
        conditional_rep: repository

        """
        return len(self.get_token_list_from_repository(conditional_rep))
        
        


    def get_tokens_from_span_list(self, conditional_rep, columns = 'Curation', item = None, value = None):

        """
        Returns a list of tokens based on a span list. 

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'type', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """

        span_list = self.get_span_list(conditional_rep, columns, item, value)
        return list(chain.from_iterable([x.tokens for x in span_list]))


    def get_token_count_from_span_list(self, conditional_rep, columns = 'Curation', item = None, value = None):

        """
        Returns token count based on a span list. 

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'type', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """
        return len(self.get_tokens_from_span_list(conditional_rep, columns, item, value)) 


    def most_frequent_labeled_tokens(self, conditional_rep, columns = 'Curation', item = None, value = None):

        """
        Returns dict containing the most labeled tokens in descending order based on a span list. 

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'type', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'type' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'type and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """
        span_list = self.get_span_list(conditional_rep, columns, item, value) #get the spanlist of all the span that match search criteria
        token_iterator = chain.from_iterable([x.tokens for x in span_list]) #retrieve all the tokens from the span list and create a iterator (since the list contains sublists)
        token_counter_dict = collections.Counter([x.text for x in token_iterator]) #get a list the text of the token, count the different elements and create a dict
        return dict(sorted(token_counter_dict.items(), key=lambda item: item[1], reverse=True))  #sort the dict by counts


    def get_label_count_per_token_distro(self, conditional_rep, return_format = 'dict'):

        """
        Returns the distribution of label counts per token based on a conditional rep in descending order.
        Can be returned as list or dict.
                
        Parameters
        ----------
        conditional_rep: repository

        return_format: {'dict', 'list'}

        """
        token_iterator = self._get_iterator_conditional_rep(conditional_rep, 'Tokens')
        token_counter_dict = collections.Counter([x.tag_count for x in token_iterator]) #get a list the text of the token, count the different elements and create a dict
        label_counter_list = [x.tag_count for x in token_iterator]
        
        if return_format == 'dict':
            #return dict(sorted(token_counter_dict.items(), key=lambda item: item[1], reverse=True))
            return dict(sorted(token_counter_dict.items()))
        
        if return_format == 'list':
            distro_list = []
            for i in range(0,max(label_counter_list)+1):
                distro_list.append(label_counter_list.count(i))
            return distro_list
    
    def get_tokens_with_label_count(self, conditional_rep, label_count):
    
        """
        Return all the tokens based on a conditional rep that have a specific label_count.
                
        Parameters
        ----------
        conditional_rep: repository

        label_count: int

        """

        token_list = self.get_token_list_from_repository(conditional_rep)
        return [tok for tok in token_list if tok.tag_count == label_count]