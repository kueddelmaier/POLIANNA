from itertools import chain
import collections


from definitions import df_annotation_marker, ROOT_DIR
import pandas as pd
import copy
import os
import json


def add_sent(obj, sentence):
    obj.rep.sentence = sentence
    return obj


class Corpus:
    
    def __init__(self, df, front_and_whereas = False):
        """
        
        Corpus Statistics
        Parameters
        ----------
        df :
            Dataframe where each column represents a article
        
        """
        self.df = copy.deepcopy(df)
        self.annotators = [col for col in self.df.columns[df_annotation_marker-1 :] if col != 'Curation'] 
        self.finished_annotators = list(set(chain.from_iterable(self.df['Finished_Annotators'])))

        

        if front_and_whereas == False:
            for matching_string in ['front', 'Whereas']:
                self.df.drop(self.df.filter(like=matching_string, axis=0).index, inplace=True)
        
        

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
        
        
    def get_span_list(self, conditional_rep, annotators = 'Curation', item = None, value = None):
        """
        Returns a list of spans based on a conditional repository and can be filtered by a item and corresponding value.

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'feature', 'tag'}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """

        if annotators == 'all':
            annotators = self.annotators + ['Curation']
        

        if annotators == 'annotators':
            annotators = self.annotators
        

        iterator = self._get_iterator_conditional_rep(conditional_rep, annotators)
        
        if item == None and value == None:
            return [x for x in iterator if type(x)!= str]
        if item == 'layer':
            return [x for x in iterator if type(x)!= str and x.layer == value]
        if item == 'feature':
            return [x for x in iterator if type(x)!= str and x.feature == value]
        if item == 'tag':
            return [x for x in iterator if type(x)!= str and x.tag == value]
            
    
    def get_span_count(self, conditional_rep, annotators = 'Curation', item = None, value = None):
                   
        """
        Returns count of spans based on a conditional repository and can be filtered by a item and corresponding value.

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'feature', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the tags matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """
        return len(self.get_span_list(conditional_rep, annotators, item, value))    

    def get_span_distro(self, conditional_rep, annotators = 'Curation', item = None, value = None, return_format = 'dict', level = 'character'):
                   
        """
        Returns a distribution of span lenghts based on a span list. 
        The distribution is calculated on token or character level and can be returned as a dict or as a list.

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'feature', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        return_format: {'dict', 'list'}

        level: {'character', 'token'}

        """
        
        span_list = self.get_span_list(conditional_rep, annotators, item, value)
        
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
        
        

    def get_tokens_from_span_list(self, conditional_rep, annotators = 'Curation', item = None, value = None):

        """
        Returns a set of tokens based on a span list. 

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'feature', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """

        span_list = self.get_span_list(conditional_rep, annotators, item, value)
        return list(set(chain.from_iterable([x.tokens for x in span_list])))


    def get_token_count_from_span_list(self, conditional_rep, annotators = 'Curation', item = None, value = None):

        """
        Returns token count based on a span list. 

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'feature', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """
        return len(self.get_tokens_from_span_list(conditional_rep, annotators, item, value)) 


    def most_frequent_labeled_tokens(self, conditional_rep, annotators = 'Curation', item = None, value = None):

        """
        Returns dict containing the most labeled tokens in descending order based on a span list. 

        Parameters
        ----------
        conditional_rep: repository

        item: {'layer', 'feature', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.

        """
        span_list = self.get_span_list(conditional_rep, annotators, item, value) #get the spanlist of all the span that match search criteria
        token_iterator = chain.from_iterable([x.tokens for x in span_list]) #retrieve all the tokens from the span list and create a iterator (since the list contains sublists)
        token_counter_dict = collections.Counter([x.text for x in token_iterator]) #get a list the text of the token, count the different elements and create a dict
        return dict(sorted(token_counter_dict.items(), key=lambda item: item[1], reverse=True))  #sort the dict by counts

    def most_frequent_labeled_spans(self, conditional_rep, annotators = 'Curation', item = None, value = None):
    
        """
        Returns dict containing the most labeled spans in descending order based on a span list. 
        Parameters
        ----------
        conditional_rep: repository
        item: {'layer', 'feature', 'tag}, optional
            If item and value are not specified, no filter is applied and thus all the spans matching the conditional rep are returned.

        value: string, optional
            The value corresponding to the desired 'layer', 'feature' or 'tag'.
            E.g item = 'layer' and value = 'Policydesigncharacteristics' or item = 'feature and value ='Compliance' and item = 'tag' and value = 'Form_monitoring'.
        """
        span_list = self.get_span_list(conditional_rep, annotators, item, value) #get the spanlist of all the span that match search criteria
        span_iterator = [x.text for x in span_list] #retrieve all the span text from the span list
        span_counter_dict = collections.Counter(span_iterator) # count the different elements (spans) and create a dict
        return dict(sorted(span_counter_dict.items(), key=lambda item: item[1], reverse=True))  #sort the dict by counts


    def get_label_count_per_token_distro(self, conditional_rep, annotators = 'Curation', return_format = 'dict'):

        """
        Returns the distribution of label counts per token based on a conditional rep in descending order.
        Can be returned as list or dict.
                
        Parameters
        ----------
        conditional_rep: repository
        annotators: {'Curation', 'Annotation', 'All'}, optional
        return_format: {'dict', 'list'}, optional

        """
        token_iterator = self._get_iterator_conditional_rep(conditional_rep, 'Tokens')
        
        if return_format == 'dict':
            #return dict(sorted(token_counter_dict.items(), key=lambda item: item[1], reverse=True))
            token_counter_dict = collections.Counter([x.get_span_count(annotators = annotators) for x in token_iterator]) #get a list the text of the token, count the different elements and create a dict
            return dict(sorted(token_counter_dict.items()))
        
        if return_format == 'list':
            label_counter_list = [x.get_span_count(annotators = annotators) for x in token_iterator]
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
        return [tok for tok in token_list if tok.get_span_count() == label_count]
    
    def keep_only_finished_articles(self):

        """
        Keeps only the articles whith article_state 'Curation Finished' and at least two annotators.

        All the other articles are stored in a second dataframe 'df_non_curated'-
                
        Parameters
        ----------
        None

        """
        self.df_non_curated = self.df[self.df['Article_State'] !='CURATION_FINISHED']
        self.df = self.df[self.df.apply(lambda x: len(x['Finished_Annotators']) >=2 and x['Article_State'] =='CURATION_FINISHED',axis=1)]
 

    def drop_articles_based_on_string(self, matching_strings):

        """
        Drops articles matching the string or the strings given in 'matching_string. 
        

        Parameters
        ----------
        matching_strings: string or string list

        """
        if isinstance(matching_strings, str):
            self.df.drop(self.df.filter(like=matching_strings, axis=0).index, inplace=True)

        elif isinstance(matching_strings, list):
            for matching_string in matching_strings:
                self.df.drop(self.df.filter(like=matching_string, axis=0).index, inplace=True)
    
        else:
            raise ValueError('The argument string should either be a string or a list')
        
    def del_span(self, span_id, repo):

            
        target = [x for x in self.df.loc[repo.index_name]['Curation'] if x.span_id == span_id]

        if len(target) == 0:
            raise ValueError('element not found')
            
        elif len(target) > 1:
            raise ValueError('more than one element found')
        else:
            self.df.loc[repo.index_name]['Curation'].remove(target[0])

    
    def export_to_jsonl(self, metadata_path, save_path = 'data/d03_processed_to_jsonl'):

        """

        Exports corpus as jsonl format.

        Parameters
        ----------
        metadata_path: string
            Path to the metadata dataframe. Contains metadata about the policies

        """
        import jsonlines

        save_path = os.path.join(ROOT_DIR, save_path)
        os.makedirs(save_path, exist_ok = True)

        metadata_df = pd.read_csv(metadata_path, index_col=0)  

        for row in self.df.itertuples():
            
            if row.Article_State != 'CURATION_FINISHED':
                continue
            
            folder_name = row.Index
            policy_name = '_'.join(folder_name.split('_')[0:2])
            
            folder_path = os.path.join(save_path, folder_name)
            os.makedirs(folder_path, exist_ok = True)
            
            ## find metadata in policy dataframe
            
            policy_metadata_col =  metadata_df[metadata_df['Filename'] == policy_name]
            
            assert policy_metadata_col.shape[0] == 1, f"more than one column found for {folder_name} namely{metadata_df[metadata_df['Filename'] == policy_name]['Filename']} "
            policy_metadata_col = policy_metadata_col.iloc[0] 
            metadata_dict = {}
            metadata_dict["Titel"] = folder_name
            metadata_dict["CELEX_Number"] = policy_metadata_col["CELEX number"]
            metadata_dict["ELI"] = policy_metadata_col["ELI"]
            metadata_dict["Annotators"] = row.Finished_Annotators
            

            
            metadata_json = json.dumps(metadata_dict, indent = 2) 
            
            with open(os.path.join(folder_path, 'Policy_Info.json'), "w") as outfile:
                outfile.write(metadata_json)
            
                        
            #text
            f= open(os.path.join(folder_path, 'Raw_Text.txt'),"w")
            f.write(row.Text)
            f.close()


            # Tokens
            token_list = [tok.to_json() for tok in row.Tokens]
            
            with jsonlines.open(os.path.join(folder_path, 'Tokens.jsonl'), 'w') as writer:
                writer.write_all(token_list)


            # Curation
            cur_span_list = [span_.to_json() for span_ in row.Curation]

            with jsonlines.open(os.path.join(folder_path, 'Curated_Annotations.jsonl'), 'w') as writer:
                writer.write_all(cur_span_list)


            # Annotations
            coder_annotation_folder = os.path.join(folder_path, 'coder_annotations')
            os.makedirs(coder_annotation_folder, exist_ok = True)

            for ann in row.Finished_Annotators:
                ann_span_list  = [span_.to_json() for span_ in getattr(row, ann)]

                with jsonlines.open(os.path.join(coder_annotation_folder, 'Annotations_{}.jsonl'.format(ann)), 'w') as writer:
                    writer.write_all(ann_span_list)
                


    def export_to_json(self, metadata_path, save_path = 'data/d03_processed_to_json'):

        """

        Exports corpus as json format.

        Parameters
        ----------
        metadata_path: string
            Path to the metadata dataframe. Contains metadata about the policies

        """
        save_path = os.path.join(ROOT_DIR, save_path)
        os.makedirs(save_path, exist_ok = True)

        metadata_df = pd.read_csv(metadata_path, index_col=0)  


        for row in self.df.itertuples():
            
            if row.Article_State != 'CURATION_FINISHED':
                continue
            
            folder_name = row.Index
            policy_name = '_'.join(folder_name.split('_')[0:2])
            
            folder_path = os.path.join(save_path, folder_name)
            os.makedirs(folder_path, exist_ok = True)
            
            ## find metadata in policy dataframe
            
            policy_metadata_col =  metadata_df[metadata_df['Filename'] == policy_name]
            
            assert policy_metadata_col.shape[0] == 1, f"more than one column found for {folder_name} namely{metadata_df[metadata_df['Filename'] == policy_name]['Filename']} "
            policy_metadata_col = policy_metadata_col.iloc[0] 
            metadata_dict = {}
            metadata_dict["Titel"] = folder_name
            metadata_dict["CELEX_Number"] = policy_metadata_col["CELEX number"]
            metadata_dict["ELI"] = policy_metadata_col["ELI"]
            metadata_dict["Annotators"] = row.Finished_Annotators
            

            
            metadata_json = json.dumps(metadata_dict, indent = 2) 
           
            with open(os.path.join(folder_path, 'Policy_Info.json'), "w") as outfile:
                outfile.write(metadata_json)
            
                        
            #text
            f= open(os.path.join(folder_path, 'Raw_Text.txt'),"w")
            f.write(row.Text)
            f.close()
            
            # Tokens
            token_list = [tok.to_json() for tok in row.Tokens]
            json_tokens = json.dumps(token_list, indent = 2)
            
            with open(os.path.join(folder_path, 'Tokens.json'), "w") as outfile:
                outfile.write(json_tokens)
                

            # Curation
            cur_span_list = [span_.to_json() for span_ in row.Curation]
            json_cur_spans = json.dumps(cur_span_list, indent = 2)
            
            with open(os.path.join(folder_path, 'Curated_Annotations.json'), "w") as outfile:
                outfile.write(json_cur_spans)
                
            # Annotations


            annotations_json = {}
            for ann in row.Finished_Annotators:
                
                annotations_json[ann]  = [span_.to_json() for span_ in getattr(row, ann)]
                
                #annotations_json = [span_ for span_ in gindent = 2etattr(row, ann)]
                
            json_ann_spans = json.dumps(annotations_json, indent = 2)
            
            with open(os.path.join(folder_path, 'Coder_Annotations.json'), "w") as outfile:
                outfile.write(json_ann_spans)


class Sent_Corpus(Corpus):

    """
    [Experimental]

    Sentence based corpus. Every sentence is a row in the dataframe
    """

    def __init__(self, stat_df, front_and_whereas = False):


        global_index = 0

        cols = list(stat_df.columns)
        cols.append("Sentence_Start")
        cols.append("Sentence_Stop")

        self.df = pd.DataFrame(columns = cols)
        self.df = self.df.rename(columns={'Policy':'Sentence'})

        from nltk.data import load
        language="english"

        tokenizer = load(f"tokenizers/punkt/{language}.pickle")

        annotators = stat_df.columns[df_annotation_marker:]

        ## change: also add correct sentence for all the tags involved,
        #finish add sentence method

        for row_index, stat_df_row in stat_df.iterrows():
            
            raw_text = stat_df_row["Text"]

            tokens = stat_df_row["Tokens"]
            article_state = stat_df_row["Article_State"]
            finished_annotators = stat_df_row["Finished_Annotators"]
            cur_spans = stat_df_row["Curation"]
            
            sent_tuples = tokenizer.span_tokenize(raw_text, realign_boundaries = True)
            
            i = 0

            for start, stop in sent_tuples:
                sentence = 'Sentence_{}'.format(str(i))
                
                row = pd.Series(index=self.df.columns, dtype = object)

                row["Sentence_Start"] = start
                row["Sentence_Stop"] = stop
                
                row["Sentence"] = stat_df_row.name +'_Sentence_{}'.format(str(i))
                row["Text"] = raw_text[start: stop]

                sent_tokens = [tok for tok in tokens if tok.start >= start and tok.stop <= stop] # do it in two lines since doing it in one line would affect all the previous tokens of the same article
                sent_tokens_copy= copy.deepcopy(sent_tokens)  
                row["Tokens"] = [add_sent(tok,sentence) for tok in sent_tokens_copy]


                row["Article_State"] = article_state
                row["Finished_Annotators"] = finished_annotators
                
                if type(cur_spans) == list:
                    cur_sent_spans = [spn for spn in cur_spans if spn.start >= start and spn.stop <= stop]
                    cur_sent_spans_copy = copy.deepcopy(cur_sent_spans)
                    row["Curation"]= [add_sent(spn,sentence) for spn in cur_sent_spans_copy]

                else:
                    row["Curation"] = ''
                
                for annotator in annotators:
                    
                    if annotator in finished_annotators and type(stat_df_row[annotator]) == list:
                        ann_sent_spans = [spn for spn in stat_df_row[annotator] if spn.start >= start and spn.stop <= stop]
                        ann_sent_spans_copy = copy.deepcopy(ann_sent_spans)
                        row[annotator] = [add_sent(spn,sentence) for spn in ann_sent_spans_copy]
                        
                    else:
                        row[annotator] = ''
                    
                self.df = self.df.append(row, ignore_index=True)
                i +=1
                global_index +=1



        

    


