
import json
import zipfile
import os
import sys
import io
from ast import literal_eval
import pandas as pd
from itertools import chain
from itertools import groupby
import unicodedata
import numpy as np
import collections
#sys.path.append('/home/kueddelmaier/eth/work/NLP_spark/src')
sys.path.append('/Users/l.kaack/Documents/Policy_coding/NLP_Spark/')
from src.experiment_utils.helper_classes import token, span, repository
from definitions import ROOT_DIR, RAW_DATA_PATH


annotator_path = os.path.join(RAW_DATA_PATH , 'annotation')  
curation_path = os.path.join(RAW_DATA_PATH , 'curation') 



stat_df = pd.DataFrame(columns = ['Policy', 'Text', 'Tokens', 'Article_State', 'Finished_Annotators', 'Curation'], dtype = object) #create the initial dataframe
tagsets = ['Policydesigncharacteristics','Technologyandapplicationspecificity','Instrumenttypes']

error_span_list = []

annotator_subdirs = [o for o in os.listdir(annotator_path) if os.path.isdir(annotator_path)]
curation_subdirs = [o for o in os.listdir(curation_path) if os.path.isdir(curation_path)]

json_files_dir = [file for file in os.listdir(RAW_DATA_PATH) if file.endswith('.json')]

if len(json_files_dir) != 1:
    for file in json_files_dir:
        print(file)

    raise ValueError('too many json project logs')

project_log = json_files_dir[0]

# save state and finsihed coders in a dict

article_state = {}
with open(os.path.join(RAW_DATA_PATH, project_log)) as json_file:
    article_state_data = json.load(json_file)
    for article in article_state_data['source_documents']:

        policy_name = article['name']
        policy_state = article['state']
        finished_coders = [doc_article['user'] for doc_article in article_state_data['annotation_documents'] if doc_article['name'] == policy_name and doc_article['state'] == 'FINISHED']

        this_state = {'state': article['state'],
                       'finished_coders': finished_coders}
        article_state[policy_name] = this_state


#curation subdirs should be a subset of annotator_subdirs
if set(curation_subdirs).issubset(set(annotator_subdirs)) == False:
    raise ValueError('Curated directories should be a subset of annotator data!')
 #identify all the subdirs which correspond to different articlesf
os.chdir(annotator_path)

#%%

for subdir in annotator_subdirs:

    annotator_count = 1
    subdir_index = subdir[0:-4]
    stat_df = stat_df.append(pd.Series(name=subdir_index, dtype = object))

    rep = repository.from_repository_name(subdir_index)  #directory name in string format
    


    for annonator in os.listdir(os.path.join(annotator_path,subdir)):
        
       
        try:
            archive = zipfile.ZipFile(os.path.join(annotator_path,subdir, annonator), 'r') #decode compressed json in zip file
            files = archive.namelist()

            json_files = [file for file in files if file.endswith('json')]

            if len(json_files) != 1:
                break

            annotator = os.path.splitext(json_files[0])[0]
            json_file_byte = archive.read(json_files[0])      #this is a binary
            
            json_file_byte_decode = json_file_byte.decode('utf8')    #decode to json
        
            data = json.loads(json_file_byte_decode)
        
        except:
            print('hey')
             #stat_df = stat_df.append(pd.Series([subdir, 'error', 'error', 'error'], index=stat_df.columns ), ignore_index=True) #append error column if cannot read data

        

        else:
            if annotator not in stat_df.columns: #check if annotator already in dataframe, if not append empry column
                stat_df[annotator] = ''
                 

            spanlist = []      #create epty list holding all the spans of the paragraph (= article)
            
            
            if annotator_count == 1:
                sentence = data['_referenced_fss']['1']['sofaString'].lower()
                stat_df['Text'].loc[subdir_index] = sentence  #raw text of the paragraph (all in lower case)
                
                all_tokens_json = data['_views']['_InitialView']['Token']
                all_tokens_json[0]['begin'] = 0                            #the first token is missing the beginning
                token_list = [token(x['begin'], x['end'], sentence[x['begin']:x['end']], rep) for x in all_tokens_json]
                stat_df['Tokens'].loc[subdir_index] = token_list
                stat_df['Article_State'].loc[subdir_index] = article_state[subdir]['state']
                stat_df['Finished_Annotators'].loc[subdir_index] = article_state[subdir]['finished_coders']
                #sentence_normalized = normalize_and_replace_text(sentence)
            else:
                token_list = stat_df['Tokens'].loc[subdir_index]
            
            for category in data['_views']['_InitialView']:  #loop trough the custom layers
    
                if category in tagsets:
                        
                    for annotation in data['_views']['_InitialView'][category]: #loop trough all the spans
                        
                        type_ = list(annotation.keys())[-1]     #this part handles empty annotations. The last entry of the dict usually contains the type and the span.
                                                                #for empty spans, the last entry is a integer
                        if type(type_) != str:                  
                            type_ = ''
                        
                        _tag_ = list(annotation.values())[-1]
                        if type(_tag_) != str:
                            _tag_ = ''
                        try:    
                            start = annotation['begin']
                            stop = annotation['end']
                        except:
                            error_span_list.append(span(category,type_, _tag_ ,0,0,'error', 'error', rep, annotator ))
                            pass
                        
                        span_tokens = [x for x in token_list if  x.start >= start and x.stop <= stop]
                        len_doubles = len([dspan for dspan in spanlist if dspan.tag_ == _tag_ and dspan.start == start and dspan.stop == stop])
                        if len_doubles != 0:
                            continue

                        for span_token in span_tokens:
                            span_token.tag_count +=1
                        
                        spanlist.append(span(category, type_, _tag_ , start ,stop , sentence[start:stop], span_tokens, rep, annotator))
            stat_df[annotator].loc[subdir_index] = spanlist
            annotator_count +=1   
            #stat_df = stat_df.append(pd.Series([subdir[0:-4], sentence ,spanlist, token_list], index=stat_df.columns ), ignore_index=True)     
    #stat_df['tokens cleaned'] = stat_df['Text'].apply(clean_text)  

#%%


#curation
os.chdir(curation_path)

for subdir in curation_subdirs:

    subdir_index = subdir[0:-4]
    rep = repository.from_repository_name(subdir_index)
       
    try:
        archive = zipfile.ZipFile(os.path.join(subdir, str(os.listdir(subdir)[0])), 'r') #decode compressed json in zip file
        json_file_byte = archive.read('CURATION_USER.json')      #this is a binary
        
        json_file_byte_decode = json_file_byte.decode('utf8')    #decode to json
    
        data = json.loads(json_file_byte_decode)
    
    except:
        print('hey')
        #stat_df = stat_df.append(pd.Series([subdir, 'error', 'error', 'error'], index=stat_df.columns ), ignore_index=True) #append error column if cannot read data

        
    else:
        spanlist = []      #create epty list holding all the spans of the paragraph (= article)
        subdir_index = subdir[0:-4]  #directory name in string format
        sentence = stat_df['Text'].loc[subdir_index]
        token_list = stat_df['Tokens'].loc[subdir_index]
        #sentence_normalized = normalize_and_replace_text(sentence) 
        
        for category in data['_views']['_InitialView']:  #loop trough the custom layers
  
            if category in tagsets:
                    
                for annotation in data['_views']['_InitialView'][category]: #loop trough all the spans
                    
                    type_ = list(annotation.keys())[-1]     #this part handles empty annotations. The last entry of the dict usually contains the type and the span.
                                                            #for empty spans, the last entry is a integer
                    if type(type_) != str:                  
                        type_ = ''
                    
                    _tag_ = list(annotation.values())[-1]
                    if type(_tag_) != str:
                        _tag_ = ''
                        
                    start = annotation['begin']
                    stop = annotation['end']
                    span_tokens = [x for x in token_list if  x.start >= start and x.stop <= stop]
                    
                    for span_token in span_tokens:
                        span_token.tag_count +=1
                    
                    spanlist.append(span(category, type_, _tag_ , start ,stop , sentence[start:stop], span_tokens, rep, 'Curation'))
                    
        stat_df['Curation'].loc[subdir_index] = spanlist     
#stat_df['tokens cleaned'] = stat_df['Text'].apply(clean_text)  



#%%
stat_df = stat_df.replace(np.nan, '')

#create data_dir

data_dir = os.path.join(ROOT_DIR, 'data')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


processed_data_dir = os.path.join(data_dir,'02_processed_to_dataframe')

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

out_dir_pkl = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe.pkl')
stat_df.to_pickle(out_dir_pkl)
out_dir_csv = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe.csv')
stat_df.to_csv(out_dir_csv)