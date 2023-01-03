import json
import zipfile
import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import data_utils as utils

#just to test, delete at the and and do with python path

sys.path.insert(0,'../..')

from src.experiment_utils.helper_classes import token, span, repository
from definitions import ROOT_DIR, RAW_DATA_PATH


""" Global variables and objects """

stat_df = pd.DataFrame(columns = ['Policy', 'Text', 'Tokens', 'Article_State', 'Finished_Annotators', 'Curation'], dtype = object) #create the initial dataframe
tagsets = ['Policydesigncharacteristics','Technologyandapplicationspecificity','Instrumenttypes']

annotator_look_up = {}
span_counter = {}
error_span_list = []


def return_annotator_letter(annotator):
    
    if annotator in annotator_look_up:
        return annotator_look_up[annotator]
    else:
        char = chr(ord('@')+len(annotator_look_up)+1)
        annotator_look_up[annotator] = char
        span_counter[annotator] = 1
        return char
    

if __name__ == "__main__":

    """ Process command line arguments """

    parser = argparse.ArgumentParser(description='Load annotation data')

    parser.add_argument('--out_dir', type = str,
        help='Where to store the processed data', default= os.path.join(ROOT_DIR, 'data'))

    parser.add_argument('--anonymous_annotators', type = bool,
        help='If annotators are replaced by alphabetic letters', default = False)

    parser.add_argument('--remove_doublicates', type = bool,
        help='If doublicate spans should be removed', default = True)  


    args = parser.parse_args()

    annotator_path = os.path.join(RAW_DATA_PATH , 'annotation')  
    curation_path = os.path.join(RAW_DATA_PATH , 'curation') 

    annotator_subdirs = [o for o in os.listdir(annotator_path) if os.path.isdir(annotator_path)]
    curation_subdirs = [o for o in os.listdir(curation_path) if os.path.isdir(curation_path)]

    json_files_dir = [file_ for file_ in os.listdir(RAW_DATA_PATH) if file_.endswith('.json')]

    if len(json_files_dir) != 1:
        for file_ in json_files_dir:
            print(file_)

        raise ValueError('too many json project logs')

    """ Get Project Logs """

    project_log = json_files_dir[0]

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
    #identify all the subdirs which correspond to different 

    """ Process Annotations of Annotators """
    os.chdir(annotator_path)

    for subdir in annotator_subdirs:

        annotator_count = 1
        subdir_index = subdir[0:-4]
        stat_df = stat_df.append(pd.Series(name=subdir_index, dtype = object))

        rep = repository.from_repository_name(subdir_index)  #directory name in string format

        for ann_folder in os.listdir(os.path.join(annotator_path,subdir)):

            try:
                archive = zipfile.ZipFile(os.path.join(annotator_path,subdir, ann_folder), 'r') #decode compressed json in zip file
                files = archive.namelist()
                json_files = [file for file in files if file.endswith('json')]

                if len(json_files) != 1:
                    break

                annotator = os.path.splitext(json_files[0])[0]

                json_file_byte = archive.read(json_files[0])      #this is a binary
                json_file_byte_decode = json_file_byte.decode('utf8')    #decode to json
                data = json.loads(json_file_byte_decode)

                ann_letter = return_annotator_letter(annotator)
            
            except:
                logging.error(f"Could not process {ann_folder}")

            else:
                if args.anonymous_annotators:
                    if return_annotator_letter(annotator) not in stat_df.columns: #check if annotator already in dataframe, if not append empry column
                        stat_df[return_annotator_letter(annotator)] = ''
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

                    if args.anonymous_annotators:
                        stat_df['Finished_Annotators'].loc[subdir_index] = [return_annotator_letter(ann) for ann in article_state[subdir]['finished_coders']]
                    else:
                        stat_df['Finished_Annotators'].loc[subdir_index] = article_state[subdir]['finished_coders']

                    #sentence_normalized = normalize_and_replace_text(sentence)
                else:
                    token_list = stat_df['Tokens'].loc[subdir_index]
                
                span_count = 0
                for category in data['_views']['_InitialView']:  #loop trough the custom layers
        
                    if category in tagsets:
                            
                        for annotation in data['_views']['_InitialView'][category]: #loop trough all the spans
                            
                            feature = list(annotation.keys())[-1]     #this part handles empty annotations. The last entry of the dict usually contains the feature and the span.
                                                                    #for empty spans, the last entry is a integer
                            if type(feature) != str:                  
                                feature = ''
                            
                            tag = list(annotation.values())[-1]
                            if type(tag) != str:
                                tag = ''
                            try:    
                                start = annotation['begin']
                                stop = annotation['end']
                            except:
                                logging.info(f"Span of folder {subdir_index}, annotator {annotator}, catergory {category}, feature {feature} and tag {tag} could not be processed")
                                pass
                            
                            span_tokens = [x for x in token_list if  x.start >= start and x.stop <= stop]
                            len_doubles = len([dspan for dspan in spanlist if dspan.tag == tag and dspan.start == start and dspan.stop == stop])
                            if len_doubles != 0:
                                continue
                            span_id =  ann_letter + str(span_counter[annotator])
                            
                            if args.anonymous_annotators:
                                span_ = span(span_id, category, feature, tag , start ,stop , sentence[start:stop], span_tokens, rep, return_annotator_letter(annotator))
                            
                            else:
                                span_ = span(span_id, category, feature, tag , start ,stop , sentence[start:stop], span_tokens, rep, annotator)

                            spanlist.append(span_)
                            [tok.add_span(span_) for tok in span_tokens]

                            span_counter[annotator] +=1
    
                if args.remove_doublicates== True:
                    spanlist_clean = utils.remove_span_doublicates(spanlist)

                if args.anonymous_annotators: 
                    stat_df[return_annotator_letter(annotator)].loc[subdir_index] = spanlist_clean
                else:
                    stat_df[annotator].loc[subdir_index] = spanlist_clean

                annotator_count +=1   



    """ Do the same for the Curation """

    os.chdir(curation_path)
    ann_letter = 'CUR'
    cur_count = 0
    for subdir in curation_subdirs:

        subdir_index = subdir[0:-4]
        rep = repository.from_repository_name(subdir_index)
        
        try:
            archive = zipfile.ZipFile(os.path.join(subdir, str(os.listdir(subdir)[0])), 'r') #decode compressed json in zip file
            json_file_byte = archive.read('CURATION_USER.json')      #this is a binary
            
            json_file_byte_decode = json_file_byte.decode('utf8')    #decode to json
        
            data = json.loads(json_file_byte_decode)
        
        except:
            logging.error(f"Could not process {ann_folder}")


            
        else:
            spanlist = []      #create epty list holding all the spans of the paragraph (= article)
            subdir_index = subdir[0:-4]  #directory name in string format
            sentence = stat_df['Text'].loc[subdir_index]
            token_list = stat_df['Tokens'].loc[subdir_index]
            #sentence_normalized = normas
            for category in data['_views']['_InitialView']:  #loop trough the custom layers
    
                if category in tagsets:
                        
                    for annotation in data['_views']['_InitialView'][category]: #loop trough all the spans
                        
                        feature = list(annotation.keys())[-1]     #this part handles empty annotations. The last entry of the dict usually contains the feature and the span.
                                                                #for empty spans, the last entry is a integer
                        if type(feature) != str:                  
                            feature = ''
                        
                        tag = list(annotation.values())[-1]
                        if type(tag ) != str:
                            tag  = ''
                        
                        try:
                            
                            start = annotation['begin']
                            stop = annotation['end']
                        except:
                            logging.info(f"Span of folder {subdir_index}, annotator Curation , catergory {category}, feature {feature} and tag {tag} could not be processed")

                        span_tokens = [x for x in token_list if  x.start >= start and x.stop <= stop]
                        

                        span_id =  ann_letter + str(cur_count)
                        span_ =  span(span_id, category, feature, tag  , start ,stop , sentence[start:stop], span_tokens, rep, 'Curation') #create span
                        spanlist.append(span_) #append span to the spanlist of this article
                        [tok.add_span(span_) for tok in span_tokens ] #assign this span to all the tokens that belong to this span
                        cur_count += 1

            if args.remove_doublicates== True:
                spanlist_clean = utils.remove_span_doublicates(spanlist)

            stat_df['Curation'].loc[subdir_index] = spanlist_clean     
    #stat_df['tokens cleaned'] = stat_df['Text'].apply(clean_text)  


    stat_df = stat_df.replace(np.nan, '')

    #create data_dir

    """ Save Data """
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    processed_out_dir = os.path.join(args.out_dir,'02_processed_to_dataframe')

    if not os.path.exists(processed_out_dir):
        os.makedirs(processed_out_dir)

    #out_dir_pkl = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe_anonymous.pkl')
    out_dir_pkl = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe_test.pkl')
    stat_df.to_pickle(out_dir_pkl)
    out_dir_csv = os.path.join(ROOT_DIR,'data/02_processed_to_dataframe', 'preprocessed_dataframe_test.csv')
    stat_df.to_csv(out_dir_csv)

    print('out_dir: ', out_dir_pkl)

