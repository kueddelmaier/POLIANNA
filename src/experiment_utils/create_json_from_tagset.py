import json
import os

path = '/home/kueddelmaier/Downloads'
ls = ['policy_design_characteristics', 'technology_and_application_specificity', 'instrument_types']

dict_ = {'layers': []}

for layer in ls:
    layer_dict = {'layer':layer,
                'layer_description':'',
                'tagsets':[]
                 }
    layer_description = ''
    layer_path = os.path.join(path, layer)
    
    for file in os.listdir(layer_path):
       
        with open(os.path.join(layer_path, file)) as json_file:

            data = json.load(json_file)
            tag_set = data['name']
            tagset_description = ''
            tagset_dict = {'tagset':tag_set,
                'tagset_description':'',
                'tags':[]
                 }
            for tag in data['tags']:
                tag_name = tag['tag_name']
                tag_description = tag['tag_description']
                tag_dict = {'tag_name': tag_name,
                           'tag_description': tag_description }
                tagset_dict['tags'].append(tag_dict)
        layer_dict['tagsets'].append(tagset_dict)
    dict_['layers'].append(layer_dict)
            

with open('personal.json', 'w') as json_save_file:
    json.dump(dict_, json_save_file)