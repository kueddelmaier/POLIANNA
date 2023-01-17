import requests as req
import pandas as pd
import os 

# Script to download EU laws based on CELEX numbers

# The CELEX numbers are stored in a file 'searches.csv', which for example results from 
# querying https://eur-lex.europa.eu/homepage.html for relevant policies.
# Loading the file:
exp_search = 'searches.csv'
df = pd.read_csv(exp_search,encoding= "ISO-8859-1")


# Function to take in a dataframe with CELEX numbers of EU laws and download the full text html file
# Input: Data frame with CELEX numbers of EU laws
# Output: Full text html files for each law, stored in the directory 'texts'

def get_html(df):
    os.mkdir('texts') 
    for i in range(len(df)):
        celex= df.loc[i,'CELEX number']
        url = 'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{}&from=EN'.format(celex)
        resp = req.get(url)
        filename = '{}.html'.format(celex)
        with open( 'texts/' + 'EU_' + filename, "w", encoding  = 'utf-8') as file:
            file.write(resp.text,)
            file.close()

get_html(df)