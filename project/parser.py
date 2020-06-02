import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
import codecs
import pickle
from bs4 import BeautifulSoup as BS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from boilerpipe.extract import Extractor
from sys import argv

if(len(argv) !=2):
    print( 'Enter parser type')
    exit(1)
_ , parser_type = argv
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = 'content/'
texts = {}
print('Start: ' + parser_type)
train_data = pd.read_csv('train_groups.csv',dtype=np.int16)
for filename in tqdm(listdir(path)):
    doc_id = int(filename.strip('.dat'))
    if doc_id not in train_data.doc_id.values:
        continue
    with codecs.open(path + filename, 'r', 'utf-8') as f:
        url = f.readline().strip()
        html = f.read()
        extractor = Extractor(extractor=parser_type, html=html)
        s = extractor.getText()
        s=s.replace('\n'," ")
        s=s.replace('\t'," ")
        s=s.replace('\r'," ")
        texts[doc_id] = s

train_data['text'] = train_data.apply(lambda row: texts[row.doc_id], axis = 1)

save_obj(train_data,'train_data' + parser_type)