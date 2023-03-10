import streamlit as st
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import koreanize_matplotlib
import plotly.express as px

import koreanize_matplotlib
#%config InlineBackend.figure_format = 'retina'

from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank
from kiwipiepy import Kiwi

import requests
from PIL import Image
from bs4 import BeautifulSoup
import cv2

st.set_page_config(
    page_title="brand review analysis",
    page_icon="๐",
    layout="wide",
)

st.markdown("# ๐ ๋ธ๋๋๋ฅผ ์ ํํด์ฃผ์ธ์. ๐")

st.sidebar.markdown("# ๋ธ๋๋ ์ ํ โ")

# select brand
brand_list = ['๋ธ๋๋ ์ ํ', '๋ผํผ์ง์คํ ์ด', '๊ผผํ๋จ', '๋๋ก์ฐํ', '์ธ์ฌ์ผ๋ฐ์ค',
            '์ปค๋ฒ๋ซ', 'ํ๋ฅดํฐ๋ฉํ ', 'ํ๋ฃจ๋ฏธ๋ค์ดํธ', '์๋ฆฟ์ด์ฆ', '์์๋ ',
            '๋ด์๋์ง์ค๊ทธ๋ํฝ', '์์ผ', '๋์ฆ์ด์ค๋ค๋ฒ๋', '์์์คํ ๋ฉ', '๋ฆฌ',
            '์ด๋ฐ๋๋ ์ค']
choice = st.selectbox('๐๋ณด๊ณ ์ ํ๋ ๋ธ๋๋๋ฅผ ์ ํํด์ฃผ์ธ์.', brand_list)

select_brand = []
for num in range(len(brand_list)):
    if num == 0:
        pass
    elif choice == brand_list[num]:
        st.write(f'{brand_list[num]}๋ฅผ ์ ํํ์จ๊ตฐ์. โณ ์ ์๋ง ๊ธฐ๋ค๋ ค ์ฃผ์ธ์.')
        select_brand.append(f'{brand_list[num]}')
    

# data load
brand_link = {
    '๋ผํผ์ง์คํ ์ด' : '1fr6RZGM_vd5L0IDS0XIJCcScn_QtFKYz',
    '๋๋ก์ฐํ' : '1KoWlv8cQXI9kpmfVL1pSr6jLH1RvkXkt',
    '์ปค๋ฒ๋ซ' : '1we5q5975vDb2iNrmTPfDNsCNrQxrrWTQ',
    'ํ๋ฅดํฐ๋ฉํ ' : '1D1BhygdkEvZU4uQQ1Y450zpZVHbdDiwc',
    'ํ๋ฃจ๋ฏธ๋ค์ดํธ' : '190VQjL5F-8KPxQlYj3_8pY9Fb9JzmPIi',
    '๊ผผํ๋จ' : '1AjkzWni2Lp1V2vzhe3-gwzjDT1ASCsA2',
    '์ธ์ฌ์ผ๋ฐ์ค' : '12vrFmoKeJ_UHaXljvsO1l4rbvV1Tw4Dq' ,
    '์๋ฆฟ์ด์ฆ' : '1C_hPRBxb0sp6bJdVV4OEmNmMWFqYILQZ' ,
    '์์๋ ' : '1KZSMUTjqqGGMVOp5KiH7X_n23IcJL3wB',
    '๋ด์๋์ง์ค๊ทธ๋ํฝ' : '1gWhGfIluszy8oVO-RwZIKtxJ0RM8yMhn',
    '์์ผ' : '1eIoxQZrDLhsCWEl1-NJ9yE3Biem3VIcG',    
    '๋์ฆ์ด์ค๋ค๋ฒ๋' : '1ZgW4xBoXcNczxjMdw6YMKVEdjtgsUoyK',    
    '์์์คํ ๋ฉ' : '1KfcwqNRRbKLgCNvgMIgZiRLaJ8Si8BJU',
    '๋ฆฌ' : '1fqw2TiNEDxyrkaSDIlcxQ6UUyUD38kgW',
    '์ด๋ฐ๋๋ ์ค' : '1YmNK_XSR03fcKgnt6ZOmWkAusXteV8tT' 
}

def data_load(select_brand):
    data_link = 'https://drive.google.com/uc?id='+brand_link[select_brand]
    data = pd.read_csv(data_link) 
    return data

  
try : 
    data_load_state = st.spinner('Loading data...') 
    data = data_load(select_brand[0])
except KeyError as k:
    pass
except IndexError as i:
    pass

# labeling
def labeling(data):
    df = data[["๋ฆฌ๋ทฐ", "ํ์ "]]
    df = df.reset_index(drop=True)

    ์ญ์  = df[(df["ํ์ "]=="60%") | (df["ํ์ "]=="80%")].index
    df = df.drop(์ญ์ )

    df.loc[(df["ํ์ "] == "100%"), "label"] = 1
    df.loc[(df["ํ์ "] == "20%") | (df["ํ์ "] == "40%"), "label"] = 0

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# ๋ฌธ์ ์ ์ฒ๋ฆฌ
def preprocessing(text):
    text = re.sub('[^๊ฐ-ํฃใฑ-ใใ-ใฃa-zA]', " ", text)
    text = re.sub('[\s]+', " ", text)
    text = text.lower()
    return text

try :
    label_data = labeling(data=data)
    positive = label_data[(label_data["ํ์ "] == "100%")]
    negative = label_data[(label_data["ํ์ "] == "20%") | (label_data["ํ์ "] == "40%")]
    positive['๋ฆฌ๋ทฐ'] = positive['๋ฆฌ๋ทฐ'].map(preprocessing)
    negative['๋ฆฌ๋ทฐ'] = negative['๋ฆฌ๋ทฐ'].map(preprocessing)
    data_load_state.spinner(f'{select_brand[0]} ๋ฐ์ดํฐ ๋ก๋ success โผ')
except KeyError as k:
    pass
except NameError as n:
    pass

# Kiwi ์ ์ฉ
def kiwi(sentence):
    results = []
    result = Kiwi().analyze(sentence)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
            results.append(token)
    return results

try :
    pos_noun_list = []
    neg_noun_list = []
    for pos in positive['๋ฆฌ๋ทฐ'].tolist():
        pos_nouns = kiwi(pos)
        pos_text = ' '.join(pos_nouns)
        pos_noun_list.append(pos_text)

    for neg in negative['๋ฆฌ๋ทฐ'].tolist():
        neg_nouns = kiwi(neg)
        neg_text = ' '.join(neg_nouns)
        neg_noun_list.append(neg_text)

except KeyError as k:
    pass
except NameError as n:
    pass

# wordrank ์ ์ฉ top10 ํค์๋ ๋ฝ๊ธฐ
def word_rank(corpus):
    beta = 0.90    # PageRank์ decaying factor beta
    max_iter = 5
    top_keywords = []
    fnames = [corpus]
    top_10=[]
    
    for fname in fnames:
        texts = fname
        wordrank_extractor = KRWordRank(min_count=5, max_length=10, verbose=False)
        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
        top_keywords.append(sorted(keywords.items(),key=lambda x:x[1],reverse=True)[:100])
        
    for i in range(10):
        if i<10:
            top_10.append(top_keywords[0][i][0])
            i += 1
    return top_10

# img data load
img_brand_link = {
    '๋ผํผ์ง์คํ ์ด_img' : '1QfwKjzDAfoowpe4LiH9yxhChdZo3iizh',
    '๋๋ก์ฐํ_img' : '1-caqKnBlM4Q26tec_4aWFm9017F9-_E4',
    '์ปค๋ฒ๋ซ_img' : '1f50QDJI6K7KeZ7WGSANV1sH6aiXa2c5T',
    'ํ๋ฅดํฐ๋ฉํ _img' : '1Nt0LAAWlvVTh60Y9Zvbb3jmw0Jdl-URg',
    'ํ๋ฃจ๋ฏธ๋ค์ดํธ_img' : '1CtYGt4E5hzqp-tvwix5WhANpvzds8TQK',
    '๊ผผํ๋จ_img' : '1-CJcWAp3WxKymk7PUxj9NYgM-3zfjM83',
    '์ธ์ฌ์ผ๋ฐ์ค_img' : '1COUpes3WPXeGn6mdYrtzG1D9dMJ43SF7' ,
    '์๋ฆฟ์ด์ฆ_img' : '1nxOa5_69KQrmbMduDhQtAPFgiclIFXO_' ,
    '์์๋ _img' : '1G5QtrNYtKNhFgRVx0Fj2-b0F626o7ccX',
    '๋ด์๋์ง์ค๊ทธ๋ํฝ_img' : '1Wm2ox9koFbYXkMtvLApCfQjDnPfSRsGF',
    '์์ผ_img' : '1MxqLCSCptl5O5shldxK513mh4G7z3j3V',    
    '๋์ฆ์ด์ค๋ค๋ฒ๋_img' : '17yuM4U3W3aKKMCQphvBePiHC5mVugVkf',    
    '์์์คํ ๋ฉ_img' : '17v-GwoTF0mgOkRta3hAhytWLPOh2YUpk',
    '๋ฆฌ_img' : '1us6tb40vHoz4hrNGfoD9n2BL0fciY97n',
    '์ด๋ฐ๋๋ ์ค_img' : '1LDAyqzM-TZZCLVrZJK08WLJe2IF6Jtxt' 
}

def img_data_load(select_brand):
    img_data_link = 'https://drive.google.com/uc?id='+img_brand_link[select_brand]
    img_data = pd.read_csv(img_data_link) 
    return img_data    


def keyword_review(link_csv, data, keywords):
    img_csv = pd.read_csv('https://drive.google.com/uc?id='+brand_link[select_brand])

    for keyword_count in range(len(keywords)):
        if st.button(keywords[keyword_count]):
            keyword_review_data = data[data['๋ฆฌ๋ทฐ'].str.contains(f'{keywords[keyword_count]}')]

            product_num = keyword_review['์ํ_num']
            top3_cumc_product = product_num.value_counts().sort_values(ascending=False)[:3].index

            review1 = keyword_review_data[keyword_review_data['์ํ_num']==f'{top3_cumc_product[0]}'].sample(3)
            review2 = keyword_review_data[keyword_review_data['์ํ_num']==f'{top3_cumc_product[1]}'].sample(3)
            review3 = keyword_review_data[keyword_review_data['์ํ_num']==f'{top3_cumc_product[2]}'].sample(3)

            img_link = []
            for num in product_num:
                link = link_csv[link_csv['์ํ']==num]
                img_link.append(link['์ฌ์ง'])

            # Load the image from the URL
            for i in range(len(img_link)):
                URL = f'https:{img_link}'
                response = requests.get(URL)
                image = Image.open(BytesIO(response.content))

                st.image(image, caption='Image from URL')
                st.text(f'review{i}')
        

try :
    pos_keyword = word_rank(pos_noun_list)
    neg_keyword = word_rank(neg_noun_list)

    img_link = img_data_load(select_brand)
    keyword_review(img_link, positive, pos_keyword)
    keyword_review(img_link, positive, neg_keyword)
except KeyError as k:
    pass
except NameError as n:
    pass

