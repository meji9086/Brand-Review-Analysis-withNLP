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


st.set_page_config(
    page_title="brand review analysis",
    page_icon="π",
    layout="wide",
)

st.markdown("# π λΈλλλ₯Ό μ νν΄μ£ΌμΈμ. π")

st.sidebar.markdown("# π μ§μνλ λΈλλ μ’λ₯")
st.sidebar.markdown("""
λΌνΌμ§μ€ν μ΄ lafudgestore            
κΌΌνλ¨ compagno         
λλ‘μ°ν Draw fit         
μΈμ¬μΌλ°μ€ insilence       
μ»€λ²λ« covernat         
νλ₯΄ν°λ©ν  partimento         
νλ£¨λ―Έλ€μ΄νΈ filluminate       
μλ¦Ώμ΄μ¦ whatitisnt       
μμλ  suare        
λ΄μλμ§μ€κ·Έλν½ nationalgeographic           
μμΌ yale       
λμ¦μ΄μ€λ€λ²λ thisisneverthat         
μμμ€ν λ© outstanding        
λ¦¬ lee      
μ΄λ°λλ μ€ avan         
""")

# select brand
brand_list = ['λΈλλ μ ν', 'λΌνΌμ§μ€ν μ΄', 'κΌΌνλ¨', 'λλ‘μ°ν', 'μΈμ¬μΌλ°μ€',
            'μ»€λ²λ«', 'νλ₯΄ν°λ©ν ', 'νλ£¨λ―Έλ€μ΄νΈ', 'μλ¦Ώμ΄μ¦', 'μμλ ',
            'λ΄μλμ§μ€κ·Έλν½', 'μμΌ', 'λμ¦μ΄μ€λ€λ²λ', 'μμμ€ν λ©', 'λ¦¬',
            'μ΄λ°λλ μ€']
choice = st.selectbox('πλ³΄κ³ μ νλ λΈλλλ₯Ό μ νν΄μ£ΌμΈμ.', brand_list)

select_brand = []
for num in range(len(brand_list)):
    if num == 0:
        pass
    elif choice == brand_list[num]:
        st.write(f'{brand_list[num]}λ₯Ό μ ννμ¨κ΅°μ. β³ μ μλ§ κΈ°λ€λ € μ£ΌμΈμ.')
        select_brand.append(f'{brand_list[num]}')
    

# data load
brand_link = {
    'λΌνΌμ§μ€ν μ΄' : '1fr6RZGM_vd5L0IDS0XIJCcScn_QtFKYz',
    'λλ‘μ°ν' : '1KoWlv8cQXI9kpmfVL1pSr6jLH1RvkXkt',
    'μ»€λ²λ«' : '1we5q5975vDb2iNrmTPfDNsCNrQxrrWTQ',
    'νλ₯΄ν°λ©ν ' : '1D1BhygdkEvZU4uQQ1Y450zpZVHbdDiwc',
    'νλ£¨λ―Έλ€μ΄νΈ' : '190VQjL5F-8KPxQlYj3_8pY9Fb9JzmPIi',
    'κΌΌνλ¨' : '1AjkzWni2Lp1V2vzhe3-gwzjDT1ASCsA2',
    'μΈμ¬μΌλ°μ€' : '12vrFmoKeJ_UHaXljvsO1l4rbvV1Tw4Dq' ,
    'μλ¦Ώμ΄μ¦' : '1C_hPRBxb0sp6bJdVV4OEmNmMWFqYILQZ' ,
    'μμλ ' : '1KZSMUTjqqGGMVOp5KiH7X_n23IcJL3wB',
    'λ΄μλμ§μ€κ·Έλν½' : '1gWhGfIluszy8oVO-RwZIKtxJ0RM8yMhn',
    'μμΌ' : '1eIoxQZrDLhsCWEl1-NJ9yE3Biem3VIcG',    
    'λμ¦μ΄μ€λ€λ²λ' : '1ZgW4xBoXcNczxjMdw6YMKVEdjtgsUoyK',    
    'μμμ€ν λ©' : '1KfcwqNRRbKLgCNvgMIgZiRLaJ8Si8BJU',
    'λ¦¬' : '1fqw2TiNEDxyrkaSDIlcxQ6UUyUD38kgW',
    'μ΄λ°λλ μ€' : '1YmNK_XSR03fcKgnt6ZOmWkAusXteV8tT' 
}

@st.cache
def data_load(select_brand):
    data_link = 'https://drive.google.com/uc?id='+brand_link[select_brand]
    data = pd.read_csv(data_link) 
    return data

  
try : 
    data_load_state = st.text('Loading data...') 
    data = data_load(select_brand[0])
    pos_data = data[data["νμ "] == "100%"]
    neg_data = data[(data["νμ "] == "20%") | (data["νμ "] == "40%")]
except KeyError as k:
    pass
except IndexError as i:
    pass

# labeling
def labeling(data):
    df = data[["λ¦¬λ·°", "νμ "]]
    df = df.reset_index(drop=True)

    μ­μ  = df[(df["νμ "]=="60%") | (df["νμ "]=="80%")].index
    df = df.drop(μ­μ )

    df.loc[(df["νμ "] == "100%"), "label"] = 1
    df.loc[(df["νμ "] == "20%") | (df["νμ "] == "40%"), "label"] = 0

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# λ¬Έμ μ μ²λ¦¬
def preprocessing(text):
    text = re.sub('[^κ°-ν£γ±-γγ-γ£a-zA]', " ", text)
    text = re.sub('[\s]+', " ", text)
    text = text.lower()
    return text

try :
    label_data = labeling(data=data)
    positive = label_data[(label_data["νμ "] == "100%")].sample(10)
    negative = label_data[(label_data["νμ "] == "20%") | (label_data["νμ "] == "40%")].sample(10)
    positive['λ¦¬λ·°'] = positive['λ¦¬λ·°'].map(preprocessing)
    negative['λ¦¬λ·°'] = negative['λ¦¬λ·°'].map(preprocessing)
except KeyError as k:
    pass
except NameError as n:
    pass

# Kiwi μ μ©
def noun_extractor(sentence):
    results = []
    result = Kiwi().analyze(sentence)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
            results.append(token)
    return results

try :
    pos_noun_list = []
    neg_noun_list = []
    print('κΈμ  λ¦¬λ·° kiwi')
    for pos in positive['λ¦¬λ·°'].tolist():
        pos_nouns = noun_extractor(pos)
        pos_text = ' '.join(pos_nouns)
        pos_noun_list.append(pos_text)

    print('λΆμ  λ¦¬λ·° kiwi')
    for neg in negative['λ¦¬λ·°'].tolist():
        neg_nouns = noun_extractor(neg)
        neg_text = ' '.join(neg_nouns)
        neg_noun_list.append(neg_text)

except KeyError as k:
    pass
except NameError as n:
    pass

# wordrank μ μ© top10 ν€μλ λ½κΈ°
def word_rank(corpus):
    beta = 0.90    # PageRankμ decaying factor beta
    max_iter = 5
    top_keywords = []
    fnames = [corpus]
    top_10=[]
    
    for fname in fnames:
        texts = fname
        wordrank_extractor = KRWordRank(min_count=1, max_length=10, verbose=False)
        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
        top_keywords.append(sorted(keywords.items(),key=lambda x:x[1],reverse=True)[:100])
        
    for i in range(len(top_keywords)):
        if i<10:
            top_10.append(top_keywords[0][i][0])
            i += 1
    return top_10

try :
    pos_keyword = word_rank(pos_noun_list)
    neg_keyword = word_rank(neg_noun_list)
except KeyError as k:
    pass
except NameError as n:
    pass
except ValueError as v:
    pass

# img data load
img_brand_link = {
    'λΌνΌμ§μ€ν μ΄' : '1QfwKjzDAfoowpe4LiH9yxhChdZo3iizh',
    'λλ‘μ°ν' : '1-caqKnBlM4Q26tec_4aWFm9017F9-_E4',
    'μ»€λ²λ«' : '1f50QDJI6K7KeZ7WGSANV1sH6aiXa2c5T',
    'νλ₯΄ν°λ©ν ' : '1Nt0LAAWlvVTh60Y9Zvbb3jmw0Jdl-URg',
    'νλ£¨λ―Έλ€μ΄νΈ' : '1CtYGt4E5hzqp-tvwix5WhANpvzds8TQK',
    'κΌΌνλ¨' : '1-CJcWAp3WxKymk7PUxj9NYgM-3zfjM83',
    'μΈμ¬μΌλ°μ€' : '1COUpes3WPXeGn6mdYrtzG1D9dMJ43SF7' ,
    'μλ¦Ώμ΄μ¦' : '1nxOa5_69KQrmbMduDhQtAPFgiclIFXO_' ,
    'μμλ ' : '1G5QtrNYtKNhFgRVx0Fj2-b0F626o7ccX',
    'λ΄μλμ§μ€κ·Έλν½' : '1Wm2ox9koFbYXkMtvLApCfQjDnPfSRsGF',
    'μμΌ' : '1MxqLCSCptl5O5shldxK513mh4G7z3j3V',    
    'λμ¦μ΄μ€λ€λ²λ' : '17yuM4U3W3aKKMCQphvBePiHC5mVugVkf',    
    'μμμ€ν λ©' : '17v-GwoTF0mgOkRta3hAhytWLPOh2YUpk',
    'λ¦¬' : '1us6tb40vHoz4hrNGfoD9n2BL0fciY97n',
    'μ΄λ°λλ μ€' : '1LDAyqzM-TZZCLVrZJK08WLJe2IF6Jtxt' 
}

def img_data_load(select_brand):
    for brand in select_brand:
        img_data_link = f'https://drive.google.com/uc?id='+img_brand_link[brand]
        img_csv = pd.read_csv(img_data_link) 
    return img_csv  


def keyword_review(img_link, df, keywords):
    # κ° ν€μλ forλ¬ΈμΌλ‘ λλ¦¬κΈ°
    for key_count in range(len(keywords)):
        # ν€μλμ λ¨μ΄λ₯Ό ν¬ν¨νλ λ¦¬λ·°λ₯Ό keyword_review_dataλ‘ ν λΉ
        keyword_review_data = df[df['λ¦¬λ·°'].str.contains(keywords[key_count])]

        # κ·Έ ν λΉν λ³μμμ μν λ²νΈλ₯Ό κ°μ Έμ€κ³ , κ°μ₯ λ§μ΄ μ°¨μ§νλ top3μ μν λ²νΈλ₯Ό κ°μ Έμ€κΈ°
        product_num = keyword_review_data['μν_num']
        top3_cumc_product = product_num.value_counts().sort_values(ascending=False)[:3].index

        # top3μ μν λ²νΈμ ν΄λΉνλ λ¦¬λ·° κ°κ° 3κ°μ λ¦¬λ·° κ°μ Έμ€κΈ°
        review1 = keyword_review_data[keyword_review_data['μν_num']==top3_cumc_product[0]]['λ¦¬λ·°'].sample(3)
        review2 = keyword_review_data[keyword_review_data['μν_num']==top3_cumc_product[1]]['λ¦¬λ·°'].sample(3)
        review3 = keyword_review_data[keyword_review_data['μν_num']==top3_cumc_product[2]]['λ¦¬λ·°'].sample(3)

        st.button(keywords[key_count])
        
        for i, number in enumerate(top3_cumc_product):
            # μνλͺ
            product_name_list = keyword_review_data[keyword_review_data['μν_num'] == top3_cumc_product[i]]['μν']
            product_name = list(set(product_name_list))
            st.text(f'μν μ΅μ : {product_name}')

            # μ΄λ―Έμ§ λ§ν¬
            #imgs_link = img_link[img_link['μν_num'] == number]['μ¬μ§'].values
            #join_link = ''.join(imgs_link)
            #link = f'https:{join_link}'
            #st.text(f'μ΄λ―Έμ§ λ§ν¬ : {link}')
            
            #image = Image.open(f'{img_link}')
            #st.image(image)

            if i==0:
                st.text(review1.values)
            if i==1:
                st.text(review2.values)
            if i==2:
                st.text(review3.values)

#            link = img_link[img_link['μν_num'] == num]['μ¬μ§'].values
#            join_link = ''.join(link)
#            URL = f'https:{join_link}'
#            response = requests.get(URL)
#            image = Image.open(BytesIO(response.content))
#            st.image(image, caption='Image from URL')
        

try :
    link_csv = img_data_load(select_brand)
    st.markdown("""## π κΈμ  λ¦¬λ·° ν΅μ¬ ν€μλ""")
    pos = keyword_review(link_csv, pos_data, pos_keyword)
    st.text(pos)
    st.markdown("""## π λΆμ  λ¦¬λ·° ν΅μ¬ ν€μλ""")
    neg = keyword_review(link_csv, neg_data, neg_keyword)
    st.text(neg)
except KeyError as k:
    pass
except IndexError as i:
    pass
except NameError as n:
    pass


import streamlit.components.v1 as components
# def main():
    #  νλΈλ‘ νΌλΈλ¦­μμ κ³΅μ νκΈ° λλ₯΄λ©΄ λ§ν¬ μμμλ λ΄μ₯μ½λ λ³΅μ¬ν΄μ λΆμ¬λ£κΈ° 
html_temp = """
<div class='tableauPlaceholder' id='viz1672850280379' style='position: relative'><noscript><a href='#'><img alt='λ©μΈλ°°μ± ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B7&#47;B7NJPTG2Q&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;B7NJPTG2Q' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B7&#47;B7NJPTG2Q&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='ko-KR' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1672850280379');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1600px';vizElement.style.height='927px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1600px';vizElement.style.height='927px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
#  μ€νΈλ¦¬λ° μΆλ ₯ ν¬κΈ° μ‘°μ   width κ°λ‘  heightμΈλ‘
components.html(html_temp, width=1600, height=1300)

# if __name__ == "__main__":    
#     main()