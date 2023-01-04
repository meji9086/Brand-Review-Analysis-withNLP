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
    page_icon="👗",
    layout="wide",
)

st.markdown("# 👕 브랜드를 선택해주세요. 👖")

st.sidebar.markdown("# 📌 지원하는 브랜드 종류")
st.sidebar.markdown("""
라퍼지스토어 lafudgestore            
꼼파뇨 compagno         
드로우핏 Draw fit         
인사일런스 insilence       
커버낫 covernat         
파르티멘토 partimento         
필루미네이트 filluminate       
와릿이즌 whatitisnt       
수아레 suare        
내셔널지오그래픽 nationalgeographic           
예일 yale       
디즈이스네버댓 thisisneverthat         
아웃스탠딩 outstanding        
리 lee      
어반드레스 avan         
""")

# select brand
brand_list = ['브랜드 선택', '라퍼지스토어', '꼼파뇨', '드로우핏', '인사일런스',
            '커버낫', '파르티멘토', '필루미네이트', '와릿이즌', '수아레',
            '내셔널지오그래픽', '예일', '디즈이스네버댓', '아웃스탠딩', '리',
            '어반드레스']
choice = st.selectbox('🔍보고자 하는 브랜드를 선택해주세요.', brand_list)

select_brand = []
for num in range(len(brand_list)):
    if num == 0:
        pass
    elif choice == brand_list[num]:
        st.write(f'{brand_list[num]}를 선택하셨군요. ⏳ 잠시만 기다려 주세요.')
        select_brand.append(f'{brand_list[num]}')
    

# data load
brand_link = {
    '라퍼지스토어' : '1fr6RZGM_vd5L0IDS0XIJCcScn_QtFKYz',
    '드로우핏' : '1KoWlv8cQXI9kpmfVL1pSr6jLH1RvkXkt',
    '커버낫' : '1we5q5975vDb2iNrmTPfDNsCNrQxrrWTQ',
    '파르티멘토' : '1D1BhygdkEvZU4uQQ1Y450zpZVHbdDiwc',
    '필루미네이트' : '190VQjL5F-8KPxQlYj3_8pY9Fb9JzmPIi',
    '꼼파뇨' : '1AjkzWni2Lp1V2vzhe3-gwzjDT1ASCsA2',
    '인사일런스' : '12vrFmoKeJ_UHaXljvsO1l4rbvV1Tw4Dq' ,
    '와릿이즌' : '1C_hPRBxb0sp6bJdVV4OEmNmMWFqYILQZ' ,
    '수아레' : '1KZSMUTjqqGGMVOp5KiH7X_n23IcJL3wB',
    '내셔널지오그래픽' : '1gWhGfIluszy8oVO-RwZIKtxJ0RM8yMhn',
    '예일' : '1eIoxQZrDLhsCWEl1-NJ9yE3Biem3VIcG',    
    '디즈이스네버댓' : '1ZgW4xBoXcNczxjMdw6YMKVEdjtgsUoyK',    
    '아웃스탠딩' : '1KfcwqNRRbKLgCNvgMIgZiRLaJ8Si8BJU',
    '리' : '1fqw2TiNEDxyrkaSDIlcxQ6UUyUD38kgW',
    '어반드레스' : '1YmNK_XSR03fcKgnt6ZOmWkAusXteV8tT' 
}

@st.cache
def data_load(select_brand):
    data_link = 'https://drive.google.com/uc?id='+brand_link[select_brand]
    data = pd.read_csv(data_link) 
    return data

  
try : 
    data_load_state = st.text('Loading data...') 
    data = data_load(select_brand[0])
    pos_data = data[data["평점"] == "100%"]
    neg_data = data[(data["평점"] == "20%") | (data["평점"] == "40%")]
except KeyError as k:
    pass
except IndexError as i:
    pass

# labeling
def labeling(data):
    df = data[["리뷰", "평점"]]
    df = df.reset_index(drop=True)

    삭제 = df[(df["평점"]=="60%") | (df["평점"]=="80%")].index
    df = df.drop(삭제)

    df.loc[(df["평점"] == "100%"), "label"] = 1
    df.loc[(df["평점"] == "20%") | (df["평점"] == "40%"), "label"] = 0

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# 문자 전처리
def preprocessing(text):
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA]', " ", text)
    text = re.sub('[\s]+', " ", text)
    text = text.lower()
    return text

try :
    label_data = labeling(data=data)
    positive = label_data[(label_data["평점"] == "100%")].sample(10)
    negative = label_data[(label_data["평점"] == "20%") | (label_data["평점"] == "40%")].sample(10)
    positive['리뷰'] = positive['리뷰'].map(preprocessing)
    negative['리뷰'] = negative['리뷰'].map(preprocessing)
except KeyError as k:
    pass
except NameError as n:
    pass

# Kiwi 적용
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
    print('긍정 리뷰 kiwi')
    for pos in positive['리뷰'].tolist():
        pos_nouns = noun_extractor(pos)
        pos_text = ' '.join(pos_nouns)
        pos_noun_list.append(pos_text)

    print('부정 리뷰 kiwi')
    for neg in negative['리뷰'].tolist():
        neg_nouns = noun_extractor(neg)
        neg_text = ' '.join(neg_nouns)
        neg_noun_list.append(neg_text)

except KeyError as k:
    pass
except NameError as n:
    pass

# wordrank 적용 top10 키워드 뽑기
def word_rank(corpus):
    beta = 0.90    # PageRank의 decaying factor beta
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
    '라퍼지스토어' : '1QfwKjzDAfoowpe4LiH9yxhChdZo3iizh',
    '드로우핏' : '1-caqKnBlM4Q26tec_4aWFm9017F9-_E4',
    '커버낫' : '1f50QDJI6K7KeZ7WGSANV1sH6aiXa2c5T',
    '파르티멘토' : '1Nt0LAAWlvVTh60Y9Zvbb3jmw0Jdl-URg',
    '필루미네이트' : '1CtYGt4E5hzqp-tvwix5WhANpvzds8TQK',
    '꼼파뇨' : '1-CJcWAp3WxKymk7PUxj9NYgM-3zfjM83',
    '인사일런스' : '1COUpes3WPXeGn6mdYrtzG1D9dMJ43SF7' ,
    '와릿이즌' : '1nxOa5_69KQrmbMduDhQtAPFgiclIFXO_' ,
    '수아레' : '1G5QtrNYtKNhFgRVx0Fj2-b0F626o7ccX',
    '내셔널지오그래픽' : '1Wm2ox9koFbYXkMtvLApCfQjDnPfSRsGF',
    '예일' : '1MxqLCSCptl5O5shldxK513mh4G7z3j3V',    
    '디즈이스네버댓' : '17yuM4U3W3aKKMCQphvBePiHC5mVugVkf',    
    '아웃스탠딩' : '17v-GwoTF0mgOkRta3hAhytWLPOh2YUpk',
    '리' : '1us6tb40vHoz4hrNGfoD9n2BL0fciY97n',
    '어반드레스' : '1LDAyqzM-TZZCLVrZJK08WLJe2IF6Jtxt' 
}

def img_data_load(select_brand):
    for brand in select_brand:
        img_data_link = f'https://drive.google.com/uc?id='+img_brand_link[brand]
        img_csv = pd.read_csv(img_data_link) 
    return img_csv  


def keyword_review(img_link, df, keywords):
    # 각 키워드 for문으로 돌리기
    for key_count in range(len(keywords)):
        # 키워드의 단어를 포함하는 리뷰를 keyword_review_data로 할당
        keyword_review_data = df[df['리뷰'].str.contains(keywords[key_count])]

        # 그 할당한 변수에서 상품 번호를 가져오고, 가장 많이 차지하는 top3의 상품 번호를 가져오기
        product_num = keyword_review_data['상품_num']
        top3_cumc_product = product_num.value_counts().sort_values(ascending=False)[:3].index

        # top3의 상품 번호에 해당하는 리뷰 각각 3개의 리뷰 가져오기
        review1 = keyword_review_data[keyword_review_data['상품_num']==top3_cumc_product[0]]['리뷰'].sample(3)
        review2 = keyword_review_data[keyword_review_data['상품_num']==top3_cumc_product[1]]['리뷰'].sample(3)
        review3 = keyword_review_data[keyword_review_data['상품_num']==top3_cumc_product[2]]['리뷰'].sample(3)

        st.button(keywords[key_count])
        
        for i, number in enumerate(top3_cumc_product):
            # 상품명
            product_name_list = keyword_review_data[keyword_review_data['상품_num'] == top3_cumc_product[i]]['상품']
            product_name = list(set(product_name_list))
            st.text(f'상품 옵션 : {product_name}')

            # 이미지 링크
            #imgs_link = img_link[img_link['상품_num'] == number]['사진'].values
            #join_link = ''.join(imgs_link)
            #link = f'https:{join_link}'
            #st.text(f'이미지 링크 : {link}')
            
            #image = Image.open(f'{img_link}')
            #st.image(image)

            if i==0:
                st.text(review1.values)
            if i==1:
                st.text(review2.values)
            if i==2:
                st.text(review3.values)

#            link = img_link[img_link['상품_num'] == num]['사진'].values
#            join_link = ''.join(link)
#            URL = f'https:{join_link}'
#            response = requests.get(URL)
#            image = Image.open(BytesIO(response.content))
#            st.image(image, caption='Image from URL')
        

try :
    link_csv = img_data_load(select_brand)
    st.markdown("""## 🙂 긍정 리뷰 핵심 키워드""")
    pos = keyword_review(link_csv, pos_data, pos_keyword)
    st.text(pos)
    st.markdown("""## 🙁 부정 리뷰 핵심 키워드""")
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
    #  태블로 퍼블릭에서 공유하기 누르면 링크 위에있는 내장코드 복사해서 붙여넣기 
html_temp = """
<div class='tableauPlaceholder' id='viz1672850280379' style='position: relative'><noscript><a href='#'><img alt='메인배성 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B7&#47;B7NJPTG2Q&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;B7NJPTG2Q' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B7&#47;B7NJPTG2Q&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='ko-KR' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1672850280379');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1600px';vizElement.style.height='927px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1600px';vizElement.style.height='927px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
#  스트리밋 출력 크기 조절  width 가로  height세로
components.html(html_temp, width=1600, height=1300)

# if __name__ == "__main__":    
#     main()