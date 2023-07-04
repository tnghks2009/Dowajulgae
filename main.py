# 라이브러리 불러오기
pip install --upgrade pip
import streamlit as st
import pandas as pd
from PIL import Image
from haversine import haversine
import folium
from streamlit_folium import folium_static
import joblib
import pickle
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

############################################## 사업 예측 START ##############################################

# 데이터 불러오기
total_market = pd.read_csv('0622_전체시장.csv').drop(['행정동명', '1km_이내_초중고수'], axis=1)
ulsan_market = total_market.loc[total_market['지자체'].str.contains('울산')]
ulsan_market = ulsan_market.assign(시군구=ulsan_market['지자체'].str[2:])

ulsan_venue = pd.read_csv('울산_공연장_좌표.csv', encoding='cp949')
ulsan_tour = pd.read_csv('울산_관광지_좌표.csv', encoding='cp949')
ulsan_bigmarket = pd.read_csv('울산_대규모점포_좌표.csv', encoding='cp949')
ulsan_library = pd.read_csv('울산_도서관_좌표.csv', encoding='cp949')
ulsan_park = pd.read_csv('울산_도시공원_좌표.csv', encoding='cp949')
ulsan_museum = pd.read_csv('울산_박물관미술관_좌표.csv', encoding='cp949')
ulsan_bus = pd.read_csv('울산_버스정류장_좌표.csv', encoding='cp949')
ulsan_movie = pd.read_csv('울산_영화관_좌표.csv', encoding='cp949')
ulsan_parking = pd.read_csv('울산_주차장_좌표.csv', encoding='cp949').drop(['소재지도로명주소', '소재지지번주소', '관리기관명', '제공기관명'], axis=1)
ulsan_road = pd.read_csv('울산_지역특화거리_좌표.csv', encoding='cp949')

# haversine 거리 계산 함수
def get_distance(lat1, lon1, lat2, lon2):
    distance = haversine((lat1, lon1), (lat2, lon2), unit='m')
    return distance

# 첫걸음기반조성 열 추가 및 초기값 할당
ulsan_market['첫걸음기반조성'] = 0

# 조건에 따라 값을 할당
condition = (ulsan_market['1km_이내_주차장수'] < ulsan_market['1km_이내_주차장수'].mean()) & \
            (ulsan_market['1km_이내_CCTV수'] < ulsan_market['1km_이내_CCTV수'].mean()) & \
            (ulsan_market['60대이상인구'] < ulsan_market['60대이상인구'].mean())

ulsan_market.loc[condition, '첫걸음기반조성'] = 1

def predict_ulsan_market(market) :
    features = ['축제수', '1km_이내_도시공원수', '1km_이내_관광지수', '배달빈도수', '1.5km_이내_대규모점포수',
                '1km_이내_버스정류장수', '10~30대인구', '1km_이내_대학수', '실업률(％)', '1km_이내_문화시설']
    
    x = ulsan_market[features]
    model = joblib.load('rf_model.joblib')
    
    prediction = model.predict(x)
    ulsan_market['예측결과'] = prediction
    ulsan_market['예측결과'].replace({0:'청년몰활성화', 1:'디지털전통시장', 2:'문화관광형시장'}, inplace=True)
    
    condition = (ulsan_market['첫걸음기반조성'] == 1) & \
                (ulsan_market['예측결과'].isin(['문화관광형시장', '디지털전통시장']))
    ulsan_market.loc[condition, '예측결과'] = '첫걸음기반조성'
    
    pred_result = ulsan_market.loc[ulsan_market['시장명']==market]['예측결과'].unique()
    
    return pred_result[0]

############################################## 사업 예측 END ##############################################


############################################## 대시보드 구성 ##############################################

# 화면 넓게 구성
st.set_page_config(layout="wide")

st.markdown(
        """
        <style>
        @import url(//fonts.googleapis.com/earlyaccess/jejugothic.css);

        * {
            font-family: 'Jeju Gothic', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True )

# 로고 표시
logo1, empty, logo2 = st.columns([0.1, 0.85, 0.05])
logo1.image(Image.open('로고_민트ver.png'))
logo2.image(Image.open('kt_logo.png'))

# tabs 만들기 
tab1, tab2, tab3, tab4 = st.tabs(['시장 컨셉 추천 서비스', '점포 활용 추천 서비스', '이용 안내', 'KT잘나가게'])

# 시장 컨셉 추천 서비스
with tab1:
    
    # 기능 표시
    st.markdown('###')
    st.markdown("<h2 style='text-align: center;'>시장 컨셉 추천 서비스</h2>", unsafe_allow_html=True)
    st.markdown('###')
    
    # selectbox 표시
    empty, col11, empty, col12, empty, col13, empty = st.columns([0.025, 0.3, 0.025, 0.3, 0.025, 0.3, 0.025])

    # 광역시도 선택
    with col11:
        options1 = ['광역시도', '서울특별시', '부산광역시', '인천광역시', '대구광역시', '대전광역시', '광주광역시', '울산광역시', '세종특별자치시',
                    '경기도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '강원특별자치도', '제주특별자치도']
        selected_option1 = st.selectbox('시/도를 선택하세요', options1, index=7)
        
    
    # 시군구 선택
    with col12:
        if selected_option1 == '광역시도':
            options2 = ['시군구']
            selected_option2 = st.selectbox('시/군/구를 선택하세요', options2)
        
        elif selected_option1 == '울산광역시':                                                   # our target
            options2 = ['시군구', '중구', '남구', '동구', '북구', '울주군']
            selected_option2 = st.selectbox('시/군/구를 선택하세요', options2)
            
    # 시장 선택
    with col13:
        if selected_option2 == '시군구':
            options3 = ['시장명']
            selected_option3 = st.selectbox('시장을 선택하세요', options3)
            
        elif selected_option2 == '중구':
            options3 = ['시장명', '구역전시장', '중앙전통시장', '신울산종합시장', '울산시장', '옥골시장', '태화종합시장',
                         '학성새벽시장', '우정전통시장', '반구시장', '병영시장', '선우시장', '서동시장']
            market = st.selectbox('시장을 선택하세요', options3)
            
        elif selected_option2 == '울주군':
            options3 = ['시장명', '곡천공설시장', '남창옹기종기시장', '덕신1차시장', '덕신2차시장', '덕하시장(덕하공설시장)',
                        '봉계공설시장', '언양공설시장', '언양종합상가시장', '언양알프스시장']
            market = st.selectbox('시장을 선택하세요', options3)
            
        elif selected_option2 == '동구':
            option3 = ['시장명', '남목전통시장', '대송농수산물시장', '동울산종합시장', '월봉시장']
            market = st.selectbox('시장을 선택하세요', options3)
    
    
##################################################################################################
############################################## 동구 ##############################################
##################################################################################################
    
    
    # 시장 선택            
    if selected_option2 == '동구':
        
        # # radio box를 가로로 정렬하기 위한 코드
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
        # market = st.radio('시장을 선택하세요', ('남목전통시장', '대송농수산물시장', '동울산종합시장', '월봉시장'))
            
        if market == '동울산종합시장' :
            prediction_result = predict_ulsan_market(market)
            st.markdown('#')
            st.warning(f'#### {market}은 [{prediction_result}]에 적합합니다.')
            infoimg = Image.open('info.PNG')
            st.image(infoimg)
            mapimg = Image.open('map.png')
            st.image(mapimg)

        else :
            prediction_result = predict_ulsan_market(market)
            st.markdown('#')
            st.warning(f'#### {market}은 [{prediction_result}]에 적합합니다.')
            

##################################################################################################
############################################## 중구 ##############################################
##################################################################################################
            
    
    if selected_option2 == '중구':
        
        # radio box를 가로로 정렬하기 위한 코드
        # st.markdown('#')
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)  
        # market = st.radio('시장을 선택하세요', ('구역전시장', '학성새벽시장', '구역전전통시장', '성남프라자', '옥골시장', '울산시장', '중앙전통시장',
        #                                        '반구시장', '병영시장', '선우시장', '서동시장', '신울산종합시장', '태화종합시장', '우정전통시장'))

        if market == '구역전시장' :
            st.markdown('#')
            
            # 사진 표시
            col001, col002 = st.columns(2)
            with col001:
                st.image('그림1.png', use_column_width=True)
                
            with col002:
                st.image('그림2.png', use_column_width=True)
            
            col999,col998 = st.columns([0.3, 0.7]) 
            
            with col999 :
                st.image('슬라이드1_cut.png')
                
            with col998 :    
                # 예측 결과 표시
                prediction_result = predict_ulsan_market(market)
                st.markdown('#')
                st.markdown(f"<div style='background-color: #c4e3de; padding: 20px; text-align: center;'><h3>{market}은  [ {prediction_result} ] 에 적합합니다.</h3>", unsafe_allow_html=True)
                
            st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
            text = '<b><span style="color: red;">문화관광형시장</span></b> 시장 사업은?'
            styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
            st.markdown(styled_text, unsafe_allow_html=True)            
            text2 = '''지역 문화·관광자원을 연계하여 시장 고유의 특장점을 집중 육성하는 사업입니다.<br>
                       문화, 관광, 역사 등 지역특색과 연계한 시장 투어코스 개발, 체험프로그램 운영 등 문화콘텐츠를 육성하고,<br>
                       전통시장의 대표상품을 개발 또는 개발 완료된 상품의 홍보·마케팅 등 판로개척 등을 지원합니다.'''
            styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text2}</div>"
            st.markdown(styled_text2, unsafe_allow_html=True)
            st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
            st.markdown('#')
            
            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                    <style>
                    hr {
                        border: none;
                        border-top: 4px solid #c4e3de;
                        margin: 1em 0;
                        width: 100%;
                    }
                    </style>""", unsafe_allow_html=True)
            st.markdown('#')
            
            # Feature 표시
            feature = ['1km_이내_도시공원수', '1km_이내_버스정류장수', '1km_이내_관광지수', '1km_이내_문화시설', '축제수']
            
            col004, col005, col006 = st.columns(3)
            with col004 :
                
                st.markdown(f"<div style='background-color: #c4e3de; padding: 5px; text-align: center;'><h5>{feature[0]}</h5>", unsafe_allow_html=True)
                mlat, mlon = 35.55730008, 129.3284132  # 구역전시장
                m = folium.Map(location=[mlat, mlon], zoom_start=14.5, width=445, height=445)
                
                folium.Circle(location=[mlat, mlon],    # 구역전시장 좌표
                              radius = 1000,            # 1km 반경 (미터 단위)
                              color ='red',             # 원의 테두리 색상
                              fill =True,               # 원 내부를 채울지 여부
                              fill_color ='red',        # 원 내부 색상
                              opacity =0.4              # 원의 투명도
                              ).add_to(m)
                
                guyeokjeon_market = folium.Marker(location = [mlat, mlon],
                                                  popup = market,
                                                  tooltip = market,
                                                  icon = folium.Icon('red', icon='star')).add_to(m)
                # 정보 담기.
                distances = []
                for lat, lon, name, types in zip(ulsan_park['위도'], ulsan_park['경도'], ulsan_park['공원명'], ulsan_park['공원구분']):
                    distance = get_distance(mlat, mlon, lat, lon)
                    distances.append(distance)
                    
                    marker = folium.Marker(location=[lat, lon],
                                           icon=folium.Icon(icon = 'glyphicon-leaf', color='green', prefix='glyphicon'))
                    marker.add_to(m)
            
                    popup_html = f'''
                        <b>공원명:</b> {name}<br>
                        <b>공원구분:</b> {types}<br>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                
                ulsan_park['거리(m)'] = distances
                folium_static(m)
                
            with col005 :
                st.markdown(f"<div style='background-color: #c4e3de; padding: 5px; text-align: center;'><h5>{feature[1]}</h5>", unsafe_allow_html=True)
                mlat, mlon = 35.55730008, 129.3284132  # 구역전시장
                m = folium.Map(location=[mlat, mlon], zoom_start=14.5, width=445, height=445)
                
                folium.Circle(location=[mlat, mlon],    # 구역전시장 좌표
                              radius = 1000,            # 1km 반경 (미터 단위)
                              color ='red',             # 원의 테두리 색상
                              fill =True,               # 원 내부를 채울지 여부
                              fill_color ='red',        # 원 내부 색상
                              opacity =0.4              # 원의 투명도
                              ).add_to(m)
                
                guyeokjeon_market = folium.Marker(location = [mlat, mlon],
                                                  popup = market,
                                                  tooltip = market,
                                                  icon = folium.Icon('red', icon='star')).add_to(m)
                # 정보 담기.
                distances = []
                for lat, lon, name, num in zip(ulsan_bus['위도'], ulsan_bus['경도'], ulsan_bus['정류장명'], ulsan_bus['정류장번호']):
                    distance = get_distance(mlat, mlon, lat, lon)
                    distances.append(distance)
                    
                    marker = folium.Marker(location=[lat, lon],
                                           icon=folium.Icon(color='blue'))
                    
                    marker.add_to(m)
            
                    popup_html = f'''
                        <b>정류장명:</b> {name}<br>
                        <b>정류장번호:</b> {num}<br>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                    
                ulsan_bus['거리(m)'] = distances
                folium_static(m)
                
            with col006 :
                st.markdown(f"<div style='background-color: #c4e3de; padding: 5px; text-align: center;'><h5>{feature[2]}</h5>", unsafe_allow_html=True)
                mlat, mlon = 35.55730008, 129.3284132  # 구역전시장
                m = folium.Map(location=[mlat, mlon], zoom_start=14.5, width=445, height=445)
                
                folium.Circle(location=[mlat, mlon],    # 구역전시장 좌표
                              radius = 1000,            # 1km 반경 (미터 단위)
                              color ='red',             # 원의 테두리 색상
                              fill =True,               # 원 내부를 채울지 여부
                              fill_color ='red',        # 원 내부 색상
                              opacity =0.4              # 원의 투명도
                              ).add_to(m)
                
                guyeokjeon_market = folium.Marker(location = [mlat, mlon],
                                                  popup = market,
                                                  tooltip = market,
                                                  icon = folium.Icon('red', icon='star')).add_to(m)
                # 정보 담기.
                distances = []
                for lat, lon, name in zip(ulsan_tour['위도'], ulsan_tour['경도'], ulsan_tour['관광지명']):
                    distance = get_distance(mlat, mlon, lat, lon)
                    distances.append(distance)
                    
                    marker = folium.Marker(location=[lat, lon],
                                           icon=folium.Icon(icon='glyphicon-heart', color='pink', prefix='glyphicon'))
                    marker.add_to(m)
            
                    popup_html = f'''
                        <b>관광지명:</b> {name}<br>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                    
                ulsan_tour['거리(m)'] = distances
                    
                distances = []
                for lat, lon, name, info in zip(ulsan_road['위도'], ulsan_road['경도'], ulsan_road['거리명'], ulsan_road['거리소개']):
                    distance = get_distance(mlat, mlon, lat, lon)
                    distances.append(distance)
                    
                    marker = folium.Marker(location=[lat, lon],
                                           icon=folium.Icon(icon='glyphicon-heart', color='pink', prefix='glyphicon'))
                    marker.add_to(m)
            
                    popup_html = f'''
                        <b>거리명:</b> {name}<br>
                        <b>거리소개:</b> {info}<br>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                    
                ulsan_road['거리(m)'] = distances    
                folium_static(m)
                
                
            # 설명 출력 
            col007, col008, col009 = st.columns(3)

            with col007 :
                text = f'{market}의 <b><span style="color: red;">{feature[0]}</span></b> 는'
                styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '울산광역시 평균보다 약 <b><span style="color: red; font-size: 25;">5.0</span></b> 높습니다.'
                styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: right; font-size: 18px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '도시공원은 전통시장 주변에 자연적인 휴식 공간을 제공하며, 방문객들이 시장의 분위기를 즐기고 휴식을 취할 수 있는 장소로 작용합니다.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                text3 = '전통시장과 도시공원이 함께 조성되면, 환경 친화적인 관광 지역을 형성할 수 있습니다.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown('#')

            with col008 :
                text = f'{market}의 <b><span style="color: red;">{feature[1]}</span></b> 는'
                styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '울산광역시 평균보다 약 <b><span style="color: red; font-size: 25;">13.0</span></b> 높습니다.'
                styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: right; font-size: 18px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '관광객들은 대중교통을 활용하여 전통시장을 쉽게 찾아갈 수 있으며, 돌아다니는 데 필요한 교통 수단을 쉽게 이용할 수 있습니다.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                text3 = '이는 전통시장을 포함한 관광 경로의 형성을 도모하고, 관광객들의 머무름 시간과 소비량을 증가시킬 수 있습니다.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                st.markdown('#')
                
            with col009 :
                text = f'{market}의 <b><span style="color: red;">{feature[2]}</span></b> 는'
                styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '울산광역시 평균보다 약 <b><span style="color: red; font-size: 25;">4.0</span></b> 높습니다.'
                styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: right; font-size: 18px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '전통시장은 자체적인 문화적인 매력과 함께, 주변의 관광지와 연계되어 전체적인 관광 체험을 구성할 수 있습니다.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                text3 = '이는 전통시장의 경쟁력을 강화하고, 지역의 문화관광 혁신과 경제 활성화에 기여할 수 있습니다.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                st.markdown('#')


            
            
############################################## 대표 우수사례 ##############################################        
        
            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                    <style>
                    hr {
                        border: none;
                        border-top: 4px solid #c4e3de;
                        margin: 1em 0;
                        width: 100%;
                    }
                    </style>""", unsafe_allow_html=True)
            st.markdown('#')
            
        
            st.markdown("<h2 style='text-align: center;'>문화관광형시장 대표 우수사례</h2>", unsafe_allow_html=True)
            st.markdown('#')
            st.markdown('#')
            
            st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
            text0 = f'<h5>{market} 과 비슷한 특징을 보이는 시장으로 <span style="font-size: 25px; color: red;"> "화개장터" </span>가 있습니다.</h5>'
            styled_text0 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text0}</div>"
            st.markdown(styled_text0, unsafe_allow_html=True)
            text = "<h5>화개장터는 2021년 첫걸음 기반사업 선정을 시작으로 추후 문화관광형 시장으로 탈바꿈하였습니다.</h5>"
            styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text}</div>"
            st.write(styled_text, unsafe_allow_html=True)
            text2 = '<h5>경남 하동군의 대표 특산물인 약초, 나물은 전반기 대비 매출액<span style="font-size: 22px; color: red;"> 54.7%</span>, 매출 건수<span style="font-size: 22px; color: red;"> 47.8%</span>가 증가했습니다.</h5>'
            styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text2}</div>"
            st.markdown(styled_text2, unsafe_allow_html=True)
            text3 = '<h5>2023년 기준 방문객 수 또한 꾸준히 증가하며 지역 특산물, 문화 상품과 잘 연계한 모범사례로 볼 수 있습니다.</h5>'
            styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text3}</div>"
            st.markdown(styled_text3, unsafe_allow_html=True)
            st.markdown('#')
            
            empty, col400, empty = st.columns([0.05, 0.8, 0.05])
            with col400 :
                option = st.selectbox('연계 사례 선택', ('지역 상품 연계', '관광 상품 연계'))
            st.markdown('#')

            if option == '지역 상품 연계':
                empty, col40, col41, empty = st.columns([0.05, 0.6, 0.2, 0.05])

                with col40:
                    st.image('약초종합.png')

                with col41:
                    text = "특산물 시장 점유율"
                    styled_text = f"<div style='font-size: 32px; font-weight: bold; text-align: left;'>{text}</div>"
                    st.write(styled_text, unsafe_allow_html=True)
                    text1 = '<span style="color: red;">52%</span>'
                    styled_text1 = f"<div style='font-size: 42px; font-weight: bold; text-align: right;'>{text1}</div>"
                    st.write(styled_text1, unsafe_allow_html=True)
                    st.write('#')
                    text = '''<span style='font-size: 18px;'>지리산과 백운산을 통해서만 얻는 신선한 나물과 약초는 화개장터에서만 볼 수 있는 <b style="font-weight: bold;">지역 특산품</b>입니다.<br><br>
                              74개의 점포 중 39개, 즉 시장의 50% 이상의 점포가 약초와 나물을 주력 상품으로 판매하고 있으며<br><br>
                              이러한 점이 화개장터의 주요 경쟁력으로 작용하고 있습니다.</span>'''
                    st.write(text, unsafe_allow_html=True)
                    
       
            if option == '관광 상품 연계':
                empty, col42, col43, empty = st.columns([0.05, 0.6, 0.2, 0.05])

                with col42:
                    st.image('행사종합.png')

                with col43:
                    text = "관광지 접근성"
                    styled_text = f"<div style='font-size: 32px; font-weight: bold; text-align: left;'>{text}</div>"
                    st.write(styled_text, unsafe_allow_html=True)
                    text1 = '<span style="font-size: 42px; color: red;">10</span><span style="color: red;">분 이내</span>'
                    styled_text1 = f"<div style='font-size: 32px; font-weight: bold; text-align: right;'>{text1}</div>"
                    st.write(styled_text1, unsafe_allow_html=True)
                    
                    st.markdown('#')
                    text_0 = '''<span style='font-size: 16px;'>화개 장터는 시장과 관광지를 연계하는 데 적극적인 시장입니다.<br><br>
                                특히 드라마의 촬영 장소인 최참판댁은 차로 10분 이내면 갈 수 있는 관광지 중 하나입니다.<br><br>
                                4~6월 최참판댁에서는 주말마다 약 20 차례의 문화공연이 펼쳐져 인기 관광명소로 자리잡았습니다.<br><br>
                                또 다른 주변 문화재인 쌍계사까지 이어지는 벚꽃길 역시 주요 상품 중 하나입니다.<br>
                                매년 벚꽃 축제가 개최되어 많은 관광객이 방문하고 있습니다.</span>'''
                    st.write(text_0, unsafe_allow_html=True)
            
            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                    <style>
                    hr {
                        border: none;
                        border-top: 4px solid #c4e3de;
                        margin: 1em 0;
                        width: 100%;
                    }
                    </style>""", unsafe_allow_html=True)
            st.markdown('#')
            
        elif market == '시장명' :
            st.write('#')
            
        else :
            prediction_result = predict_ulsan_market(market)
            st.write('')
            st.markdown('#')
            st.warning(f'#### {market}은 [{prediction_result}]에 적합합니다.')
            
            
###################################################################################################
############################################## 울주군 ##############################################
###################################################################################################

            
    if selected_option2 == '울주군':
        
        # radio box를 가로로 정렬하기 위한 코드
        # st.markdown('#')
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)  
        # market = st.radio('시장을 선택하세요', ('곡천공설시장', '남창옹기종기시장', '덕신1차시장', '덕신2차시장', '덕하시장(덕하공설시장)', '봉계공설시장', '언양공설시장',
        #                                        '언양종합상가시장', '언양알프스시장'))

        if market == '남창옹기종기시장' :
            st.markdown('#')
            
            # 사진 표시
            col001, col002 = st.columns(2)
            with col001:
                st.image('그림3.png', use_column_width=True)
                
            with col002:
                st.image('그림4.png', use_column_width=True)
            
            col999,col998 = st.columns([0.3, 0.7]) 
            
            with col999 :
                st.image('슬라이드2_cut.png')
                
            with col998 :    
                # 예측 결과 표시
                prediction_result = predict_ulsan_market(market)
                st.markdown(f"<div style='background-color: #FCFCC8; padding: 20px; text-align: center;'><h3>{market}은  [ {prediction_result} ] 에 적합합니다.</h3>", unsafe_allow_html=True)
                
            st.markdown("<div style='background-color: #EEEEEE; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
            text = '<b><span style="color: red;">디지털전통시장</span></b> 사업은?'
            styled_text = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
            st.markdown(styled_text, unsafe_allow_html=True)            
            text2 = '''전통시장 디지털 전환을 위해 온라인 입점 및 마케팅, 배송인프라 구축, 전담인력 지원 등 인적·물적기반을 종합지원 하는 사업입니다.<br>
                       운영 조직 구성 및 역량 강화, 상품 발굴, 입점지원, 인프라 구축, 공동마케팅 비용과 함께<br>
                       배송에 필요한 픽업 및 배송 인력 등 인적 기반과 배송센터, 공동 장비 등 물적 기반 구축비용을 지원합니다.'''
            styled_text2 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text2}</div>"
            st.markdown(styled_text2, unsafe_allow_html=True)
            st.markdown("<div style='background-color: #EEEEEE; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
            st.markdown('#')
            
            # Feature 표시
            feature = ['1.5km_이내_대규모점포수', '1km_이내_주차장수', '10~30대인구']
            
            col004, col005, col006 = st.columns(3)
            with col004 :
                
                st.markdown(f"<div style='background-color: #8BB396; padding: 5px; text-align: center;'><h5>{feature[0]}🛒</h5>", unsafe_allow_html=True)
                mlat, mlon = 35.41668686, 129.2827813   # 남창옹기종기시장
                m = folium.Map(location=[mlat, mlon], zoom_start=14, width=445, height=445)
                
                folium.Circle(location=[mlat, mlon],    # 남창옹기종기시장 좌표
                              radius = 1500,            # 1.5km 반경 (미터 단위)
                              color ='red',             # 원의 테두리 색상
                              fill =True,               # 원 내부를 채울지 여부
                              fill_color ='red',        # 원 내부 색상
                              opacity =0.4              # 원의 투명도
                              ).add_to(m)
                
                ongi_market = folium.Marker(location = [mlat, mlon],
                                                  popup = market,
                                                  tooltip = market,
                                                  icon = folium.Icon('red', icon='star')).add_to(m)
                # 정보 담기.
                distances = []
                for lat, lon, name, info in zip(ulsan_bigmarket['Y'], ulsan_bigmarket['X'], ulsan_bigmarket['사업장명'], ulsan_bigmarket['업태구분명']):
                    distance = get_distance(mlat, mlon, lat, lon)
                    distances.append(distance)
                    
                    marker = folium.Marker(location=[lat, lon],
                                           icon=folium.Icon(icon = 'glyphicon-shopping-cart', color='black', prefix='glyphicon'))
                    marker.add_to(m)
            
                    popup_html = f'''
                        <b>사업장명:</b> {name}<br>
                        <b>업태구분명:</b> {info}<br>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                
                ulsan_bigmarket['거리(m)'] = distances
                folium_static(m)
                
            with col005 :
                st.markdown(f"<div style='background-color: #8BB396; padding: 5px; text-align: center;'><h5>{feature[1]}🅿</h5>", unsafe_allow_html=True)
                mlat, mlon = 35.41668686, 129.2827813   # 남창옹기종기시장
                m = folium.Map(location=[mlat, mlon], zoom_start=14.5, width=445, height=445)
                
                folium.Circle(location=[mlat, mlon],    # 남창옹기종기시장 좌표
                              radius = 1000,            # 1km 반경 (미터 단위)
                              color ='red',             # 원의 테두리 색상
                              fill =True,               # 원 내부를 채울지 여부
                              fill_color ='red',        # 원 내부 색상
                              opacity =0.4              # 원의 투명도
                              ).add_to(m)
                
                ongi_market = folium.Marker(location = [mlat, mlon],
                                                  popup = market,
                                                  tooltip = market,
                                                  icon = folium.Icon('red', icon='star')).add_to(m)
                # 정보 담기.
                distances = []
                for lat, lon, name, cost in zip(ulsan_parking['위도'], ulsan_parking['경도'], ulsan_parking['주차장명'], ulsan_parking['요금정보']):
                    distance = get_distance(mlat, mlon, lat, lon)
                    distances.append(distance)
                    
                    marker = folium.Marker(location=[lat, lon],
                                           icon=folium.Icon(icon = 'glyphicon-road', color='blue', prefix='glyphicon'))
                    
                    marker.add_to(m)
            
                    popup_html = f'''
                        <b>주차장명:</b> {name}<br>
                        <b>요금정보:</b> {cost}<br>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                    
                ulsan_parking['거리(m)'] = distances
                folium_static(m)
                
            with col006 :
                st.markdown(f"<div style='background-color: #8BB396; padding: 5px; text-align: center;'><h5>{feature[2]}👨‍👩‍👧‍👦</h5>", unsafe_allow_html=True)
                st.image('population.png', use_column_width=True)
                
                # 정보 담기.
                
                
                
            # 설명 출력 
            col007, col008, col009 = st.columns(3)

            with col007 :
                text = f'{market}의 <b><span style="color: red;">{feature[0]}</span></b> 는'
                styled_text = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)
                text2 = '울산광역시 평균보다 약 <b><span style="color: red;">2.28</span></b> 낮습니다.'
                styled_text2 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '전통시장의 소상공인들이 지역 배달 서비스와 협력하여 상호 유기적인 관계를 형성하면, 지역 경제에 활기를 불어넣고 일자리 창출에 기여할 수 있습니다.'
                styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                text3 = '나아가 전통시장의 특색과 고유한 매력을 강조한 배달 서비스나 패키지를 개발하여 고객들에게 다양한 경험을 제공함으로써 시장의 경쟁력을 향상시킬 수 있습니다.'
                styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown('#')

            with col008 :
                text = f'{market}의 <b><span style="color: red;">{feature[1]}</span></b> 는'
                styled_text = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '울산광역시 평균보다 약 <b><span style="color: red;">11.13</span></b> 낮습니다.'
                styled_text2 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '주차 공간이 제한적인 전통시장에서는 고객들이 직접 차량으로 이동하여 구매를 하기보다는 배달 서비스를 활용하는 경우가 많습니다.'
                styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                text3 = '디지털전통시장 사업을 통해 상인들은 온라인 주문을 받고, 주변 지역으로 배달하는 방식으로 고객들에게 상품을 제공하여 소비자들의 접근성과 편의성을 높일 수 있습니다.'
                styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown('#')
                
            with col009 :
                text = f'{market}인근 <b><span style="color: red;">{feature[2]}</span></b> 는'
                styled_text = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '울산광역시 내의 다른 행정동 보다 약 <b><span style="color: red;">3373</span></b> 높습니다.'
                styled_text2 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '10~30대는 디지털매체에 능숙한 세대로, 온라인 쇼핑과 모바일 애플리케이션을 통한 서비스 이용에 익숙합니다.'
                styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                text3 = '디지털전통시장 사업을 통해 디지털 네이티브 소비자 층을 타겟팅하여 온라인 주문, 배달 서비스, 온라인 마케팅 등의 디지털 전략을 구사할 수 있습니다.'
                styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: left; font-size: 15px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown("<div style='background-color: #EEEEEE; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                st.markdown('#')
                st.markdown('#')
                st.markdown('#')
                
############################################## 대표 우수사례 ##############################################        
        

            st.markdown("<h1 style='text-align: center;'>디지털전통시장 대표 우수사례</h1>", unsafe_allow_html=True)
            st.markdown('#')
            st.markdown('#')
            text = "서울특별시 강동구에 위치한 암사종합시장은 디지털전통시장 사업과 지역 특산물을 잘 연계한 대표적인 시장입니다."
            styled_text = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: center; font-size: 18px;'>{text}</div>"
            st.write(styled_text, unsafe_allow_html=True)
            text2 = '디지털 역량 부문에서 우수 사례에서 뽑혔습니다.'
            styled_text2 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: center; font-size: 18px;'>{text2}</div>"
            st.markdown(styled_text2, unsafe_allow_html=True)
            text3 = '소상공인진흥공단에서 발표한 사례집에서도 소개될 정도로 30~40대 연령층과 1~2인 가구들에게 매력적인 시장으로 손꼽혔습니다.'
            styled_text3 = f"<div style='background-color: #EEEEEE; padding: 10px; text-align: center; font-size: 18px;'>{text3}</div>"
            st.markdown(styled_text3, unsafe_allow_html=True)
            st.markdown('#')

            col52, col53 = st.columns([0.6, 0.3])
        
            with col52:
                st.image('암사종합.png')

            with col53:
                text = "배송 서비스 <span style='color: red;'>시장</span>"
                styled_text = f"<p style='font-size: 24px; font-weight: bold;'>{text}</p>"
                st.write(styled_text, unsafe_allow_html=True)
                st.markdown('1. 지역배달 : 전국 배송. 설, 추석 같은 명절에 맞게 배송 서비스를 제공하고 있습니다.')
                st.markdown('2. 배송, 홍보, 고객관리 지원센터 구축 : 라이브커머스 진행 스튜디오, 고객센터')
                st.markdown('3. 온라인 위주 데이터를 통해서 고객관리에 힘쓴다.')
            
        else :
            prediction_result = predict_ulsan_market(market)
            st.write('')
            st.markdown('#')
            st.warning(f'#### {market}은 [{prediction_result}]에 적합합니다.')
            

        
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================
# =================================================================================================================


# 점포 활용 추천 서비스
with tab2:
    
    # Inject custom CSS styles
    CSS = st.markdown("""
                    <style>
                        div[data-baseweb="select"] > div {
                        background-color: #c4e3de;
                        }
                    </style>""", unsafe_allow_html=True)
    # 제목 표시
    st.markdown("<h2 style='text-align: center;'>점포 활용 추천 서비스</h2>", unsafe_allow_html=True)
    st.markdown('###')
    
    # selectbox 표시
    empty, col21, empty, col22, empty, col23, empty = st.columns([0.025, 0.3, 0.025, 0.3, 0.025, 0.3, 0.025])

    # 광역시도 선택
    with col21:
        options1 = ['광역시도', '울산광역시', '서울특별시', '부산광역시', '인천광역시', '대구광역시', '대전광역시', '광주광역시', '세종특별자치시',
                    '경기도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '강원특별자치도', '제주특별자치도']
        selected_option1 = st.selectbox('시/도를 선택하세요  ', options1, index=1)
    
    # 시군구 선택
    with col22:
        if selected_option1 == '광역시도':
            options2 = ['시군구']
            selected_option2 = st.selectbox('시/군/구를 선택하세요', options2)
            
        if selected_option1 == '울산광역시':
            options2 = ['시군구', '중구', '남구', '동구', '북구', '울주군']
            selected_option2 = st.selectbox('시/군/구를 선택하세요  ', options2)
            
    # 시장 선택
    with col23:
        if selected_option2 == '시군구':
            options3 = ['시장명']
            selected_option3 = st.selectbox('시장을 선택하세요 ', options3)
            
        if selected_option2 == '중구':
            options3 = ['시장명', '중앙전통시장', '구역전시장', '신울산종합시장', '울산시장', '옥골시장', '태화종합시장',
                         '학성새벽시장', '우정전통시장', '반구시장', '병영시장', '선우시장', '서동시장']
            selected_option3 = st.selectbox('시장을 선택하세요', options3)
            
        if selected_option2 == '남구':
            options3 = ['시장명', '신정평화시장', '신정시장', '수암상가시장', '수암종합시장', '야음상가시장', '울산번개시장']
            selected_option3 = st.selectbox('시장을 선택하세요', options3)
    
# ----------------------------------------------------------------------------------------------------------------------
    
    if selected_option3 == '중앙전통시장':
        
        # Expander Style
        st.markdown(
                    '''
                    <style>
                    .streamlit-expanderHeader {
                        background-color: #c4e3de;
                        color: black; # Adjust this for expander header color; }
                    .streamlit-expanderContent {
                        background-color: white;
                        color: black; # Expander content color }
                    </style>
                    ''', unsafe_allow_html=True )
        
        # 사진 표시
        st.markdown('#')
        img1, img2= st.columns(2)
        img1.image(Image.open('중앙전통시장_.png'), use_column_width=True)
        img2.image(Image.open('중앙전통시장3.png'), use_column_width=True)
        st.markdown('#')
        
        # Line
        # st.markdown('#')
        # st.markdown("<hr>", unsafe_allow_html=True)
        # st.markdown("""
        #             <style>
        #             hr {
        #                 border: none;
        #                 border-top: 4px solid #c4e3de;
        #                 margin: 1em 0;
        #                 width: 100%;
        #             }
        #             </style>""", unsafe_allow_html=True)
        # st.markdown('#')
        
        # st.image('season.png')
        
        st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
        text2 = '최신 소비 트렌드를 살펴보니 　<b><span style="font-size: 25px;color: red;">1. 강아지 우비 　2. 여성장화 　3. 레인부츠</span></b>　아이템을 추천해요.'
        styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 20px;'>{text2}</div>"
        st.markdown(styled_text2, unsafe_allow_html=True)
        text3 = '<b><span style="font-size: 22px;">20</b><b>대</b> 를 대상으로 <b><span style="color: red;">온라인</span></b> 프로모션을, <b><span style="font-size: 22px;">50</span></b><b>대 이상</b>을 대상으로 <b><span style="color: red;">오프라인</span></b> 프로모션을 추천해요.'
        styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 20px;'>{text3}</div>"
        st.markdown(styled_text3, unsafe_allow_html=True)
        text4 = '이번 여름에는 <b><span style="color: red;">태화강 마두희 축제</span></b>를 비롯한 축제와 관련한 상품을 개발 해보세요.'
        styled_text4 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 20px;'>{text4}</div>"
        st.markdown(styled_text4, unsafe_allow_html=True)
        st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
        
        # Line
        st.markdown('#')
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
                    <style>
                    hr {
                        border: none;
                        border-top: 4px solid #c4e3de;
                        margin: 1em 0;
                        width: 100%;
                    }
                    </style>""", unsafe_allow_html=True)
        st.markdown('#')
        
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- 유동인구 분석 결과 -------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        with st.expander('유동인구 분석') :
            # 유동인구 분석
            st.markdown('#')
            st.markdown("<div style='background-color: #c4e3de; padding: 10px; text-align: center;'><h4>유동인구 분석 결과</h4>", unsafe_allow_html=True)
            st.markdown('#')

            # 유동인구 그래프 표시
            col24, col25, col26 = st.columns(3)

            # 연령대 분포 막대 그래프
            with col24:
                st.markdown('#')
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>연령대 분포</b></p>", unsafe_allow_html=True)
                age = pd.read_excel('age_ratio.xlsx', index_col=None)
                fig = px.bar(age, x='age', y='ratio', text='ratio')
                fig.update_traces(texttemplate='%{text:.0f}%')  # 숫자에 '%' 추가
                fig.update_layout(xaxis_title=None, yaxis_title=None)  # 축 이름 표시 안 함
                last_bar_index = len(fig.data[0].y) - 1
                colors = ['#D8D8D8'] * last_bar_index + ['#99A98F']  # 색상 설정
                fig.update_traces(marker_color=colors)
                fig.update_layout(width=400, height=400)  # 원하는 크기로 수정
                st.plotly_chart(fig)

            # 시간대 분포 라인 그래프
            with col25:
                st.markdown('#')
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>시간대 분포</b></p>", unsafe_allow_html=True)
                day_time = pd.read_excel('day_time_ratio.xlsx', index_col=None)
                fig = px.line(day_time, x='time', y=['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
                fig.update_layout(xaxis_title=None, yaxis_title=None)  # 축 이름 표시 안 함
                colors = px.colors.qualitative.Pastel  # 색상 설정
                for i, line_name in enumerate(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']):
                    fig['data'][i]['line']['color'] = colors[i % len(colors)]
                fig.update_layout(width=460, height=400)  # 원하는 크기로 수정
                st.plotly_chart(fig)

            # 인구 분포 도넛 그래프
            with col26:
                st.markdown('#')
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>인구 분포</b></p>", unsafe_allow_html=True)
                people = pd.read_excel('people_ratio.xlsx', index_col=None)
                fig = px.pie(people, values='ratio', names='people', hole=.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(font=dict(size=16))
                fig.update(layout_showlegend=False)  # 범례 표시 제거
                colors = ['#9CB4CC', '#748DA6']  # 색상 설정
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=450, height=400)  # 원하는 크기로 수정
                st.plotly_chart(fig)

    # ----------------------------------------------------------------

            # 그래프 분석 내용
            empty, col27, empty, col28, col29, empty = st.columns([0.05, 0.5, 0.04, 0.06, 0.3, 0.05])

            # 유동인구 분석
            with col27:
                st.markdown('#')
                st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                text = '유동인구 분석 결과, <b><span style="color: red;">60대 이상</span></b>이 가장 많고 그 뒤로 <b>20대</b>와 <b>50대</b>가 많아요.'
                styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '<b><span style="color: red;">부모/자녀</span></b> 관계를 가진 유동인구가 많이 방문하는 것으로 예측돼요.'
                styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '<b><span style="color: red;">토요일 저녁</span></b> 시간대를 활용하는 것이 효과적일 것으로 보여요.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백

            # 이미지 표시
            with col28:
                st.markdown('#')
                col28.image(Image.open('instagram.png'))
                col28.image(Image.open('brochure.png'))

            # 프로모션 추천
            with col29:
                st.markdown('#')
                st.markdown(' ')
                text4 = "<b>20대</b>가 많이 이용하는 인스타그램을 활용해<br> <b><span style='color: red;'>온라인</span></b> 프로모션을 추천해요."
                styled_text4 = f"<div style='text-align: center; font-size: 17px;'>{text4}</div>"
                st.markdown(styled_text4, unsafe_allow_html=True)
                st.markdown('#')
                text5 = "<b>5~60대</b>를 대상으로 전단지/팜플렛을 배포해<br> <b><span style='color: red;'>오프라인</span></b> 프로모션을 추천해요."
                styled_text5 = f"<div style='text-align: center; font-size: 17px;'>{text5}</div>"
                st.markdown(styled_text5, unsafe_allow_html=True)

            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        hr {
                            border: none;
                            border-top: 4px solid #c4e3de;
                            margin: 1em 0;
                            width: 100%;
                        }
                        </style>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- 네이버 쇼핑 트렌드 분석 -------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        with st.expander('네이버 쇼핑 트렌드 분석') :
            # 네이버 쇼핑 트렌드 분석
            st.markdown('#')
            st.markdown("<div style='background-color: #c4e3de; padding: 10px; text-align: center;'><h4>네이버 쇼핑 트렌드 분석</h4>", unsafe_allow_html=True)
            st.markdown('#')
            st.markdown('#')
            
            # 카테고리 선택 및 시장 정보 표시
            empty, col31, col32, col33, col34 = st.columns([0.05, 0.20, 0.25, 0.25, 0.25])

            with col31:                
                st.markdown('#')
                st.markdown('#')
                options41 = ['전체', '패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어', 
                             '출산/육아', '식품', '스포츠/레저', '생활/건강', '여가/생활편의']
                selected_option41 = st.selectbox('카테고리를 선택하세요', options41)
                st.markdown('#')
                
                if 'button' not in st.session_state:
                    st.session_state.button = False
                def click_button():
                    st.session_state.button = not st.session_state.button
                    
                # Button Style
                st.markdown("""
                            <style>
                                div.stButton > button:first-child { background-color: #f5ede3;
                                                                    color:black; }
                            </style>
                            """, unsafe_allow_html=True)
                st.button('주 고객 소비트렌드 확인', on_click = click_button, use_container_width=True)

            # 빈점포 수 그래프
            with col32:
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>빈점포 수</b></p>", unsafe_allow_html=True)
                empty_store = pd.read_excel('empty_store.xlsx', index_col=None)
                fig = px.pie(empty_store, values='empty', names='store')
                fig.update_traces(textposition='inside', textinfo='value', insidetextfont=dict(size=100))
                fig.update_layout(font=dict(size=16, color='white'))
                fig.update(layout_showlegend=False)       # 범례 표시 제거
                colors = ['#18564d']                      # 색상 설정
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=320, height=300)  # 원하는 크기로 수정
                st.plotly_chart(fig)

            # 주 고객 연령층
            with col33:
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>주 고객 연령층</b></p>", unsafe_allow_html=True)
                image = Image.open('sixty.png')
                st.image(image, use_column_width=True)

            # 성별 분포 도넛 그래프
            with col34:
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>성별 분포</b></p>", unsafe_allow_html=True)
                gender = pd.read_excel('gender_ratio.xlsx', index_col=None)
                fig = px.pie(gender, values='ratio', names='gender', hole=.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(font=dict(size=13))
                fig.update(layout_showlegend=False)  # 범례 표시 제거
                colors = ['#B3E5F2', '#FECCCF']  # 색상 설정
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=300, height=300)  # 원하는 크기로 수정
                st.plotly_chart(fig)
                
            # 카테고리에 따른 순위 출력
            if st.session_state.button :
                    
                # 소비 트랜드 순위
                naver = pd.read_csv('0629_네이버쇼핑.csv', index_col=None)

                # 순위 출력 함수
                def display_ranking(category):
                    text6 = "<h4>60대 소비 트랜드</h4>"
                    styled_text6 = f"<div style='padding: 10px; text-align: center;'>{text6} [{category}]</div>"
                    st.markdown(styled_text6, unsafe_allow_html=True)            

                    empty, col35, empty, col36, empty = st.columns([0.1, 0.35, 0.1, 0.35, 0.1])

                    # 순위 1~10위
                    with col35:
                        df = naver.loc[naver['분류']=='A00', [category]].head(10)
                        df.index = [f'{i}위' for i in range(1, 11)]
                        df['비고'] = np.where(df[category].str.contains('NEW'), 'NEW 🔺', '')
                        df[category] = df[category].str.replace('NEW', '')
                        st.dataframe(df, use_container_width=True)

                    # 순위 11~20위
                    with col36:
                        dff = naver.loc[naver['분류'] == 'A00', [category]].tail(10)
                        dff.index = [f'{i}위' for i in range(11, 21)]
                        dff['비고'] = np.where(dff[category].str.contains('NEW'), 'NEW 🔺', '')
                        dff[category] = dff[category].str.replace('NEW', '')
                        st.dataframe(dff, use_container_width=True)
                    
                if selected_option41 == '전체':
                    display_ranking('전체')
                elif selected_option41 == '패션의류':
                    display_ranking('패션의류')
                elif selected_option41 == '패션잡화':
                    display_ranking('패션잡화')
                elif selected_option41 == '화장품/미용':
                    display_ranking('화장품/미용')
                elif selected_option41 == '디지털/가전':
                    display_ranking('디지털/가전')
                elif selected_option41 == '가구/인테리어':
                    display_ranking('가구/인테리어')
                elif selected_option41 == '출산/육아':
                    display_ranking('출산/육아')
                elif selected_option41 == '식품':
                    display_ranking('식품')
                elif selected_option41 == '스포츠/레저':
                    display_ranking('스포츠/레저')
                elif selected_option41 == '생활/건강':
                    display_ranking('생활/건강')
                elif selected_option41 == '여가/생활편의':
                    display_ranking('여가/생활편의')
                        
            else:
                st.write('')

            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        hr {
                            border: none;
                            border-top: 4px solid #c4e3de;
                            margin: 1em 0;
                            width: 100%;
                        }
                        </style>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- 축제 정보 --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        
        with st.expander('울산광역시 축제 정보') :
            # 울산광역시 축제 정보
            st.markdown('#')
            st.markdown("<div style='background-color: #c4e3de; padding: 10px; text-align: center;'><h4>울산광역시 축제 정보</h4>", unsafe_allow_html=True)
            st.markdown('#')
            st.markdown('#')

            empty, col40, empty = st.columns([0.05, 0.53, 0.42])

            # 거리에 따른 슬라이드바 형성
            with col40:
                distance = st.slider("전통시장 반경 거리(km)를 설정하세요", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
                st.markdown('###')

            # 지도 표시    
            empty, col41, col42, empty = st.columns([0.05, 0.57, 0.33, 0.05])
            ulsan_festival = pd.read_csv('울산광역시_문화축제데이터.csv', encoding='cp949')

            with col41:
                lat, lon = 35.5544754, 129.323146         # 중앙전통시장 좌표
                m = folium.Map(location=[lat, lon], zoom_start=13)

                folium.Circle(
                    location = [35.5544754, 129.323146],  # 중앙전통시장 좌표
                    radius = distance * 1000,             # 1km 반경 (미터 단위)
                    color = 'red',                        # 원의 테두리 색상
                    fill = True,                          # 원 내부를 채울지 여부
                    fill_color = 'red',                   # 원 내부 색상
                    opacity = 0.1,                        # 원의 투명도
                ).add_to(m)

                bangu_market = folium.Marker(
                    location = [35.5544754, 129.323146],  # 중앙전통시장 좌표
                    popup = '울산 중앙전통시장',
                    tooltip = '울산 중앙전통시장',
                    icon = folium.Icon('red'),
                ).add_to(m)

                #축제 정보 담기.
                for lat, lon, name, contents, date, youtube_link in zip(ulsan_festival['위도'], ulsan_festival['경도'], ulsan_festival['축제명'],
                                                                        ulsan_festival['축제내용'], ulsan_festival['축제시작일자'], ulsan_festival['유튜브']):
                    marker = folium.Marker(
                        location=[lat, lon],
                        icon=folium.Icon(icon = 'star', color='pink')
                    )
                    marker.add_to(m)

                    popup_html = f'''
                        <b>축제명:</b> {name}<br>
                        <b>축제내용:</b> {contents}<br>
                        <b>축제시작일자:</b> {date}<br>
                        <a href="{youtube_link}" target="_blank">YouTube 동영상 보기</a>
                        <br>
                        <iframe width="560" height="315" src="https://www.youtube.com/embed/{youtube_link.split("=")[1]}" frameborder="0" allowfullscreen></iframe>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                folium_static(m)

            with col42:
                location = [35.5544754, 129.323146]  # 중앙전통시장 좌표
                festival_lat = ulsan_festival['위도'].tolist()
                festival_lon = ulsan_festival['경도'].tolist()

                # 각 축제 위치와 중앙전통시장의 거리 계산
                distances = [haversine((lat, lon), (location[0], location[1]), unit='km') for lat, lon in zip(festival_lat, festival_lon)]
                ulsan_festival['거리'] = [round(dist, 2) for dist in distances]
                filtered_festivals = ulsan_festival[ulsan_festival['거리'] <= distance]
                filtered_festivals.sort_values(by='거리', ascending=True, inplace=True)
                filtered_festivals.drop(['위도', '경도', '유튜브', '소재지도로명주소', '전화번호'], axis=1, inplace=True)

                # 선택한 거리 안에 있는 축제 표로 나타내기
                st.write("선택한 거리 안에 있는 축제 리스트")
                st.dataframe(filtered_festivals)

    # ----------------------------------------------------------------

            # 전통시장과 지역 축제 Report
            empty, col43, empty = st.columns([0.1, 0.8, 0.1])

            with col43:
                st.markdown('#')
                st.markdown('#### 전통시장과 지역축제 협력 사례')
                st.markdown('###')
                st.image(Image.open('레포트.png'))

                col44, empty, col45 = st.columns([0.455, 0.030, 0.515])

                with col44:
                    text = '전통시장은 지역의 특색과 문화를 살려 축제를 개최함으로써 관광객들에게 지방색을 전달하는 중요한 역할을 할 수 있습니다.'
                    styled_text = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)            
                    text2 = '인천시는 전통시장을 중심으로 한 지역축제를 개최하여 관광객의 발걸음을 성공적으로 유치하고 지역 경제 활성화를 이루어내는 모범 사례로 평가받았습니다.'
                    styled_text2 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text2}</div>"
                    st.markdown(styled_text2, unsafe_allow_html=True)
                    text3 = '인천시의 전통시장을 활용해 다양한 프로그램과 특산물을 축제에 접목하여 방문객들의 호응을 이끌어내고 시장 활성화를 도모하는 성과를 얻었습니다.'
                    styled_text3 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text3}</div>"
                    st.markdown(styled_text3, unsafe_allow_html=True)
                    text4 = '전통시장과 축제의 협력은 지역 경제 활성화와 관광 산업 발전에 기여하는 효과가 있으며, 이는 다른 지자체에서도 참고할 수 있는 좋은 사례로 간주됩니다.'
                    styled_text4 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text4}</div>"
                    st.markdown(styled_text4, unsafe_allow_html=True)

                with col45:
                    text = ' <b>1. 지역 브랜드 판매 및 홍보</b><br> 빈점포에 유명 특산물이나 지역 고유의 제품을 전시하고 판매하는 것을 추천합니다. 이를 통해 지역 상인들의 수익 증대와 지역 브랜드의 홍보를 동시에 이끌어낼 수 있습니다.'
                    styled_text = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)            
                    text2 = ' <b>2. 체험형 부스</b><br> 관광객들이 지역의 전통 문화나 공예를 직접 체험하고 참여할 수 있는 부스를 추천합니다. 이러한 체험을 통해 문화 교류와 지속 가능한 관광 발전에 기여할 수 있습니다.'
                    styled_text2 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text2}</div>"
                    st.markdown(styled_text2, unsafe_allow_html=True)
                    text3 = ' <b>3. 문화 예술 공간</b><br> 축제와 함께 예술 문화 행사를 연계하는 지역 예술가들의 작품 전시나 공연 개최를 추천합니다. 이를 통해 관광객들에게 풍부한 문화적인 경험과 예술적 감동을 선사할 수 있습니다.'
                    styled_text3 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text3}</div>"
                    st.markdown(styled_text3, unsafe_allow_html=True)
                    st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                    st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                    
                # Line
                st.markdown('#')
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("""
                            <style>
                            hr {
                                border: none;
                                border-top: 4px solid #c4e3de;
                                margin: 1em 0;
                                width: 100%;
                            }
                            </style>""", unsafe_allow_html=True)
                
# =============================================================================================================================================================================================
# =============================================================================================================================================================================================   
    
    if selected_option3 == '신정평화시장':

        # 사진 표시
        st.markdown('#')
        img1, img2= st.columns(2)
        img1.image(Image.open('신정평화시장_.png'), use_column_width=True)
        img2.image(Image.open('신정평화시장3.png'), use_column_width=True)
        
        # Line
        st.markdown('#')
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
                    <style>
                    hr {
                        border: none;
                        border-top: 4px solid #c4e3de;
                        margin: 1em 0;
                        width: 100%;
                    }
                    </style>""", unsafe_allow_html=True)
        st.markdown('#')
        
        # Expander Style
        st.markdown(
                    '''
                    <style>
                    .streamlit-expanderHeader {
                        background-color: #c4e3de;
                        color: black; # Adjust this for expander header color }
                    .streamlit-expanderContent {
                        background-color: white;
                        color: black; # Expander content color }
                    </style>
                    ''', unsafe_allow_html=True )

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- 유동인구 분석 결과 -------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        with st.expander('유동인구 분석') :
            
            # 프로모션 방법 추천
            st.markdown('#')
            st.markdown("<div style='background-color: #c4e3de; padding: 10px; text-align: center;'><h4>유동인구 분석 결과</h4>", unsafe_allow_html=True)
            st.markdown('#')

            # 유동인구 그래프 표시
            col24, col25, col26 = st.columns(3)
        
            # 연령대 분포 막대 그래프
            with col24:
                st.markdown('#')
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>연령대 분포</b></p>", unsafe_allow_html=True)
                age = pd.read_excel('age_ratio2.xlsx', index_col=None)
                fig = px.bar(age, x='age', y='ratio', text='ratio')
                fig.update_traces(texttemplate='%{text:.0f}%')  # 숫자에 '%' 추가
                fig.update_layout(xaxis_title=None, yaxis_title=None)  # 축 이름 표시 안 함
                last_bar_index = len(fig.data[0].y) - 1
                colors = ['#D8D8D8'] * last_bar_index + ['#99A98F']  # 색상 설정
                fig.update_traces(marker_color=colors)
                fig.update_layout(width=400, height=400)  # 원하는 크기로 수정
                st.plotly_chart(fig)
        
            # 시간대 분포 라인 그래프
            with col25:
                st.markdown('#')
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>시간대 분포</b></p>", unsafe_allow_html=True)
                day_time = pd.read_excel('day_time_ratio2.xlsx', index_col=None)
                fig = px.line(day_time, x='time', y=['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
                fig.update_layout(xaxis_title=None, yaxis_title=None)  # 축 이름 표시 안 함
                colors = px.colors.qualitative.Pastel  # 색상 설정
                for i, line_name in enumerate(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']):
                    fig['data'][i]['line']['color'] = colors[i % len(colors)]
                fig.update_layout(width=460, height=400)  # 원하는 크기로 수정
                st.plotly_chart(fig)
        
            # 인구 분포 도넛 그래프
            with col26:
                st.markdown('#')
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>인구 분포</b></p>", unsafe_allow_html=True)
                people = pd.read_excel('people_ratio2.xlsx', index_col=None)
                fig = px.pie(people, values='ratio', names='people', hole=.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(font=dict(size=16))
                fig.update(layout_showlegend=False)  # 범례 표시 제거
                colors = ['#9CB4CC', '#748DA6']  # 색상 설정
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=450, height=400)  # 원하는 크기로 수정
                st.plotly_chart(fig)
        
# ---------------------------------------------------------------------------------------------------------------------------

            # 그래프 분석 내용
            empty, col27, empty, col28, col29, empty = st.columns([0.05, 0.5, 0.04, 0.06, 0.3, 0.05])
        
            # 유동인구 분석
            with col27:
                st.markdown('#')
                st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
                text = '유동인구 분석 결과, <b><span style="color: red;">60대 이상</span></b>이 가장 많고 그 뒤로 <b>50대</b>와 <b>30대</b>가 많아요.'
                styled_text = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)            
                text2 = '주로 50대 이상 거주인구가 많이 방문하는 것으로 예측돼요.'
                styled_text2 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text2}</div>"
                st.markdown(styled_text2, unsafe_allow_html=True)
                text3 = '<b><span style="color: red;">월요일/수요일 저녁</span></b> 시간대를 활용하는 것이 효과적일 것으로 보여요.'
                styled_text3 = f"<div style='background-color: #f5ede3; padding: 10px; text-align: center; font-size: 18px;'>{text3}</div>"
                st.markdown(styled_text3, unsafe_allow_html=True)
                st.markdown("<div style='background-color: #f5ede3; padding: 10px;'> </div>", unsafe_allow_html=True) # 여백
            
            # 이미지 표시
            with col28:
                st.markdown('#')
                col28.image(Image.open('brochure.png'))
                col28.image(Image.open('membership.png'))

            # 프로모션 추천
            with col29:
                st.markdown('#')
                st.markdown(' ')
                text4 = "<b>5~60대</b>를 대상으로 전단지/팜플렛을 배포해<br> <b><span style='color: red;'>오프라인</span></b> 프로모션을 추천해요."
                styled_text4 = f"<div style='text-align: center; font-size: 17px;'>{text4}</div>"
                st.markdown(styled_text4, unsafe_allow_html=True)
                st.markdown('#')
                text5 = "<b>거주인구</b>를 대상으로 방문객을 늘리기 위해<br> <b><span style='color: red;'>멤버십 포인트 적립</span></b>을 추천해요."
                styled_text5 = f"<div style='text-align: center; font-size: 17px;'>{text5}</div>"
                st.markdown(styled_text5, unsafe_allow_html=True)
            
            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        hr {
                            border: none;
                            border-top: 2px solid #c4e3de;
                            margin: 1em 0;
                            width: 100%;
                        }
                        </style>""", unsafe_allow_html=True)
            
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- 네이버 쇼핑 트렌드 분석 -------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        with st.expander('네이버 쇼핑 트렌드 분석') :
            
            # 네이버 쇼핑 트렌드 분석
            st.markdown('#')
            st.markdown("<div style='background-color: #c4e3de; padding: 10px; text-align: center;'><h4>네이버 쇼핑 트렌드 분석</h4>", unsafe_allow_html=True)
            st.markdown('#')
            st.markdown('#')

            # 카테고리 선택 및 시장 정보 표시
            empty, col31, col32, col33, col34 = st.columns([0.05, 0.20, 0.25, 0.25, 0.25])

            with col31:                
                st.markdown('#')
                st.markdown('#')
                options41 = ['전체', '패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어', 
                             '출산/육아', '식품', '스포츠/레저', '생활/건강', '여가/생활편의']
                selected_option41 = st.selectbox('카테고리를 선택하세요', options41)
                st.markdown('#')
                
                if 'button' not in st.session_state:
                    st.session_state.button = True
                def click_button():
                    st.session_state.button = not st.session_state.button
                    
                # Button Style
                st.markdown("""
                            <style>
                                div.stButton > button:first-child { background-color: #f5ede3;
                                                                    color:black; }
                            </style>
                            """, unsafe_allow_html=True)
                st.button('주 고객 소비트렌드 확인', on_click = click_button, use_container_width=True)

            # 빈점포 수 그래프
            with col32:
                st.markdown("<p style='text-align: center; font-size: 22px;'><b>빈점포 수</b></p>", unsafe_allow_html=True)
                empty_store = pd.read_excel('empty_store2.xlsx', index_col=None)
                fig = px.pie(empty_store, values='empty', names='store')
                fig.update_traces(textposition='inside', textinfo='value', insidetextfont=dict(size=75))
                fig.update_layout(font=dict(size=16, color='white'))
                fig.update(layout_showlegend=False)                # 범례 표시 제거
                colors = ['#18564d']                               # 색상 설정
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=300, height=300)           # 원하는 크기로 수정
                st.plotly_chart(fig)

            # 주 고객 연령층
            with col33:
                st.markdown("<p style='text-align: center; font-size: 18px;'><b>주 고객 연령층</b></p>", unsafe_allow_html=True)
                image = Image.open('sixty.png')
                st.image(image, use_column_width=True)

            # 성별 분포 도넛 그래프
            with col34:
                st.markdown("<p style='text-align: center; font-size: 22px;'><b>성별 분포</b></p>", unsafe_allow_html=True)
                gender = pd.read_excel('gender_ratio2.xlsx', index_col=None)
                fig = px.pie(gender, values='ratio', names='gender', hole=.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(font=dict(size=14))
                fig.update(layout_showlegend=False)  # 범례 표시 제거
                colors = ['#B3E5F2', '#FECCCF']  # 색상 설정
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=300, height=300)  # 원하는 크기로 수정
                st.plotly_chart(fig)
            
# ---------------------------------------------------------------------------------------------------------------------------

            # 카테고리에 따른 순위 출력
            if st.session_state.button :
                    
                # 소비 트랜드 순위
                naver = pd.read_csv('0629_네이버쇼핑.csv', index_col=None)

                # 순위 출력 함수
                def display_ranking(category):
                    text6 = "<h4>60대 소비 트랜드</h4>"
                    styled_text6 = f"<div style='padding: 10px; text-align: center;'>{text6} [{category}]</div>"
                    st.markdown(styled_text6, unsafe_allow_html=True)            

                    empty, col35, empty, col36, empty = st.columns([0.1, 0.35, 0.1, 0.35, 0.1])

                    # 순위 1~10위
                    with col35:
                        df = naver.loc[naver['분류']=='A00', [category]].head(10)
                        df.index = [f'{i}위' for i in range(1, 11)]
                        df['비고'] = np.where(df[category].str.contains('NEW'), 'NEW 🔺', '')
                        df[category] = df[category].str.replace('NEW', '')
                        st.dataframe(df, use_container_width=True)

                    # 순위 11~20위
                    with col36:
                        dff = naver.loc[naver['분류'] == 'A00', [category]].tail(10)
                        dff.index = [f'{i}위' for i in range(11, 21)]
                        dff['비고'] = np.where(dff[category].str.contains('NEW'), 'NEW 🔺', '')
                        dff[category] = dff[category].str.replace('NEW', '')
                        st.dataframe(dff, use_container_width=True)
                    
                if selected_option41 == '전체':
                    display_ranking('전체')
                elif selected_option41 == '패션의류':
                    display_ranking('패션의류')
                elif selected_option41 == '패션잡화':
                    display_ranking('패션잡화')
                elif selected_option41 == '화장품/미용':
                    display_ranking('화장품/미용')
                elif selected_option41 == '디지털/가전':
                    display_ranking('디지털/가전')
                elif selected_option41 == '가구/인테리어':
                    display_ranking('가구/인테리어')
                elif selected_option41 == '출산/육아':
                    display_ranking('출산/육아')
                elif selected_option41 == '식품':
                    display_ranking('식품')
                elif selected_option41 == '스포츠/레저':
                    display_ranking('스포츠/레저')
                elif selected_option41 == '생활/건강':
                    display_ranking('생활/건강')
                elif selected_option41 == '여가/생활편의':
                    display_ranking('여가/생활편의')
                        
            else:
                st.write('')

            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        hr {
                            border: none;
                            border-top: 4px solid #c4e3de;
                            margin: 1em 0;
                            width: 100%;
                        }
                        </style>""", unsafe_allow_html=True)
            
            
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- 축제 정보 --------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
        with st.expander('울산광역시 축제 정보') :
            
            # 울산광역시 축제 정보
            st.markdown('#')
            st.markdown("<div style='background-color: #c4e3de; padding: 10px; text-align: center;'><h4>울산광역시 축제 정보</h4>", unsafe_allow_html=True)
            st.markdown('#')
            st.markdown('#')

            empty, col40, empty = st.columns([0.05, 0.53, 0.42])

            # 거리에 따른 슬라이드바 형성
            with col40:
                distance = st.slider("전통시장 반경 km수를 설정하세요", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
                st.markdown('###')

            empty, col41, col42, empty = st.columns([0.05, 0.57, 0.33, 0.05])
            ulsan_festival = pd.read_csv('울산광역시_문화축제데이터.csv', encoding='cp949')

            with col41:
                lat, lon = 35.5383773, 129.3113596  #울산광역시 좌표
                m = folium.Map(location=[lat, lon], zoom_start=12)

                folium.Circle(
                    location = [35.5383981, 129.303545],  # 신정평화시장 좌표
                    radius = distance * 1000,  # 1km 반경 (미터 단위)
                    color = 'red',  # 원의 테두리 색상
                    fill = True,  # 원 내부를 채울지 여부
                    fill_color = 'red',  # 원 내부 색상
                    opacity = 0.1,  # 원의 투명도
                ).add_to(m)

                bangu_market = folium.Marker(
                    location = [35.5383981, 129.303545],  # 신정평화시장 좌표
                    popup = '울산 신정평화시장',
                    tooltip = '울산 신정평화시장',
                    icon = folium.Icon('red'),
                ).add_to(m)

                #축제 정보 담기.
                for lat, lon, name, contents, date, youtube_link in zip(ulsan_festival['위도'], ulsan_festival['경도'], ulsan_festival['축제명'],
                                                                        ulsan_festival['축제내용'], ulsan_festival['축제시작일자'], ulsan_festival['유튜브']):
                    marker = folium.Marker(
                        location=[lat, lon],
                        icon=folium.Icon(icon = 'star', color='pink')
                    )
                    marker.add_to(m)

                    popup_html = f'''
                        <b>축제명:</b> {name}<br>
                        <b>축제내용:</b> {contents}<br>
                        <b>축제시작일자:</b> {date}<br>
                        <a href="{youtube_link}" target="_blank">YouTube 동영상 보기</a>
                        <br>
                        <iframe width="560" height="315" src="https://www.youtube.com/embed/{youtube_link.split("=")[1]}" frameborder="0" allowfullscreen></iframe>
                    '''
                    popup = folium.Popup(html=popup_html, max_width=800)
                    marker.add_child(popup)
                folium_static(m)

            with col42:
                location = [35.5383981, 129.303545]  # 신정평화시장 좌표
                festival_lat = ulsan_festival['위도'].tolist()
                festival_lon = ulsan_festival['경도'].tolist()

                # 각 축제 위치와 중앙전통시장의 거리 계산
                distances = [haversine((lat, lon), (location[0], location[1]), unit='km') for lat, lon in zip(festival_lat, festival_lon)]
                ulsan_festival['거리'] = [round(dist, 2) for dist in distances]
                filtered_festivals = ulsan_festival[ulsan_festival['거리'] <= distance]
                filtered_festivals.sort_values(by='거리', ascending=True, inplace=True)
                filtered_festivals.drop(['위도', '경도', '유튜브', '소재지도로명주소', '전화번호'], axis=1, inplace=True)

                # 선택한 거리 안에 있는 축제 표로 나타내기
                st.write("선택한 거리 안에 있는 축제 리스트")
                st.dataframe(filtered_festivals)

# ---------------------------------------------------------------------------------------------------------------------------

            # 전통시장과 지역 축제 Report
            empty, col43, empty = st.columns([0.1, 0.8, 0.1])

            with col43:
                st.markdown('#')
                st.markdown('#### 전통시장과 지역축제 협력 사례')
                st.markdown('###')
                st.image(Image.open('레포트.png'))

                col44, empty, col45 = st.columns([0.475, 0.02, 0.505])

                with col44:
                    text = '전통시장은 지역의 특색과 문화를 살려 축제를 개최함으로써 관광객들에게 지방색을 전달하는 중요한 역할을 할 수 있습니다.'
                    styled_text = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)            
                    text2 = '공주시는 전통시장을 중심으로 한 지역축제를 개최하여 관광객의 발걸음을 성공적으로 유치하고 지역 경제 활성화를 이루어내는 모범 사례로 평가받았습니다.'
                    styled_text2 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text2}</div>"
                    st.markdown(styled_text2, unsafe_allow_html=True)
                    text3 = '공주시의 전통시장을 활용해 다양한 프로그램과 특산물을 축제에 접목하여 방문객들의 호응을 이끌어내고 농가 소득 보전을 도모하는 성과를 얻었습니다.'
                    styled_text3 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text3}</div>"
                    st.markdown(styled_text3, unsafe_allow_html=True)
                    text4 = '전통시장과 축제의 협력은 지역 경제 활성화와 관광 산업 발전에 기여하는 효과가 있으며, 이는 다른 지자체에서도 참고할 수 있는 좋은 사례로 간주됩니다.'
                    styled_text4 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text4}</div>"
                    st.markdown(styled_text4, unsafe_allow_html=True)

                with col45:
                    text = ' <b>1. 지역 브랜드 판매 및 홍보</b><br> 유명 특산물이나 지역 고유의 제품을 전시하고 판매함으로써 지역 상인들의 수익 증대와 지역 브랜드의 홍보를 동시에 이끌어냅니다.'
                    styled_text = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)            
                    text2 = ' <b>2. 체험형 부스</b><br> 관광객들이 지역의 전통 문화나 공예를 직접 체험하고 참여할 수 있는 활동을 제공합니다.'
                    styled_text2 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text2}</div>"
                    st.markdown(styled_text2, unsafe_allow_html=True)
                    text3 = ' <b>3. 문화 예술 공간</b><br> 축제와 함께 예술 문화 행사를 연계하여 지역 예술가들의 작품 전시나 공연을 개최합니다.'
                    styled_text3 = f"<div style='background-color: #f5ede3; padding: 20px; font-size: 18px;'>{text3}</div>"
                    st.markdown(styled_text3, unsafe_allow_html=True)
                    
            # Line
            st.markdown('#')
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                        <style>
                        hr {
                            border: none;
                            border-top: 2px solid #c4e3de;
                            margin: 1em 0;
                            width: 100%;
                        }
                        </style>""", unsafe_allow_html=True)
        
        
# ==================================================================================================================================================================        
# ==================================================================================================================================================================
# ==================================================================================================================================================================


with tab4:
    def main():
        st.markdown('## [KT잘나가게]')
        st.markdown('<iframe src="https://jalnagage.kt.co.kr/home/" width="1400" height="700"></iframe>', unsafe_allow_html=True)

    if __name__ == '__main__':
        main()

# 파일실행: File > New > Terminal(anaconda prompt) - cd streamlit\market streamlit run market.py
