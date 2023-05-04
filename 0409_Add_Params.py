import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_folium import st_folium
import pickle, folium, gzip
import call_model

im = Image.open("AI_Lab_logo.jpg")

st.set_page_config(
    page_title="계량기 동파예측(Meter Freeze preiction)",
    page_icon='📈',
    layout="wide",
)

col1, col2 = st.columns([1, 8])
col1.image(im, width=100)
col2.write("# 지방상수도 함내 계량기 동파 예측")
col2.write("### (Predicting meter freezes in municipal water boxes)")
col2.write("")

st.write("##### 📌 예측을 위한 데이터를 입력하세요(Select Data)")
with st.form('My Form'):
    st.markdown("##### ⁎ 파라미터 선택(Selecting Parameters)")
    col1, col2, col3, col4 = st.columns(4)
    n_estimators = col1.slider("트리 개수(n_estimators)", min_value=0, max_value=500, step=1, value=100)
    col1.text_area("", value="• Number of trees\n"
                             "• Default: 100\n"
                             "• 숫자가 커지면 성능이 좋아지나, 시간이 오래걸릴 수 있음",
                   height=250)

    max_depth = col2.slider("트리 최대깊이(max_depth)",min_value=0, max_value=20, step=1, value=0)
    col2.text_area("", value="• maximun depth of the tree\n"
                             "• Default: None\n"
                             "• 0선택시 None이 입력됩니다\n"
                             "• 숫자가 커지면 성능이 좋아지나, 과적합의 위험이 있음",
                   height=250)

    min_samples_split = col3.slider("분기 조건(min_samples_split", min_value=2, max_value=20, step=1, value=2)
    col3.text_area("", value="• Minimum number of samples required to split an internal node\n"
                             "• 노드를 분할하기 위한 최소 샘플 수\n"
                             "• Default: 2\n"
                             "• 숫자가 커지면 모델이 일반화 됨(과적합 방지)\n "
                             "• 숫자가 작으면 정확도가 높아짐",
                   height=250)

    min_samples_leaf = col4.slider("리프 노드 크기(min_samples_leaf", min_value=1, max_value=20, step=1, value=1)
    # 보통 min_samples_leaf은 min_samples_split보다 작아야함
    if min_samples_leaf > min_samples_split:
        min_samples_leaf = min_samples_split - 1
    col4.text_area("", value="• Minimum number of samples required to be at a leaf node\n"
                             "• 리프노드가 될 수 있는 최소한의 샘플 데이터 수\n"
                             "• Default: 1\n "
                             "• 일반적으로 [분기조건]보다 작아야함"
                             "• [분기조건]보다 큰 값이 선택될 경우, 자동으로 -1 \n"
                             "• 숫자가 커지면 모델이 단순해짐\n "
                             "• 숫자가 작으면 모델이 복잡해짐",
                   height=250)

    st.markdown("")
    st.markdown("")

    # 원래 있던 입력변수들
    st.markdown("##### ⁎ 독립변수 선택(Selecting independent variables)")
    col1, col2, col3, col4, col5 = st.columns(5)
    latitude = col1.text_input("위도(Latitude, 34~39)", "37.41465")
    latitude = float(latitude)
    longitude = col2.text_input("경도(Longitude, 126~130)", "127.2876")
    longitude = float(longitude)
    elevation = col3.text_input("고도(Altitude)", "62.91475")
    elevation = float(elevation)
    warmer = col4.selectbox("보온재(Insulation)", ['없음', '많음'])
    nwarmer = 1 if warmer == '많음' else 0
    shade = col5.selectbox("음/양지(Shade/Sunny)", ['양지', '음지'])
    nshade = 1 if shade == '양지' else 0

    col1, col2, col3, col4 = st.columns(4)
    temperature = col1.slider("기온(Temperature)", -50.0, 50.0, -7.1)
    humidity = col2.slider("습도(Humidity)", 0.0, 100.0, 71.1)
    wind_speed = col3.slider("풍속(Wind Speed)", 0.0, 50.0, 1.9)
    rainfall = col4.slider("강수량(Rainfall)", 0.0, 100.0, 0.0)

    col1, col2 = st.columns([8, 1])
    submit = col2.form_submit_button('🎓 동파예측(Prediction)')


    if submit:
        #params 추가됨
        # 2304_Add_Parameters.py 파일 불러서 파라미터 넘겨서 학습시키고
        call_model.call_model(n_estimators, max_depth, min_samples_split, min_samples_leaf)
        # 밑에 코드 돌리기
        # 비효율의 극치인걸....

        non_scale = np.array([latitude, longitude, elevation, temperature, rainfall, wind_speed, humidity]).reshape(1,-1)
        with open('scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)

        x = scaler.transform(non_scale)
        x = np.append(x, nwarmer)
        x = np.append(x, nshade).reshape(1, -1)
        full_data = pd.DataFrame(x, columns=["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "보온재여부", "양지/음지"])
        x_data_n_sel = full_data[["고도", "위도", "경도", "보온재여부"]]
        with open('kmeans_model.pkl', 'rb') as f:
            Kmeans_model = pickle.load(f)

        ncluster = Kmeans_model.predict(x_data_n_sel)[0]  # 값이 배열
        # print(ncluster)

        # k 값에 따른 모델을 찾아서 온도 예측
        # models = ["RFRegress_Model_Zip0.pkl", "RFRegress_Model_Zip1.pkl", "RFRegress_Model_Zip2.pkl",
        #           "RFRegress_Model_Zip3.pkl"]

        models = ["RFRegress_Model0.pkl", "RFRegress_Model1.pkl", "RFRegress_Model2.pkl",
                  "RFRegress_Model3.pkl"]


        select_model = models[ncluster]
        with open(select_model, 'rb') as m:
            rf_model = pickle.load(m)

        x_data = full_data[["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "양지/음지"]]

        y_pred = rf_model.predict(x_data)[0]

        loc_map = folium.Map(location=[latitude, longitude],
                             zoom_start=12)

        tooltip_prn = "예상 온도는 {0:0.3f}입니다".format(y_pred)
        folium.Marker([latitude, longitude], icon=folium.Icon(color='red'),
                      popup=tooltip_prn, tooltip=tooltip_prn).add_to(loc_map)
        st_folium(loc_map, width=1400)
        col1.write("#### ▶ 예상 온도는 {0:0.3f}℃입니다 ".format(y_pred))
        col1.write("#### ▶ The expected temperature is {0:0.3f}℃ ".format(y_pred))

