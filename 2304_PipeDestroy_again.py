#필요 라이브러리 import
import pandas as pd
from IPython.display import display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

# 데이터 불러오기
df = pd.read_csv("C:/Users/kwater/PyCharm_test/pipe_codes/freeze_data_dropna.csv", encoding="cp949")
# display(df.head())
'''
   Unnamed: 0        검침일시        무선온도센서  계량기함온도  ...   기온  강수량   풍속    습도
0           0  2021112700  WM7A22W00126     6.1  ... -3.1  0.0  0.0  85.0
1           1  2021112700  WM7A22W00127     8.4  ... -3.1  0.0  0.0  85.0
2           2  2021112700  WM7A22W00128     5.1  ... -3.1  0.0  0.0  85.0
3           3  2021112701  WM7A22W00126     6.2  ... -3.3  0.0  0.1  88.0
4           4  2021112701  WM7A22W00127     8.4  ... -3.3  0.0  0.1  88.0
'''
df = df.loc[df["계량기함온도"] <= 20]
coords = df.copy()

# 이상치 제거
df = df.loc[df['계량기함온도'] <= 100]

# 하이퍼 파라메타
# 변수 : 위도, 경도, 고도, 기온, 강수량, 풍속, 습도, 양/음지
k = 4
x_lst = ['위도', '경도', '고도', '기온', '강수량', '풍속', '습도', '양지/음지', '보온재여부']
scales = ["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "양지/음지"]

## 표준화
# MinMax보다 Standard가 더 성능 좋았음
scaler = StandardScaler()
df[scales] = scaler.fit_transform(df[scales])

# scaler 저장, 파일명은 scaler.pkl -> 필요시 갖다가 사용하면 됨
with open("scaler.pkl", "wb") as s:
    pickle.dump(scaler, s)

# df 저장
df.to_csv("pipe_data.csv", encoding='cp949')

# X
x_lists = [["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "양지/음지", 'cluster', '계량기함온도']]

### 자문 이후에 PCA를 없앤다고 했지만, 일단 만들어놓고 그 다음에 수정해보자
# 비지도 분류
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df[["고도", "위도",  "경도", "보온재여부"]])

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

Kmeans_model = KMeans(n_clusters=k, random_state=10, n_init=10)
Kmeans_model.fit(principalDf)

# 비지도 분류 모델 저장
with open("kmeans_model.pkl", "wb") as m:
     pickle.dump(Kmeans_model, m)

df['cluster'] = Kmeans_model.predict(principalDf)
coords['cluster'] = Kmeans_model.predict(principalDf)

coords = list(coords.groupby(["위도", "경도", "cluster"]).groups.keys())

# 0404 여기까지 살펴봄
# 다음에는 이거 streamlit으로 옮겨가는거보고
# 파라미터 어떻게 어떤거 넣을지 보면 됨
###모델돌리기 -> 여기도 나중에 좀 더 간단하게 만들 수 있을 듯.
model = RandomForestRegressor(n_estimators=100, random_state=42)

seq_length = 1
for x_list in x_lists:
    print("=" * 70)

    for cluster in range(k):
        tmp = df[df['cluster'] == cluster]
        x = tmp[x_list].drop(['cluster', '계량기함온도'], axis=1)

        count = 0
        x = x.values.tolist()
        li = list()
        li2 = list()
        for i in range(len(x)):
            if i < seq_length - 1:
                continue
            for j in range(0, seq_length):
                li2.extend(x[i - j])
            li.append(li2)
            li2 = list()

        x = li
        y = tmp['계량기함온도']
        y = y[seq_length - 1:]  # 첫번째 행 빼고

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        fname = "RFRegress_Model" + str(cluster) + ".pkl"
        with open(fname, "wb") as m:
            pickle.dump(model, m)

        print("-" * 70)
        print("cluster : ", cluster)
        print("데이터수 : ", tmp.shape[0])
        print("mae : ", mean_absolute_error(y_test, y_pred))
        print("mse : ", mean_squared_error(y_test, y_pred))
        print("rmse : ", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("r2 : ", r2_score(y_test, y_pred))

