import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def call_model(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    print('들어왔어요')

    if max_depth==0:
        max_depth=None

    df = pd.read_csv("data_fin.csv", encoding="cp949")

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)

    seq_length = 1
    k=4
    x_lists = [["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "양지/음지", 'cluster', '계량기함온도']]


    for x_list in x_lists:
        # print('x_list : {}, count: {}'.format(x_list, i))
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
            # print('x:', x)
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

