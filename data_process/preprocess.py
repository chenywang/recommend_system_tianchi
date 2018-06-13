# -*- encoding: utf8 -*-
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
RANDOM_STATE = 0

def preprocess(user_action_data):
    print "start preprocess.."
    # target feature
    behavior_type = user_action_data['behavior_type']
    print "unique behavior", len(set(behavior_type))
    # behavior_type = behavior_type / 4
    user_action_data = user_action_data.drop(['behavior_type', 'user_geohash', 'time'], axis=1)
    # one hot化
    string_columns = ['user_id', 'item_id', 'item_category']
    for column in string_columns:
        user_action_data[column] = user_action_data[column].astype('string')
    user_action_data = user_action_data.to_dict(orient='records')
    vec = DictVectorizer()
    user_action_data = vec.fit_transform(user_action_data).toarray()

    # 将dict的list转成Dataframe
    user_action_data = pd.DataFrame(user_action_data, columns=vec.get_feature_names())

    # print "user_action_data3"
    # print user_action_data.head(n=2)

    # 划分训练集，测试集
    X_train, X_test, y_train, y_test = train_test_split(user_action_data, behavior_type, test_size=0.3,
                                                        random_state=RANDOM_STATE)
    print "======================="
    return X_train, X_test, y_train, y_test