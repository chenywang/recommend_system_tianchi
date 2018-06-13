# -*- encoding: utf8 -*-

from config.config import PROJECT_PATH
from data_process.load_data import load_data
from data_process.preprocess import preprocess
from model.implements.lr_model import LrModel
from util.evaluate import evaluate_model

item_path = PROJECT_PATH + "/data/tianchi_fresh_comp_train_item.csv"
user_action_path = PROJECT_PATH + "/data/tianchi_fresh_comp_train_user.csv"
user_action_path_mini = PROJECT_PATH + "/data/tianchi_fresh_comp_train_user_mini.csv"



def main_process(model_type):
    mini_size = 10000
    user_action_data = load_data(user_action_path).iloc[:mini_size]

    X_train, X_test, y_train, y_test = preprocess(user_action_data)

    if model_type == "lr":
        model = LrModel()
    else:
        model = LrModel()
    model.train(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main_process("lr")
