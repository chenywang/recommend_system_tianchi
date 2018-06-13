# -*- encoding: utf8 -*-
from collections import Counter

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_test, y_test):
    print 'start evaluate'
    y_probability = model.predict(X_test)
    element_list = list()
    for line in zip(y_probability, y_test):
        element = {
            "1 prob": line[0][0],
            "2 prob": line[0][1],
            "3 prob": line[0][2],
            "4 prob": line[0][3],
            "actual": line[1]
        }
        element_list.append(element)

    result = pd.DataFrame(element_list)
    print "predict result"
    print result[result["actual"] == 1].head(n=10)
    print result[result["actual"] == 2].head(n=10)
    print result[result["actual"] == 3].head(n=10)
    print result[result["actual"] == 4].head(n=10)

    print '======================='


def draw_roc_curve(y_probability, y_test):
    # 画出roc曲线
    fpr, tpr, threshold = roc_curve(y_test, y_probability)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()