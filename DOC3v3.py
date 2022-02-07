import os
from numpy import random
from pandas.core.reshape.concat import concat
from scipy.sparse.construct import rand
from sklearn.utils import shuffle
from tensorflow.python.keras.engine import sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sys
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

def create_seen(data):
    pass

def create_unseen(data):
    pass

def get_doc_model(input_shape, output_dim):
    model = Sequential()
    model.add(Dense(100, input_shape = input_shape,activation='relu'))
    model.add(Dense(100))
    model.add(Dense(output_dim, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_alt_doc_model(input_shape,output_dim):
    model = Sequential()
    model.add(Dense(100, input_shape = input_shape,activation='relu'))
    model.add(Dense(100))
    model.add(Dense(output_dim, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_nn_model(input_shape,output_dim):
    model = Sequential()    
    model.add(Dense(100, input_shape = input_shape,activation='relu'))
    model.add(Dense(100))
    model.add(Dense(output_dim, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_adaptive_thresholds(train_pred_probs,y_train,n_train_classes,adapt):
    """
    This is a DocString test
    """
    
    thresholds = []
    train_pred_probs = pd.Series(np.max(train_pred_probs, axis=1), index=y_train.index)
    for i in range(n_train_classes):
        class_probs = train_pred_probs[y_train == i]
        class_probs_ext = np.concatenate([class_probs, (2 - class_probs)])
        std = np.std(class_probs_ext)
        thresholds.append(max(0.5,1-(3*std)))
    return thresholds

def to_labels(preds,thresholds,new_class_label=3):
    _labels = np.argmax(preds,axis=1)
    max_vals = np.max(preds,axis=1)

    labels = []
    for label, max_val in zip(_labels, max_vals):
        threshold = thresholds[label]
        if max_val >= threshold:
            labels.append(label)
        else:
            labels.append(new_class_label)
    return labels

def loss_and_acc_plot(X,Y,model):
    history = model.fit(X,Y, validation_split=0.33, epochs=15, verbose=0)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    pass

#data loading
n_train_classes = 3
seen = list(range(3))
unseen = list(range(3,4))

benAndBot43 = pd.read_csv('D:/BT Placement\Implementations/DOC3/DOC3.3/benAndBot.csv')
botnet42 = pd.read_csv('D:/BT Placement\Implementations/DOC3/DOC3.3/botnet42.csv')
botnet47 = pd.read_csv('D:/BT Placement\Implementations/DOC3/DOC3.3/botnet47.csv')
botnet45 = pd.read_csv('D:/BT Placement\Implementations/DOC3/DOC3.3/botnet45.csv')

botnetData, benignData = [x for _, x in benAndBot43.groupby(benAndBot43['Label'] == "Benign")]

benAndBot43.loc[(benAndBot43.Label == 'Benign'), 'Label'] = 0
benAndBot43.loc[(benAndBot43.Label == 'Botnet43'), 'Label'] = 1
botnet47.loc[(botnet47.Label == 'Botnet47'), 'Label'] = 2
botnet42.loc[(botnet42.Label == 'Botnet42'), 'Label'] = 3

ben = benAndBot43.loc[benAndBot43.Label == 0]
bot43 = benAndBot43.loc[benAndBot43.Label == 1]
benAndBot43 = pd.concat([ben.sample(20000, random_state=42),bot43])

colNames = list(benAndBot43.columns)

benAndBot = concat([benAndBot43,botnet47]).reset_index(drop=True)

print(benAndBot.head())
print(benAndBot['Label'].value_counts())

scaler = MinMaxScaler()
scaled = scaler.fit_transform(benAndBot.loc[:,benAndBot.columns != 'Label'])
unseen_scaled = scaler.transform(botnet42.loc[:,botnet42.columns != 'Label'])

normalDF = pd.DataFrame(scaled,columns=colNames[:15]).reset_index(drop=True)
normalDF.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
print("Normalised Training Data: ")
print(normalDF.head())
unseen_normalDF = pd.DataFrame(unseen_scaled,columns=colNames[:15])
unseen_normalDF.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

normalDFx = concat([normalDF,benAndBot.loc[:,benAndBot.columns == 'Label']], axis = 1)
unseen_normalDFx = concat([unseen_normalDF,botnet42.loc[:,botnet42.columns == 'Label']], axis = 1)

shuffledDF = normalDFx.sample(frac=1,random_state=4)
unseen_shuffledDF = unseen_normalDFx.sample(frac=1,random_state=4)

botnetDF = shuffledDF.loc[(shuffledDF['Label'] == 1 ) | (shuffledDF['Label'] == 2)]
unseen_botnetDf = unseen_shuffledDF.loc[unseen_shuffledDF['Label'] == 3 ].sample(n=3500,random_state=42)

balancedDF = shuffledDF

X = balancedDF.loc[:,balancedDF.columns != 'Label']
unseen_x = unseen_botnetDf.loc[:, unseen_botnetDf.columns != 'Label']
y = balancedDF['Label']
unseen_y = unseen_botnetDf['Label']

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=1)

X_test = np.concatenate([X_test,unseen_x], axis=0)
test_y_gt = np.concatenate([Y_test, unseen_y], axis = 0)
print(X_test.shape,test_y_gt.shape)
testYGT = test_y_gt.tolist()

print("Training Data Distributions: ")
print(Y_train.value_counts())
cate_seen_train_y = to_categorical(Y_train,num_classes=n_train_classes)

tf.keras.backend.clear_session()

input_shape = (15,)

#model training and evaluation loop
for count in range(1):

    count +=1
    doc_model = get_doc_model(input_shape,n_train_classes)
    doc_model.fit(X_train,cate_seen_train_y,epochs=15)
    print(doc_model.summary())
    

    alt_doc_model = get_alt_doc_model(input_shape,n_train_classes)
    alt_doc_model.fit(X_train,cate_seen_train_y,epochs=15)
    print(alt_doc_model.summary())

    nn_model = get_nn_model(input_shape,n_train_classes)
    nn_model.fit(X_train,cate_seen_train_y,epochs=15)
    print(nn_model.summary())

    train_X_pred = doc_model.predict(X_train)
    test_X_pred = doc_model.predict(X_test)
    print(test_X_pred)

    maxpreds = np.amax(train_X_pred,1)

    from statistics import mean
    maxthresh = mean(maxpreds)
    print("maxxxxxy", maxthresh)

    sns.displot(train_X_pred,kde=True)
    plt.show()

    thresholds = get_adaptive_thresholds(train_X_pred,Y_train,n_train_classes,0.5)
    print("Thresholds: {}".format(thresholds))

    test_y_pred = to_labels(test_X_pred,thresholds)
    train_y_pred = to_labels(train_X_pred,thresholds)
    print("Pred Labels: ")
    print(pd.Series(train_y_pred).value_counts())
    print(pd.Series(test_y_pred).value_counts())


    alt_train_X_pred = alt_doc_model.predict(X_train)
    alt_test_X_pred = alt_doc_model.predict(X_test)

    alt_maxpreds = np.amax(train_X_pred,1)

    sns.distplot(alt_maxpreds,hist=True,kde=True,bins=int(180/5))
    plt.show()

    alt_thresholds = get_adaptive_thresholds(alt_train_X_pred,Y_train,n_train_classes,0.5)
    print("Alt Thresholds: {}".format(alt_thresholds))

    alt_test_y_pred = to_labels(alt_test_X_pred,alt_thresholds)
    alt_train_y_pred = to_labels(alt_train_X_pred,alt_thresholds)
    print("Alt Pred Labels: ")
    print(pd.Series(alt_train_y_pred).value_counts())
    print(pd.Series(alt_test_y_pred).value_counts())


    #######no new thresholds

    X_train_pred = nn_model.predict(X_train)
    X_test_pred = nn_model.predict(X_test)
    print(X_test_pred)

    sns.displot(X_train_pred[:,1])
    plt.show()

    nn_thresholds = get_adaptive_thresholds(X_train_pred,Y_train,n_train_classes,0.5)
    print("NN Thresholds: {}".format(nn_thresholds))


    y_test_pred = to_labels(X_test_pred,nn_thresholds)
    y_train_pred = to_labels(X_train_pred,nn_thresholds)
    print("NN Pred Labels: ")
    print(pd.Series(y_test_pred).value_counts())

    cm = confusion_matrix(testYGT,test_y_pred)
    cmCat = confusion_matrix(testYGT,alt_test_y_pred)
    cmNN = confusion_matrix(testYGT,y_test_pred)

    normalCM = cm / cm.astype(np.float64).sum(axis=1)
    print(cm)

    from sklearn.metrics import classification_report

    # print("Doc train set results: ")
    # print(classification_report(Y_train.tolist(), train_y_pred))
    print("Doc test set results: ")
    print(classification_report(testYGT,test_y_pred))
    print("Cat DOC test set results: ")
    print(classification_report(testYGT,alt_test_y_pred))
    print("\nNN test set results: ")
    print(classification_report(testYGT,y_test_pred))


    def precision(label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
        
    def recall(label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def precision_macro_average(confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            sum_of_precisions += precision(label, confusion_matrix)
        return sum_of_precisions / rows

    def recall_macro_average(confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += recall(label, confusion_matrix)
        return sum_of_recalls / columns

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 

    from sklearn.metrics import f1_score

    DOCf1 = f1_score(testYGT,test_y_pred,average='macro')
    alt_DOCf1 = f1_score(testYGT,alt_test_y_pred,average='macro')
    NNf1 = f1_score(testYGT,y_test_pred,average='macro') 
    print("label precision recall")
    for label in range(4):
        print(f"{label:5d} {precision(label, cmNN):9.3f} {recall(label, cmNN):6.3f}")


    print("precision total:", precision_macro_average(cmNN))

    print("recall total:", recall_macro_average(cmNN))

    print(accuracy(cmNN))

    from sklearn.metrics import cohen_kappa_score

    cohenDOC = cohen_kappa_score(testYGT,test_y_pred)
    alt_cohenDOC = cohen_kappa_score(testYGT,alt_test_y_pred)
    cohenNN = cohen_kappa_score(testYGT,y_test_pred)

    models_scores_table = pd.DataFrame({'BinDOC':[precision_macro_average(cm),
                                                    recall_macro_average(cm),
                                                    accuracy(cm),
                                                    DOCf1,
                                                    cohenDOC], 

                                        'CatDOC':[precision_macro_average(cmCat),
                                                    recall_macro_average(cmCat),
                                                    accuracy(cmCat),
                                                    alt_DOCf1,
                                                    alt_cohenDOC], 

                                        'SoftMax':[precision_macro_average(cmNN),
                                                    recall_macro_average(cmNN),
                                                    accuracy(cmNN),
                                                    NNf1,
                                                    cohenNN],}, 
                                        index=['Precision','Recall','Accuracy','F1-Score','Cohen Kappa'])

    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    print(models_scores_table)

    # filename = open('count_test_11_alpha1.txt', 'a')
    # filename.write(str(models_scores_table.iloc[3,0:3].values))
    # filename.write("\n")
    # filename.close()

from sklearn import manifold
from sklearn.manifold import TSNE

def embedding_plot(X, title,colors):

    num_classes = 4
    palette = np.array(sns.color_palette("hls", num_classes))


    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=palette[colors.astype(int)])
    shown_images = np.array([[1., 1.]])

    txts = []
    for i in range(X.shape[0]):
        if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
        shown_images = np.r_[shown_images, [X[i]]]
        xtext, ytext = np.median(X[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=16)
        txts.append(txt)
        #ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(range(3), cmap=plt.cm.gray_r), X[i]))
    plt.xticks([]), plt.yticks([])
    plt.title(title)

# X_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(X_test)
# embedding_plot(X_tsne,"t-SNE",test_y_gt)
# plt.show()

#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# tY = np.array(testYGT)

# # first reduce dimensionality before feeding to t-sne
# # pca = PCA(n_components=10)
# # X_pca = pca.fit_transform(X_test) 
# # randomly sample data to run quickly
# rows = np.arange(20000)
# np.random.shuffle(rows)
# n_select = 10000 
# # reduce dimensionality with t-sne
# tsne = TSNE(n_components=2, verbose=2, perplexity=25, n_iter=1000, learning_rate=200)
# tsne_results = tsne.fit_transform(X_test[rows[:n_select],:])
# # visualize
# df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
# df_tsne['label'] = tY[rows[:n_select]]
# sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='label', fit_reg=False)
# plt.show()


