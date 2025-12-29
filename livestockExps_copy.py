# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.constraints import MaxNorm
from keras.layers import Bidirectional
from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE

import os

DATASET_PATH = '/content/drive/MyDrive/MagentaAzul 4 colunas.csv'

# Define a pasta de saída para salvar os arquivos
# Recomendação: crie uma subpasta dentro do /content/
OUTPUT_DIR = '/content/drive/My Drive/Iniciação Científica/'

# 1. Crie o diretório se ele ainda não existir
# 'exist_ok=True' evita um erro caso a pasta já exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

tscv = TimeSeriesSplit(n_splits=5)



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    print("\n\n----------Separar sequencias------------\n\n")
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)

        y.append(seq_y)
    return np.array(X), np.array(y)

# add lag variables to shallow ml methods
def add_lags_cls(data, lags):
    cl = data.iloc[:, -1] # take last column
    cl = cl.iloc[lags:] # remove first lags rows
    data = data.iloc[:, :-1] # remove last column

    df = pd.DataFrame(data)
    cols = []

    for col in data.columns:
        for lag in range(1, lags + 1):
            lag_col = f'{col}_lag_{lag}' # Creates a column name
            df[lag_col] = df[col].shift(lag) # Lags the data
            cols.append(lag_col)

    df_combined = pd.concat([df, cl], axis=1)
    df_combined.dropna(inplace=True)
    return df_combined

def smoteSequence(train_X, train_y):
    # balanceia conjunto de dados
    #oversample = SMOTE()
    #train_X_smoted, train_y_smoted = oversample.fit_resample(train_X, train_y)

    #random oversampling
    print("\n\n--------------Random OverSampler-------------\n\n")
    ros = RandomOverSampler()
    train_X_smoted, train_y_smoted = ros.fit_resample(train_X, train_y)

    # sm = BorderlineSMOTE()
    # train_X_smoted, train_y_smoted = sm.fit_resample(train_X, train_y)

    #sm = SVMSMOTE()
    #train_X_smoted, train_y_smoted = sm.fit_resample(train_X, train_y)

    # adiciona dimensão para transformar (x,) em (x,1)
    train_y_smoted = train_y_smoted[:, np.newaxis]

    # concatena atributos e classes
    newSetSmoted = np.concatenate((train_X_smoted, train_y_smoted), axis=1)
    newSet = np.concatenate((train_X, train_y), axis=1)

    # concatena horizontalmente para encontrar os exemplos artificiais, duplicando os dados reais e usando unique
    auxSet = np.concatenate((newSetSmoted, newSet), axis=0)

    SmoteSet = np.unique(auxSet, axis=0)
    dfSmote = pd.DataFrame(SmoteSet) # df com dados artificiais
    dfOriginal = pd.DataFrame(newSet) # df com dados originais

    # Pega dinamicamente o índice da última coluna (que é a coluna da classe)
    class_col_index = dfOriginal.shape[1] - 1


    # loop para inserir dados artificiais na sequencia real, insere dados artificiais em pontos de corte de classe
    # a quantidade inserida em cada ponto é constante e proporcional ao numero de cortes para aquela classe
    print("\n\n------------Inserir dados artificiais-----------\n\n")
    for cl in range(5): # 5 classes
        # seleciona exemplos da classe i
        
        grouped = dfSmote.groupby(dfSmote[class_col_index] == cl) 
        classe = grouped.get_group(True)

        # encontra alternancias de classe, entre i e j != i
        change_indices = []

        # Iterate through the column to find indices of value changes
        for i in range(len(dfOriginal) - 1):
            
            if dfOriginal[class_col_index].iloc[i] == cl and dfOriginal[class_col_index].iloc[i] != dfOriginal[class_col_index].iloc[i + 1]:
                change_indices.append(i)

        # divide dados dos Smote (da classe i) em intervalos iguais (em relação ao numero de alternancias)
        num_parts = len(change_indices)
        
        # Adicionado um 'if' para evitar divisão por zero se não houver 'change_indices'
        if num_parts > 0:
            rows_per_part = len(classe) // num_parts
            df_parts = np.array_split(classe, num_parts) # df_parts é uma lista de dataframes

            soma_parts = 0
            for j, cuts in enumerate(change_indices): # for c in change_indices:
                insert_index = cuts + 1 + soma_parts
                df_before = dfOriginal.iloc[:insert_index]
                df_after = dfOriginal.iloc[insert_index:]
                dfOriginal = pd.concat([df_before, df_parts[j], df_after], ignore_index=True)
                siz, _ = df_parts[j].shape
                soma_parts = soma_parts + siz

    return dfOriginal

#################################################################4

print("\n\n---------------Ler csv-------------\n\n")

dataset = read_csv(DATASET_PATH)#, header=0, index_col=0)
#dataset.drop(columns='azul', inplace=True) # remove labels não numericos


# ESTA LINHA FOI COMENTADA/REMOVIDA PARA GARANTIR O FLUXO DE DADOS CORRETO PARA DL!
## Comment for DL methods
# dataset = add_lags_cls(dataset,5) ## add lags to shallow methods
########


values = dataset.values

# ensure all data is float
values = values.astype('float32')

# separa em features e labels
features = values[:, :-1]
labels = values[:,-1].reshape(-1, 1)

n_train = int(len(values) * 0.7) # % dos dados para treinamento
# separa em treino e teste
train_X = features[:n_train, :]
test_X = features[n_train:, :]
train_y = labels[:n_train]
test_y = labels[n_train:]

print("\n\n---------------smoteSequence-------------\n\n")

dfO = smoteSequence(train_X,train_y) # retorna classe concatenada com treino
values = dfO.values
#
train_X = values[:, :-1]
train_y = values[:,-1].reshape(-1, 1)
############################################


# transforma dados com base nos dados de treinamento
#scaler = MinMaxScaler(feature_range=(-1, 1)) # (0,1)
scaler = StandardScaler() # (0,1)
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)


# correção de dimensão (x,) para (x,1) ## Over e Under samples
train_y=train_y.reshape(len(train_y),1)
#train_y = np.ravel(train_y)
#train_y.values.ravel()

# para metodos DL - PREPARAÇÃO SEQUENCIAL CORRETA
A = np.hstack((train_X, train_y)) 
B = np.hstack((test_X, test_y))

print("\n\n---------------Parte de lags -------------\n\n")






# cria conjunt com y a cada time step (5) - 5
train_X, train_y = split_sequences(A, 20) # train_X AGORA ESTÁ EM 3D
test_X, test_y = split_sequences(B, 20) # test_X AGORA ESTÁ EM 3D







# transformação das classes em one-hot encoding
train_y_hot = tf.keras.utils.to_categorical(train_y)
test_y_hot = tf.keras.utils.to_categorical(test_y)


def expLSTM():#train_X, train_y, test_X, test_y):
    # Function to create model, required for KerasClassifier
    def create_model(neurons):
        # create model
        model = Sequential()
        # train_X ESTÁ AGORA EM 3D: (samples, timesteps, features)
        model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(5,activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model

    # time series inside gridsearch



    # Use train_y_hot para o fit (One-Hot Encoding)
    model = KerasClassifier(model=create_model, loss='categorical_crossentropy', verbose=0)

    # define the grid search parameters
    optimizer = ['Adam', 'RMSprop']#,'SGD' ]
    neurons = [10,25,50,100]
    batch_size = [50, 100, 200]
    epochs = [25,50,100]

    param_grid = dict(model__neurons=neurons, batch_size=batch_size, epochs=epochs,optimizer=optimizer)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', refit=True, n_jobs=1, cv=tscv, error_score='raise') # accuracy

    # O fit usa train_X (3D) e train_y_hot (One-Hot)
    grid_result = grid.fit(train_X, train_y_hot)
    # summarize results

    # RESULTS
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    #predictions of best estimator
    y_pred_fm = grid.predict(test_X) # Usa test_X (3D)

    # AVALIAÇÃO: Usa as classes puras (test_y, 1D/2D) ou as preditas (y_pred_fm)
    if test_y.ndim == 2:
        classeTest = test_y.flatten() # test_y é a saída do split, que é 2D
    else:
        classeTest = test_y # Already class labels

    if y_pred_fm.ndim == 2:
        classePred = np.argmax(y_pred_fm, axis=1) # Predição é One-Hot
    else:
        classePred = y_pred_fm # Already class labels

    with open(os.path.join(OUTPUT_DIR + 'LSTM.txt'), 'w') as f:
        f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r" % (mean, stdev, param))

        f.write('\n\nPerformance metrics:')
        f.write(classification_report(classeTest, classePred))

        f.write('\nKappa Score: ')
        f.write(str(cohen_kappa_score(classeTest, classePred)))

        f.close()

    cm = confusion_matrix(classeTest, classePred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=('Standing','Walking','Eating','Drinking','Laying'))

    disp.plot()
    disp.ax_.set_title("LSTM")
    disp.figure_.savefig(OUTPUT_DIR + 'LSTM.png')


def expLSTMdense():
    # Function to create model, required for KerasClassifier

    def create_model(neurons1, neurons2, dp_rate, act, weight_constraint):
        # create model
        model = Sequential()
        # train_X ESTÁ AGORA EM 3D: (samples, timesteps, features)
        model.add(LSTM(neurons1, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(dp_rate))
        model.add(
            Dense(neurons2, activation=act, kernel_initializer='uniform', kernel_constraint=MaxNorm(weight_constraint)))
        model.add(Dense(5, activation='softmax'))
        return model

    # time series inside gridsearch




    # Use train_y_hot para o fit (One-Hot Encoding)
    model = KerasClassifier(model=create_model, loss="categorical_crossentropy", verbose=0)

    # define the grid search parameters
    optimizer = ['Adam'] 
    activation = ['relu']
    neurons1 = [10,25,50,100]
    neurons2 = [5,25,50]
    weight_constraint = [1.0]
    dropout_rate = [0.25, 0.5]
    batch_size = [50,200]
    epochs = [25,50,100]

    param_grid = dict(model__neurons1=neurons1, model__neurons2=neurons2, model__weight_constraint=weight_constraint,
                        model__dp_rate=dropout_rate, model__act=activation,
                        batch_size=batch_size, epochs=epochs, optimizer=optimizer)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', refit=True, n_jobs=1, cv=tscv)
    # O fit usa train_X (3D) e train_y_hot (One-Hot)
    grid_result = grid.fit(train_X, train_y_hot)
    # summarize results

    # RESULTS
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # predictions of best estimator
    y_pred = grid.predict(test_X) # Usa test_X (3D)

    # AVALIAÇÃO: Usa as classes puras (test_y, 1D/2D) ou as preditas (y_pred)
    if test_y.ndim == 2:
        classeTest = test_y.flatten()
    else:
        classeTest = test_y # Already class labels

    if y_pred.ndim == 2:
        classePred = np.argmax(y_pred, axis=1)
    else:
        classePred = y_pred # Already class labels

    with open(os.path.join(OUTPUT_DIR + 'LSTMdense.txt'), 'w') as f:
        f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r" % (mean, stdev, param))

        f.write('\n\nPerformance metrics:')
        f.write(classification_report(classeTest, classePred)) # Corrigido para classeTest e classePred

        f.write('\nKappa Score: ')
        f.write(str(cohen_kappa_score(classeTest, classePred))) # Corrigido para classeTest e classePred

        f.close()

    cm = confusion_matrix(classeTest, classePred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=('Standing', 'Walking', 'Eating', 'Drinking', 'Laying'))

    disp.plot()
    disp.ax_.set_title("LSTM \ Dense Layer")

    disp.figure_.savefig(OUTPUT_DIR + 'LSTMDense.png')

def expLSTMbidirecional():
    # Function to create model, required for KerasClassifier
    print("\n\n-----------Criando os neuronios------------\n\n")
    def create_model(neurons):
        # create model
        model = Sequential()

        print("\n\n----------------Dados em 3 dimensoes----------------\n\n")
        # train_X ESTÁ AGORA EM 3D: (samples, timesteps, features)
        model.add(Bidirectional(LSTM(neurons), input_shape=(train_X.shape[1], train_X.shape[2]))) 
        model.add(Dense(5, activation='softmax'))
        return model
    
    print("\n\n-----------------Colocando os lags-----------------\n\n")

    # time series inside gridsearch



    # Use train_y_hot para o fit (One-Hot Encoding)
    model = KerasClassifier(model=create_model, loss="categorical_crossentropy", verbose=0)





    print("\n\nDefine the grid search parameters\n\n")




    # define the grid search parameters
    optimizer = ['Adam', 'RMSprop']
    neurons = [25, 50, 100, 150]
    batch_size = [50,200]
    epochs = [25,50,100]

    param_grid = dict(model__neurons=neurons, batch_size=batch_size,
                        epochs=epochs, optimizer=optimizer)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, refit=True, scoring='accuracy', n_jobs=1, cv=tscv) 
    # O fit usa train_X (3D) e train_y_hot (One-Hot)


    print("\n\nOrganize the trainX\n\n")



    grid_result = grid.fit(train_X, train_y_hot)
    # summarize results

    # RESULTS
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # predictions of best estimator
    y_pred = grid.predict(test_X) # Usa test_X (3D)



    print("\n\nClasses avaliation\n\n")





    # AVALIAÇÃO: Usa as classes puras (test_y, 1D/2D) ou as preditas (y_pred)
    if test_y.ndim == 2:
        classeTest = test_y.flatten()
    else:
        classeTest = test_y # Already class labels

    if y_pred.ndim == 2:
        classePred = np.argmax(y_pred, axis=1)
    else:
        classePred = y_pred # Already class labels

    with open(os.path.join(OUTPUT_DIR + 'LSTMBidirecional.txt'), 'w') as f:
        f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r" % (mean, stdev, param))

        f.write('\n\nPerformance metrics:')
        f.write(classification_report(classeTest, classePred)) # Corrigido para classeTest e classePred

        f.write('\nKappa Score: ')
        f.write(str(cohen_kappa_score(classeTest, classePred))) # Corrigido para classeTest e classePred

        f.close()

    cm = confusion_matrix(classeTest, classePred,normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=('Standing', 'Walking', 'Eating', 'Drinking', 'Laying'))

    disp.plot()
    disp.ax_.set_title("Bidirectional LSTM")

    disp.figure_.savefig(OUTPUT_DIR + 'BidiLSTM.png')

def expLSTMbidiDense():

    def create_model(neurons1, neurons2, dp_rate, act, weight_constraint):
        # create model
        model = Sequential()
        # train_X ESTÁ AGORA EM 3D: (samples, timesteps, features)
        model.add(Bidirectional(LSTM(neurons1), input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(dp_rate))
        model.add(
            Dense(neurons2, activation=act, kernel_initializer='uniform', kernel_constraint=MaxNorm(weight_constraint)))
        model.add(Dense(5, activation='softmax'))
        return model

    # time series inside gridsearch




    # Use train_y_hot para o fit (One-Hot Encoding)
    model = KerasClassifier(model=create_model, loss="categorical_crossentropy", verbose=0)

    # define the grid search parameters
    optimizer = ['Adam']
    activation = ['relu']
    neurons1 = [25, 50, 100]
    neurons2 = [25, 50, 100]
    weight_constraint = [1.0]
    dropout_rate = [0.25, 0.5, 0.75]
    batch_size = [50,200]
    epochs = [20,50, 100]
    # param_grid = dict(batch_size=batch_size, epochs=epochs)

    param_grid = dict(model__neurons1=neurons1, model__neurons2=neurons2, model__weight_constraint=weight_constraint,
                        model__dp_rate=dropout_rate, model__act=activation,
                        batch_size=batch_size, epochs=epochs, optimizer=optimizer)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', refit=True, n_jobs=1, cv=tscv)
    # O fit usa train_X (3D) e train_y_hot (One-Hot)
    grid_result = grid.fit(train_X, train_y_hot)
    # summarize results

    # RESULTS
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # predictions of best estimator
    y_pred = grid.predict(test_X) # Usa test_X (3D)

    # AVALIAÇÃO: Usa as classes puras (test_y, 1D/2D) ou as preditas (y_pred)
    if test_y.ndim == 2:
        classeTest = test_y.flatten()
    else:
        classeTest = test_y # Already class labels

    if y_pred.ndim == 2:
        classePred = np.argmax(y_pred, axis=1)
    else:
        classePred = y_pred # Already class labels


    with open(os.path.join(OUTPUT_DIR + 'LSTMBidiDense.txt'), 'w') as f:
        f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r" % (mean, stdev, param))

        f.write('\n\nPerformance metrics:')
        f.write(classification_report(classeTest, classePred))

        f.write('\nKappa Score: ')
        f.write(str(cohen_kappa_score(classeTest, classePred)))

        f.close()

    cm = confusion_matrix(classeTest, classePred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=('Standing', 'Walking', 'Eating', 'Drinking', 'Laying'))

    disp.plot()
    disp.ax_.set_title("Bidirectional LSTM \ Dense Layer")

    disp.figure_.savefig(OUTPUT_DIR + 'BidiLSTMDense.png')

def expCNN1D():
    # --- CORREÇÃO 1 ---
    # Dimensões definidas uma única vez, usando train_y_hot
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y_hot.shape[1]

    def create_model(filter1, filter2, dp_rate, neurons1, act='relu'):
        # create model
        model = Sequential()
        model.add(
            Conv1D(filters=filter1, kernel_size=4, activation=act, input_shape=(n_timesteps, n_features)))  # ks = 3
        model.add(Conv1D(filters=filter2, kernel_size=2, activation=act))
        model.add(Dropout(dp_rate))
        model.add(MaxPooling1D(pool_size=2))  # pool_size=2
        model.add(Flatten())
        model.add(Dense(neurons1, activation=act))
        model.add(Dense(n_outputs, activation='softmax'))
        return model

    # time series inside gridsearch
    tscv = TimeSeriesSplit(n_splits=5)
    model = KerasClassifier(model=create_model, loss="categorical_crossentropy", verbose=0)

    # define the grid search parameters
    # optimization algorithm
    optimizer = ['Adam']#, 'RMSprop', 'SGD']  # , 'Adagrad', 'Adadelta', 'SGD', 'Adamax']
    activation = ['relu']#, 'linear', 'sigmoid']
    neurons1 = [25, 50, 100]#, 150, 200]
    filter1 = [50, 20]
    filter2 = [20, 10]
    dropout_rate = [0.25, 0.5]#, 0.75]
    batch_size = [50,200]
    epochs = [25,50,100]
    # param_grid = dict(batch_size=batch_size, epochs=epochs)

    param_grid = dict(model__neurons1=neurons1, model__filter1=filter1, model__filter2=filter2,
                      model__dp_rate=dropout_rate, model__act=activation,
                      batch_size=batch_size, epochs=epochs, optimizer=optimizer)

    #grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_macro', n_jobs=1, cv=tscv)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', refit=True, n_jobs=1, cv=tscv)
    
    # --- CORREÇÃO 2 ---
    # Use train_y_hot para o treinamento
    grid_result = grid.fit(train_X, train_y_hot)
    # summarize results

    # RESULTS
    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # predictions of best estimator
    y_pred = grid.predict(test_X)

    if test_y.ndim == 2:
        classeTest = np.argmax(test_y, axis=1)
    else:
        classeTest = test_y  # Already class labels

    if y_pred.ndim == 2:
        classePred = np.argmax(y_pred, axis=1)
    else:
        classePred = y_pred  # Already class labels

    #classeTest = np.argmax(test_y, axis=1)
    #classePred = np.argmax(y_pred, axis=1)

    # --- CORREÇÃO 3 ---
    # Use classeTest e classePred no relatório
    with open(os.path.join(OUTPUT_DIR + 'CNN1D.txt'), 'w') as f:
        f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r" % (mean, stdev, param))

        f.write('\n\nPerformance metrics:')
        f.write(classification_report(classeTest, classePred)) # Corrigido

        f.write('\nKappa Score: ')
        f.write(str(cohen_kappa_score(classeTest, classePred))) # Corrigido

        f.close()

    cm = confusion_matrix(classeTest, classePred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=('Standing', 'Walking', 'Eating', 'Drinking', 'Laying'))

    disp.plot()
    # disp.figure_.savefig('conf_mat.png',dpi=300)
    disp.ax_.set_title("1D CNN")

    # print('Saving confusion matrix in file {}.png'.format(DATASET_PATH.rsplit('/')[-1].split('.')[0]))
    disp.figure_.savefig(OUTPUT_DIR + 'CNN1d.png') #{}.png'.format(DATASET_PATH.rsplit('/')[-1].split('.')[0]))

def expCNNLSTM():
    # Dimensões definidas uma única vez, usando train_y_hot
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y_hot.shape[1]

    def create_model(filter1, neurons1, act='relu'):
        # create model
        model = Sequential()
        # train_X ESTÁ AGORA EM 3D: (samples, timesteps, features)
        model.add(Conv1D(filters=filter1, kernel_size=3, activation=act, padding = 'same',
                            input_shape=(n_timesteps, n_features))) # ks = 3 padding='same'
        model.add(LSTM(neurons1))
        model.add(Dense(n_outputs, activation='softmax'))
        return model





    # Dimensões definidas uma única vez, usando train_y_hot
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y_hot.shape[1]




    model = KerasClassifier(model=create_model, loss="categorical_crossentropy", verbose=0)

    # define the grid search parameters
    optimizer = ['Adam', 'RMSprop']
    activation = ['relu']
    neurons1 = [25, 50, 100]
    filter1 = [20, 50] 
    batch_size = [50,200]
    epochs = [20,50,100]

    param_grid = dict(model__neurons1=neurons1, model__filter1=filter1,
                        model__act=activation,
                        batch_size=batch_size, epochs=epochs, optimizer=optimizer)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', refit=True, n_jobs=1, cv=tscv)
    
    # Use train_y_hot para o treinamento
    grid_result = grid.fit(train_X, train_y_hot)
    # summarize results

    # RESULTS
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # predictions of best estimator
    y_pred = grid.predict(test_X)#grid_result.best_estimator_.predict(test_X)

    if test_y.ndim == 2:
        classeTest = test_y.flatten()
    else:
        classeTest = test_y # Already class labels

    if y_pred.ndim == 2:
        classePred = np.argmax(y_pred, axis=1)
    else:
        classePred = y_pred # Already class labels

    # Use classeTest e classePred no relatório
    with open(os.path.join(OUTPUT_DIR +'CNNLSTM.txt'), 'w') as f:
        f.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r" % (mean, stdev, param))

        f.write('\n\nPerformance metrics:')
        f.write(classification_report(classeTest, classePred)) # Corrigido

        f.write('\nKappa Score: ')
        f.write(str(cohen_kappa_score(classeTest, classePred))) # Corrigido

        f.close()

    cm = confusion_matrix(classeTest, classePred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=('Standing', 'Walking', 'Eating', 'Drinking', 'Laying'))

    disp.plot()
    disp.ax_.set_title("CNN \ LSTM")

    disp.figure_.savefig(OUTPUT_DIR + 'CNNLSTM.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('--------------------------ULTIMO UPDATE 22/12 7H45------------------------------')
    # As funções de DL (LSTM, Bidirecional, CNN) agora devem funcionar corretamente
    # porque os dados (train_X, test_X) estão no formato 3D esperado.
    #expLSTMdense()
    #expLSTM()
    expCNN1D()
    #expLSTMbidirecional()
    #expLSTMbidiDense()
    #expCNNLSTM()
