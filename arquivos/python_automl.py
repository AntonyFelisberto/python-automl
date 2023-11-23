import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import autokeras as ak
from sklearn.model_selection import train_test_split
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


def biblioteca_h2o():
    h2o.init()

    imp = pd.read_csv("Dados\\Churn_treino.csv",sep=";")
    imp = h2o.H2OFrame(imp)

    treino,teste = imp.split_frame(ratios=[0.7])

    treino["Exited"] = treino["Exited"].asfactor()
    teste["Exited"] = teste["Exited"].asfactor()

    modelo = H2OAutoML(max_runtime_secs=60)
    modelo.train(y="Exited",training_frame=treino)

    ranking = modelo.leaderboard
    ranking = ranking.as_data_frame()

    teste = pd.read_csv("Dados\\Churn_prever.csv",sep=";")
    teste = h2o.H2OFrame(teste)

    prever = modelo.leader.predict(teste)
    prever = prever.as_data_frame(prever)
    print(prever)


def biblioteca_autokeras():
    #pip3 install numpy==1.23.5  || caso de erro de np.object
    imp = pd.read_csv("Dados\\Churn_treino.csv",sep=";")
    x = imp.iloc[:,0:10]
    y = imp.iloc[:,10]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    modelo = ak.StructuredDataClassifier(max_trials=10)
    modelo.fit(x=x_train,y=y_train,epochs=100)
    modelo.evaluate(x=x_test,y=y_test)

    print(modelo.evaluate(x=x_test,y=y_test))

    prever = pd.read_csv("Dados\\Churn_prever.csv",sep=";")
    previsao = modelo.predict(prever)
    print(previsao)

def biblioteca_mlbox():
    caminho = ["Dados\\Churn_treino.csv","Dados\\Churn_teste.csv"]
    imp = Reader(sep=";")
    dados = imp.train_test_split(caminho,"Exited")
    rdrift = Drift_Thresholder()
    dados = rdrift.fit_transform(dados)

    otimizador = Optimiser()

    space = {
        "fs__strategy":{"search":"choice","space":["variance","rf_feature_importance"]},
        "est__colsample_bytree":{"search":"uniform","space":[0.3,0.7]}
    }

    modelo = otimizador.optimise(space,dados,max_evals=15)
    previsor = Predictor()
    previsor.fit_predict(modelo,dados)
