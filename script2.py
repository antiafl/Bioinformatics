import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#Leemos los datos de los dos ficheros
datos = pd.read_csv("~/datos.csv")
output = pd.read_csv("~/output.csv")
datos_train, datos_test, output_train, ouput_test = train_test_split(datos, output, test_size = 0.20)
# Crear 5 modelos para cada uno de los targets
# Cada modelo usa los mismos datos, es de tipo "es X, es no X"


OneVsRestClassifier(LinearSVC(random_state=0)).fit(datos_train, output_train)

svclassifier = SVC(kernel='linear')
svclassifier.fit(datos_train)