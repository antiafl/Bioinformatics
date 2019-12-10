import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#Leemos los datos de los dos ficheros
datos = pd.read_csv("~/datos.csv")
output = pd.read_csv("~/output.csv")
#partir los datos tanto las entradas como los targets de forma aleatoria pero consistente
datos_train, datos_test, output_train, ouput_test = train_test_split(datos, output, test_size = 0.20, random_state = 42)
#el interés en poner nuestra propia semilla es la de aleatorizar siempre de la misma manera los datos.

for x in xrange(1,10000):
	#inicializar el modelo
	svclassifier = SVC(kernel='linear')
	#entrenar el modelo
	training = svclassifier.fit(datos_train)

	test = svclassifier.predict(datos_test)
	lista.append(test.metrics.accuracy)

