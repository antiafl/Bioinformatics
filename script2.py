import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def compare():
# función que compara las distintas métricas que nos interesan para quedarnos con el mejor modelo
# entrarían los valores de las métricas y saldría un valor binario -> 0 old, 1 new, que indicará cual de los modelos es mejor
# TODO darle una vuelta a como está formulado esto, porque puede ser más sencillo
	pass

#Leemos los datos de los dos ficheros
datos = pd.read_csv("~/datos.csv")
output = pd.read_csv("~/output.csv")
#partir los datos tanto las entradas como los targets de forma aleatoria pero consistente
#el interés en poner nuestra propia semilla es la de aleatorizar siempre de la misma manera los datos.
datos_train, datos_test, output_train, ouput_test = train_test_split(datos, output, test_size = 0.20, random_state = 42)

# listas con las métricas que se deseen guardar de los modelos
list_accuracy = []
list_precission = []
list_f1 = []

#almacenar gráfica roc para el mejor modelo

for x in xrange(1,10000):
	#inicializar el modelo
	svclassifier = SVC(kernel='linear')
	#entrenar el modelo
	svclassifier.fit(datos_train)

	# saca predicción para compararlas con las salidas reales, etiquetado manual
	pred_test = svclassifier.predict(datos_test)
	list_accuracy.append(metrics.accuracy(ouput_test, pred_test))
	print(list_accuracy)

# COMPARAR valores para obtener el mejor modelo

	filename = 'finalized_model.sav'
	pickle.dump(model, open(filename, 'wb'))
	 
	# load the model from disk
	loaded_model = pickle.load(open(filename, 'rb'))
	