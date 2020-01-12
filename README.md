# Bioinformatics

Repositorio para la práctica conjunta de **Fundamentos de Bioinformática**.

Dataset: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#

## El código de la práctica consta de múltiples scripts. 

- **read_data.py**: este script se encarga de leer los datos de entrada y salida originales y procesarlos. Sobre los datos de entrada, se realiza una PCR que obtiene N atributos además de una normalización. Por otra parte, sobre los datos de salida se obtiene una matriz numérica en lugar de utilizando las etiquetas de texto de las que se dispone originalmente.

- **train_svm.py**: este script incluye todo el código necesario para la lectura de los datos ya procesados con el fin de entrenar una máquina de soporte vectorial y proporcionar unos resultados en formato .csv, junto con el modelo que obtenga el mejor valor de precisión.

- **train_kNN.py**: este script incluye todo el código necesario para la lectura de los datos ya procesados con el fin de entrenar un algoritmo kNN y proporcionar los resultados en formato .csv.

- **train_rfor.py**: este script incluye todo el código necesario para la lectura de los datos ya procesados con el fin de entrenar un random forest y proporcionar los resultados.

- **plot_results.py**: este script permite la visualización gráfica de los resultados obtenidos posteriormente a los procesos de entrenamiento, con el fin de realizar comparativas entre los modelos entrenados. 

- **plot_values.py**: este script permite la visualización gráfica de la dispersión los datos de entrada. Como es de esperar, solo es utilizable en caso de que se utilicen 2 o 3 dimensiones. 

Todos los scripts disponen de un Makefile que permite automatizar fácilmente los procesos asociados a los mismos. 
