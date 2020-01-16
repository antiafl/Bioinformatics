# Fundamentals of Bioinformatics

Repositorio para la práctica conjunta de **Fundamentos de Bioinformática**.

Dataset: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#

## El código de la práctica consta de múltiples scripts

- **read_data.py**: este script se encarga de leer los datos de entrada y salida originales y procesarlos. Sobre los datos de entrada, se realiza una PCR que obtiene N atributos además de una normalización. Por otra parte, sobre los datos de salida se obtiene una matriz numérica en lugar de utilizando las etiquetas de texto de las que se dispone originalmente.

- **train_svm.py**: este script incluye todo el código necesario para la lectura de los datos ya procesados con el fin de entrenar una máquina de soporte vectorial y proporcionar unos resultados en formato .csv, junto con el modelo que obtenga el mejor valor de precisión.

- **train_kNN.py**: este script incluye todo el código necesario para la lectura de los datos ya procesados con el fin de entrenar un algoritmo kNN y proporcionar los resultados en formato .csv.

- **train_rfor.py**: este script incluye todo el código necesario para la lectura de los datos ya procesados con el fin de entrenar un random forest y proporcionar los resultados.

- **plot_results.py**: este script permite la visualización gráfica de los resultados obtenidos posteriormente a los procesos de entrenamiento, con el fin de realizar comparativas entre los modelos entrenados. 

- **plot_values.py**: este script permite la visualización gráfica de la dispersión los datos de entrada. Como es de esperar, solo es utilizable en caso de que se utilicen 2 o 3 dimensiones. 

## También se incluyen otros archivos

- **inputs_PCA_N.csv**: ficheros de entrada a los modelos de machine learning resultantes de la aplicación de una normalización y una PCA de N atributos.

- **Directorio results**: contiene los resultados para cada uno de los experimentos realizados. Los ficheros que aquí se encuentran se denotan como $results\_<tipo\_de\_modelo>\_<fichero\_de\_entrada>.csv$ donde $<tipo\_de\_modelo>$ puede ser "kNN", "svm" o "rfor" (random forest) y $<fichero\_de\_entrada>$ el archivo .csv producto de aplicar la PCA y la normalización sobre los datos originales.

- **Directorio models**: contiene los modelos con la mejor precisión obtenidos en un experimento. Se denotan como $<fichero\_de\_entrada>\_<tipo\_de\_modelo>.pth$. 

- **Directorio images**: directorio donde se encuentran las imágenes utilizadas para el Jupyter Notebook de la práctica.

- **Directorios RFOR, SVM y kNN**: contienen algunos resultados de ciertos experimentos que se desearon almacenar para presentar más tarde la práctica. La notación de estos ficheros es la misma que para los que se encuentran en el fichero *results*.

Todos los scripts disponen de un Makefile que permite automatizar fácilmente los procesos asociados a los mismos. 

## Uso de los scripts

### Scripts de entrenamiento de los modelos

    python3 [train_svm.py|train_kNN.py|train_rfor.py] --inputs_file <fichero_de_entrada> --targets_file <fichero_de_salidas_esperadas> --noftrains N --test_size <tamaño_conjunto_test> --results_path <directorio_almacenamiento_resultados> --models_path <directorio_almacenamiento_modelo_mejor_precision>

También es posible realizar la ejecución con el Makefile haciendo uso de la opción "make [train_svm | train_kNN | train_rfor]".

### Script para la normalización y aplicación de la PCA sobre los datos originales.

    python3 read_data.py --inputs_file <fichero_datos_originales> --targets_file <fichero_salidas_deseadas_original> --pca_n N --input_storing <fichero_datos_normalizados_y_PCA> --targets_storing <fichero_salidas_esperadas_procesado>
    
También es posible realizar la ejecución con el Makefile haciendo uso de la opción "make read_data".

### Script para mostrar los patrones obtenidos con una PCA de dos atributos en un espacio bidimensional con la correspondiente separación realizada con un clasificador. También se puede ver la dispersión de los valores con una PCA de 3 atributos (pero sin la separación de los clasificadores)

    python3 plot_values.py --inputs_file <fichero_de_entrada> --targets_file <fichero_de_salidas_esperadas> --model <ruta_del_modelo_utilizado> --dim D

**Es importante tener en cuenta los siguientes detalles**:

- D = 2 (fichero de entrada con PCA de 2 atributos y modelo entrenado con 2 atributos).

- D = 3 (fichero de entrada con PCA de 3 atributos y modelo entrenado con 3 atributos).

Por simplicidad, el Makefile proporciona la opción "make plot_values".

### Script para mostrar los resultados obtenidos de un experimento realizado de forma gráfica.

    python3 plot_results.py --title <titulo_de_la_grafica> --input_dir <fichero_resultados_a_comparar> --opt N
    
**El valor de N especifica dos opciones:**

- N = 0 (debe utilizarse cuando se desea visualizar los resultados de un modelo concreto).

- N = 1 (debe utilizarse cuando los resultados pertenecen a modelos diferentes).

Se puede utilizar, del mismo modo, la opción del Makefile "make plot_results".

## Detalles adicionales

Debe tenerse en cuenta que los ficheros con la información original del repositorio UCI Machine Learning no se incluyen en este proyecto, puesto que los más de 200 MB que ocupa aumentarían considerablemente el tiempo de descarga de los archivos del repositorio GitHub utilizado para albergar este proyecto.