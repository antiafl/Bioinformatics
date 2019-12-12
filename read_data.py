import argparse
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# En esta función se obtiene la matriz correspondiente al fichero .csv que
# se le pase como parámetro.
def get_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            matrix.append(row)

    return np.array(matrix)

# Esta función es utilizada para almacenar datos en un fichero .csv.
def store_data(matrix, path):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in range(0, np.shape(matrix)[0]):
            writer.writerow(matrix[row])

# Convertir la cadena de texto que representa la clase a la que pertenece un
# patrón en su vector de números correspondiente.
def get_numerical_output(tag_name):
    if (tag_name=='PRAD'):
        tag_number = [1, 0, 0, 0, 0]
    elif (tag_name=='LUAD'):
        tag_number = [0, 1, 0, 0, 0]
    elif (tag_name=='COAD'):
        tag_number = [0, 0, 1, 0, 0]
    elif (tag_name=='KIRC'):
        tag_number = [0, 0, 0, 1, 0]
    elif (tag_name=='BRCA'):
        tag_number = [0, 0, 0, 0, 1]
    else:
        tag_number = None

    return tag_number

def get_model_target(matrix):
    nrows = np.shape(matrix)[0]

    target = []
    for row in range(0, nrows):
        target.append(get_numerical_output(matrix[row]))

    return np.array(target)

def executePCA(input_matrix, args):
    n_components = args.pca_n
    pca = PCA(n_components)
    principal_components = pca.fit_transform(input_matrix)
    principal_components = StandardScaler().fit_transform(principal_components)

    return principal_components

def main(args):
    input = get_data(args.inputs_file)
    output = get_data(args.targets_file)

    # Se obtiene el número de instancias de la base de datos.
    nofinstances = np.shape(input)[0]-1
    # Se obtiene el número de atributos de la base de datos.
    nofattributes = np.shape(input)[1]-1

    input = input[1:nofinstances+1, 1:nofattributes+1].astype(float)
    output = get_model_target(output[1:nofinstances+1, 1:])
    input = StandardScaler().fit_transform(input)
    principal_components = executePCA(input, args)

    store_data(principal_components, args.inputs_storing)
    store_data(output, args.targets_storing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Código de la práctica de bioinformática')
    parser.add_argument('--inputs_file', type=str)
    parser.add_argument('--targets_file', type=str)
    parser.add_argument('--pca_n', type=int)
    parser.add_argument('--inputs_storing', type=str)
    parser.add_argument('--targets_storing', type=str)
    args = parser.parse_args()

    main(args)
