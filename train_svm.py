import argparse
from sklearn import svm
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# En esta función se obtiene la matriz correspondiente al fichero .csv que
# se le pase como parámetro.
def get_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            matrix.append(row)

    return np.array(matrix)

# Esta función se utiliza para convertir los vectores de la matriz target en
# un número entero.
def convert_vector_to_number(target_vector):
    if (target_vector[0]==1):
        output_value = 0
    if (target_vector[1]==1):
        output_value = 1
    if (target_vector[2]==1):
        output_value = 2
    if (target_vector[3]==1):
        output_value = 3
    if (target_vector[4]==1):
        output_value = 4

    return output_value

# Esta función se utiliza para convertir un número entero representativo de un
# tag en un vector.
def convert_number_to_vector(values):
    vectors = []
    for target_value in values:
        if (target_value==0):
            output_value = [1, 0, 0, 0, 0]
        if (target_value==1):
            output_value = [0, 1, 0, 0, 0]
        if (target_value==2):
            output_value = [0, 0, 1, 0, 0]
        if (target_value==3):
            output_value = [0, 0, 0, 1, 0]
        if (target_value==4):
            output_value = [0, 0, 0, 0, 1]
        vectors.append(output_value)

    return vectors

def process_targets(target_data):
    target_size = np.shape(target_data)

    final_targets = []
    for row in range(0, target_size[0]):
        final_targets.append(convert_vector_to_number(target_data[row, :]))

    return final_targets

def main(args):
    input_data = get_data(args.inputs_file).astype(float)
    target_data = get_data(args.targets_file).astype(int)

    classifier = svm.SVC()
    target_data = process_targets(target_data)

    classifier.fit(input_data[0:500], target_data[0:500])
    prediction = classifier.predict(input_data[500:801])
    decision = classifier.decision_function(input_data[500:801])
    score = classifier.score(input_data[500:801], target_data[500:801])

    target = np.array(convert_number_to_vector(target_data[500:801]))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 5

    plt.figure()
    colours = ['red', 'green', 'orange', 'brown', 'black']
    for it in range(0, n_classes):
        fpr[it], tpr[it], _ = roc_curve(target[:, it], decision[:, it])
        roc_auc[it] = auc(fpr[it], tpr[it])
        plt.plot(fpr[it], tpr[it], color=colours[it])

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Código de la práctica de bioinformática')
    parser.add_argument('--inputs_file', type=str)
    parser.add_argument('--targets_file', type=str)
    args = parser.parse_args()

    main(args)
