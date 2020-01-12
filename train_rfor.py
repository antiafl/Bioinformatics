import argparse
from sklearn.ensemble import RandomForestClassifier as rfc
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import pickle

# En esta función se obtiene la matriz correspondiente al fichero .csv que
# se le pase como parámetro.
def get_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            matrix.append(row)

    return np.array(matrix)

def store_metrics(precision, sensivity, auc_roc, auc_pr, path):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['', 'Global', '0', '1', '2', '3', '4'])
        writer.writerow(['Precisión', precision, '', '', '', '', ''])
        writer.writerow(['Sensibilidad', '', sensivity[0], sensivity[1], \
                                    sensivity[2], sensivity[3], sensivity[4]])
        writer.writerow(['AUC-ROC', '', auc_roc[0], auc_roc[1], \
                                    auc_roc[2], auc_roc[3], auc_roc[4]])
        writer.writerow(['AUC-PR', '', auc_pr[0], auc_pr[1], \
                                    auc_pr[2], auc_pr[3], auc_pr[4]])

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

def calculate_precision(conf_matrix, num_of_classes):
    precision = 0.0
    for aux_class in range(0, num_of_classes):
        precision+=conf_matrix[aux_class, aux_class]
    precision/=np.sum(conf_matrix)

    return precision

def calculate_sensivity(conf_matrix, selected_class):
    sensivity = 0.0
    sensivity = conf_matrix[selected_class, selected_class]/np.sum(conf_matrix[selected_class])

    return sensivity

def calculate_auc_roc(targets, outputs):
    fpr, tpr, _ = roc_curve(targets, outputs)
    roc_auc = auc(fpr, tpr)

    return roc_auc

def calculate_auc_pr(targets, outputs):
    pr_auc = average_precision_score(targets, outputs)

    return pr_auc

def show_roc_curve():
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

def main(args):
    input_data = get_data(args.inputs_file).astype(float)
    target_data = get_data(args.targets_file).astype(int)

    classifier = rfc(max_depth=4, random_state=0)
    target_data = process_targets(target_data)
    noftrains = args.noftrains
    classes = 5
    chosen_test_size = args.test_size

    precision_list = []
    sensivity_list = []
    auc_roc_list = []
    auc_pr_list = []
    best_precision = -1
    model_filename = '%s/%s_rfor.pth'%(args.models_path, \
                                args.inputs_file.replace('.csv', ''))
    for it in range(0, noftrains):
        # Separación aleatoria de los conjuntos de entrenamiento y test.
        train_input, test_input, train_output, test_output = \
            train_test_split(input_data, target_data, test_size=chosen_test_size)
        classifier.fit(train_input, train_output)
        prediction = classifier.predict(test_input)
        decision = classifier.predict_proba(test_input)
        conf_matrix = confusion_matrix(test_output, prediction)
        precision = calculate_precision(conf_matrix, classes)*100
        precision_list.append(precision)
        sensivity = []
        auc_roc = []
        auc_pr = []
        for class_selected in range(0, classes):
            sensivity.append(calculate_sensivity(conf_matrix, class_selected)*100)
            output_vectors = np.array(convert_number_to_vector(test_output))
            prediction_vectors = np.array(convert_number_to_vector(prediction))
            auc_roc.append(calculate_auc_roc(output_vectors[:, class_selected], \
                                        prediction_vectors[:, class_selected])*100)
            auc_pr.append(calculate_auc_pr(output_vectors[:, class_selected], \
                                        prediction_vectors[:, class_selected])*100)
        sensivity_list.append(sensivity)
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)

        if (precision>best_precision):
            best_precision = precision
            pickle.dump(classifier, open(model_filename, 'wb'))

    precision_string = '%.4f +- %.4f'%(np.mean(precision_list), \
                                                    np.std(precision_list))
    sensivity_string_list = []
    auc_roc_string_list = []
    auc_pr_string_list = []
    for class_selected in range(0, classes):
        sensivity_string = '%.4f +- %.4f'%(np.mean(sensivity_list[class_selected]),
                                                                    np.std(sensivity_list[class_selected]))
        sensivity_string_list.append(sensivity_string)
        auc_roc_string = '%.4f +- %.4f'%(np.mean(auc_roc_list[class_selected]), \
                                            np.std(auc_roc_list[class_selected]))
        auc_roc_string_list.append(auc_roc_string)
        auc_pr_string = '%.4f +- %.4f\n'%(np.mean(auc_pr_list[class_selected]), \
                                            np.std(auc_pr_list[class_selected]))
        auc_pr_string_list.append(auc_pr_string)

    results_path = '%s/results_rfor_%s'%(args.results_path, args.inputs_file)
    store_metrics(precision_string, sensivity_string_list, auc_roc_string_list, \
                    auc_pr_string_list, results_path)
    print('Results were stored at %s'%results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Código de la práctica de bioinformática')
    parser.add_argument('--inputs_file', type=str)
    parser.add_argument('--targets_file', type=str)
    parser.add_argument('--noftrains', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--models_path', type=str)
    args = parser.parse_args()

    main(args)
