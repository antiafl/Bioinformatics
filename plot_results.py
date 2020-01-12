import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse
from os import listdir

def get_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            matrix.append(row)

    return np.array(matrix)

def plot_results_of_a_train(filename, colour, train_name, opt):
    matrix = get_data(filename)
    nofclasses = 5
    figures = ['^', '*', '<', '>', 'x']

    precision = np.array(matrix[1, 1].split('+-')).astype(float)
    sensivity = []
    auc_roc = []
    auc_pr = []

    for it in range(0, 5):
        sensivity.append(np.array(matrix[2, it+2].split('+-')).astype(float))
        auc_roc.append(np.array(matrix[3, it+2].split('+-')).astype(float))
        auc_pr.append(np.array(matrix[4, it+2].split('+-')).astype(float))
        plt.plot(auc_pr[it][1], auc_pr[it][0], '%s%s'%(colour, figures[it]))

    plt.xlabel('Desviación típica (%s)'%'%')
    plt.ylabel('Media (%s)'%('%'))
    plt.plot(precision[1], precision[0], '%so'%colour, label=train_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Código de la práctica de bioinformática')
    parser.add_argument('--title', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--opt', type=int)
    args = parser.parse_args()

    # Where 0 means using only one model and 1 means using several models.
    if (args.opt!=0 and args.opt!=1):
        print('Error: --opt must be zero or one.')
        exit(0)

    files_list = listdir(args.input_dir)
    colours = ['k', 'm', 'r', 'g']
    it = 0
    plt.title(args.title)
    for filename in files_list:
        if (args.opt==0):
            train_name = filename.replace('results_svm_inputs_', '').replace('.csv', '')
            train_name = train_name.replace('results_kNN_inputs_', '').replace('.csv', '')
            train_name = train_name.replace('results_rfor_inputs_', '').replace('.csv', '')
            train_name = train_name.replace('_', ' de ')
            train_name = '%s atributos'%train_name
        if (args.opt==1):
            train_name = filename.replace('results_', '').replace('.csv', '')
            train_name = train_name.replace('inputs_', '').replace('.csv', '')
            train_name = train_name.upper()

        plot_results_of_a_train('%s/%s'%(args.input_dir, filename), \
                                    colours[it], train_name, args.opt)
        it+=1

    plt.legend()
    plt.savefig('%s.png'%args.input_dir)
