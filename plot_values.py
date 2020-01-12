import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from mpl_toolkits.mplot3d import Axes3D
import pickle

def get_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            matrix.append(row)

    return np.array(matrix)


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
# Thanks to dennybritz and its Github repository to provide this function.
def plot_decision_boundary(pred_func, X, y, class_tag):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, marker='.', \
                                                    label='Clase %d'%class_tag)

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

def process_targets(target_data):
    target_size = np.shape(target_data)

    final_targets = []
    for row in range(0, target_size[0]):
        final_targets.append(convert_vector_to_number(target_data[row, :]))

    return final_targets

def plot_values(args, loaded_model):
    inputs = get_data(args.inputs_file).astype(float)
    outputs = get_data(args.targets_file).astype(int)
    converted_outputs = np.transpose(process_targets(outputs))
    for it in range(0, 5):
        plt.scatter(inputs[outputs[:, it]==1, 0], inputs[outputs[:, it]==1, 1], \
                        c=converted_outputs[converted_outputs==it], marker='.', \
                            cmap=plt.cm.Spectral)
        plot_decision_boundary(lambda x: loaded_model.predict(x), inputs, \
                                                        converted_outputs, it)
    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.savefig('values_2d.png')

def plot_values_3d(args, loaded_model):
    inputs = get_data(args.inputs_file).astype(float)
    outputs = get_data(args.targets_file).astype(int)
    colours = ['r', 'm', 'b', 'k', 'y']
    markers = ['.', '.', '.', '.', '.']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for it in range(0, 5):
        ax.scatter(inputs[outputs[:, it]==1, 0], inputs[outputs[:, it]==1, 1], \
                        inputs[outputs[:, it]==1, 2], color=colours[it],
                            label='Clase %d'%it, marker=markers[it])
    ax.set_xlabel('Atributo 1')
    ax.set_ylabel('Atributo 2')
    ax.set_zlabel('Atributo 3')
    plt.legend()
    plt.savefig('values_3d.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Código de la práctica de bioinformática')
    parser.add_argument('--inputs_file', type=str)
    parser.add_argument('--targets_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dim', type=int)
    args = parser.parse_args()

    loaded_model = pickle.load(open(args.model, 'rb'))
    if (args.dim==2):
        plot_values(args, loaded_model)
    elif (args.dim==3):
        plot_values_3d(args, loaded_model)
    else:
        print('Incorrect dimension value (must be 2 or 3)')
