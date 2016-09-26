from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy

NUM_CLASSES = 10
d = 5000
b = np.random.uniform(0, 2 * np.pi, d)
variance = 0.2


G = np.random.normal(0, variance, 784 * d).reshape((784, d))

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    #y_train = one_hot(y_train)
    x_t = np.transpose(X_train)
    y_t = np.transpose(y_train)
    a = np.dot(x_t, X_train)
    a = np.subtract(a, np.dot(reg, np.identity(len(x_t))))
    a = np.linalg.inv(a)
    b = np.dot(x_t, y_train)
    return np.dot(a,b)

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    return np.zeros((X_train.shape[1], NUM_CLASSES))

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    return np.zeros((X_train.shape[1], NUM_CLASSES))

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    prediction = np.dot(X, model)
    return np.argmax(prediction, axis=1)

def phi(X):
    ''' Featurize the inputs using random Fourier features '''

    print(np.shape(G))
    print(np.shape(X))
    new_matrix = np.dot(X, G)

    # for x_row in X:
    #     new_array = []
    #     for row in G:
    #         row = row.reshape(1, shape[1])
    #         x_col = np.array(x_row).reshape(shape[1], 1)
    #         term = np.dot(row, x_col)[0]
    #         new_array.append(term)
    #     new_vector = np.array(new_array)
    #     #new_vector shape: (50, 1)
    #     # cosine_term = new_vector + b
    #     # cosined = np.cos(cosine_term)
    #     # final_vector = np.sqrt(float(2)/d)*cosined
    #     new_matrix.append(new_vector)
    # new_matrix = np.squeeze(new_matrix, axis=(2,))
    # print(np.shape(new_matrix))
    new_matrix += b
    new_matrix = np.cos(new_matrix)
    new_matrix = np.sqrt(2.0/d)*new_matrix
    print(np.shape(new_matrix))
    return new_matrix


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train), phi(X_test)

    model = train(X_train, y_train, reg=0.1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=20000)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Batch gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=100000)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Stochastic gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
