from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=1):
    ''' Build a model from X_train -> y_train '''
    y_train = one_hot(y_train)
    x_t = np.transpose(X_train)
    y_t = np.transpose(y_train)
    a = np.dot(x_t, X_train)
    a = np.subtract(a, np.dot(reg, np.identity(len(x_t))))
    a = np.linalg.inv(a)
    b = np.dot(x_t, y_train)
    print(np.shape(a))
    print(np.shape(b))
    return np.dot(a,b)
    #return np.zeros((X_train.shape[0], y_train.shape[0]))

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    #print(labels_train)
    return np.eye(NUM_CLASSES)[labels_train]
    #return np.zeros((X_train.shape[0], NUM_CLASSES))

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    prediction = np.dot(X, model)
    return np.argmax(prediction, axis=1)
    #return np.zeros(X.shape[0])

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    model = train(X_train, labels_train)
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)


    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
