import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer,MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(csv_file_name,col_name_list):
    data_frame = pd.read_csv(csv_file_name,names = col_name_list)
    data_array = data_frame.values
    feature = data_array[:,:-1]
    label = data_array[:,-1]
    return feature,label

def split_data_set(x_data,y_data,t_percent):
    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=t_percent)
    return x_train,x_test,y_train,y_test


def normalization_data(norm_type, data_set):
    if norm_type == "l1":
        normlizer = Normalizer(norm = 'l1')
        norm_data = normlizer.fit_transform(data_set)
    if norm_type == "l2":
        normlizer = Normalizer(norm = "l2")
        norm_data = normlizer.fit_transform(data_set)
    if norm_type =="min_max":
        normlizer = MinMaxScaler(feature_range=(0,1))
        norm_data = normlizer.fit_transform(data_set)
    return norm_data


def sigmoid(z):
    return 1.0  / (1 + np.exp(-z))

def logistic_value_func(x,theta):
    test_func = sigmoid(np.dot(np.c_[np.ones(x.shape[0]),x],theta.T))
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]),x],theta.T))

def logistic_gradient(x,y,theta):
    y_hat = logistic_value_func(x,theta)
    # print y_hat.astype("float64") - y.astype("float64")

    grad = np.dot((y_hat - y).T,np.c_[np.ones(x.shape[0]),x])

    return grad/x.shape[0]


def logistic_regression(theta,x_train,y_train,max_epoch,batch_size, lr = 0.03, converge_change = .000001,momentum = None):

    error = np.zeros(theta.shape[0])
    i = 1
    while i < max_epoch:
        order = np.asarray(range(0,len(x_train)))
        np.random.shuffle(order)
        x_train = x_train[order]
        y_train = y_train[order]

        for batch in xrange(0,x_train.shape[0],batch_size):
            grad = logistic_gradient(x_train[batch:batch + batch_size,:],y_train[batch:batch_size+batch,:],theta)
            # print grad
            
            theta = theta - lr * grad
        
        if np.linalg.norm(theta - error) < converge_change:
            break
        else:
            error = theta
        i = i + 1
    
    return theta


def calculate_accuracy(x_test_set,y_test_set,theta):
    prob,value = predict(theta,x_test_set)
    right_answer = np.sum(value == y_test_set)
    all_target= y_test_set.shape[0]
    #rember should multipy 1.0
    accuracy = right_answer * 1.0 / all_target

    print("Coefficients:{}".format(theta))
    print("Accuracy:{}".format(accuracy))
    print("")


def predict(theta,x):
    predict_prob = logistic_value_func(x,theta)
    predict_value = np.where(predict_prob > 0.5,1,0)
    return predict_prob,predict_value




if __name__ == '__main__':
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    file_name = "pima-indians-diabetes.data.csv"
    x_data,y_data = load_data(file_name,columns)
    # print x_data[:100,]
    x_train,x_test,y_train,y_test = split_data_set(x_data,y_data,0.7)
    y_train = y_train[:,None]
    y_test = y_test[:,None]
    theta = np.random.rand(1,x_train.shape[1] + 1)

    fitted_theta = logistic_regression(theta,x_train,y_train,10,30)
    # print fitted_theta
    calculate_accuracy(x_test,y_test,fitted_theta)
