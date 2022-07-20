import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Return fitted model parameters to the dataset at datapath for each choice in degrees.
# Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
# Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
# coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []
    data = read_csv(datapath)
    data = read_csv(datapath, sep = ' ', header=None)
    data = np.array(data)
    feature_x = data[:,0]
    for n in degrees:
        feature = feature_matrix(feature_x, n)
        param = least_squares(feature, data[:,1])
        paramFits.append(param)
    # fill in
    # read the input file, assuming it has two columns, where each row is of the form [x y] as
    # in poly.txt.
    # iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.

    return paramFits


# Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
# Input: x as a list of the independent variable samples, and d as an integer.
# Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
# for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = [[y**n for n in range(d, -1, -1)] for y in x] 

    return X


# Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
# Input: X as a list of features for each sample, and y as a list of target variable samples.
# Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = np.matmul(np.matmul(np.linalg.inv(X.T @ X), np.transpose(X)), y)
    return B

if __name__ == "__main__":
    datapath = "poly.txt"
    degrees = [1,2,3,4,5]

    paramFits = main(datapath, degrees)

    
    data = read_csv(datapath)
    data = read_csv(datapath, sep = ' ', header=None)
    data = np.array(data)
    x = data[:,0]
    y = data[:,1]
    poly_x = np.linspace(-6, 6, num = 1000)
    figure = plt.scatter(x, y, c = 'k', label = 'Data')
    for n in paramFits:
        polyY = [np.polyval(n, i) for i in poly_x]
        r2Y = [np.polyval(n, i) for i in x]
        plt.plot(poly_x, polyY, label = 'Polynomial Degree ' + str(len(n) - 1))
        print([np.polyval(n, 2)])
        print('Polynomial Degree ' + str(len(n) - 1) + '\'s r2: '  + str(r2_score(y, r2Y)))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    
