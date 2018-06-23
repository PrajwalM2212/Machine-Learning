import numpy as np


def prediction(W,X):
    value = W[0];
    for i in range(len(W)-1):
        value += W[i+1]*X[i];
    if(value>=0):
        return 1;
    else:
        return 0;

    
def train_perceptron(dataset,num_epoch,l_rate):

    weights = [0.0 for i in range(len(dataset[0]))]
    for epoch in range(num_epoch):

        for row in dataset:
            y_hat = prediction(weights,row);
            error = row[-1] - y_hat;
            weights[0] += l_rate*error;
            for i in range(len(row)-1):
                weights[i+1] += l_rate*error*row[i];
            
    return weights;



dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]


l_rate = 0.01;
num_epoch = 50;
print(train_perceptron(dataset,num_epoch,l_rate));




