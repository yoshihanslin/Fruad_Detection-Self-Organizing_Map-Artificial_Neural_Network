# Self Organizing Map
# use SOM to identify fraud customers using credit application features X (y= app approved or not), 
# identify customer segments who have outlier features, 
        #ie outlier neurons in 2D SOM, who will be far from neighboring neurons in Euclidean dist

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# In unsupervised learning, only indep variables X (features) used, dep variable y (class) not used 
# In supervised learning, both indep variables X (features) and dep variable y (class) are used, 
    # each observation (data entry) i is labeled with its class y_i
dataset = pd.read_csv('Credit_Card_Applications.csv')
# feature dataset is all rows ( : ) and all columns except the last column ( :-1), upper bound -1 is excluded
X = dataset.iloc[:, :-1].values
numXColumns=X.shape[1]
# class dataset is all rows ( : ) and only last column (-1)
y = dataset.iloc[:, -1].values
#.values- turn into numpy array

# Feature Scaling
# MinMaxScaler: normalize: (x-min)/(max-min); normalize data feature values for RNN, to btwn 0 and 1 (feature range)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM- Self-Organizing Map
# make sure have minisom.py in working directory, from https://pypi.python.org/pypi/MiniSom/1.0
from minisom import MiniSom
#SOM map/grid dimensions 10x10, but can be bigger for more accuracy; 
#input_len= #features (14) and customer ID (1) to identify fraud customer
#sigma is radius of different neighborhoods in SOM grid
#learning_rate det how much weights are updated during each iteration of training, higher learning rate --> faster convergence
som = MiniSom(x = 10, y = 10, input_len = X.shape[1], sigma = 1.0, learning_rate = 0.5)
#randomly initialize weights close to but >0
som.random_weights_init(X)
#train SOM for 100 iterations by feeding input features X
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
# in 2D SOM, the higher the MID (mean interneuron dist) of winning node (that SOM identified),
    #(MID is mean of all dist of wining node from neighboring neurons inside neighborhood defined by sigma radius) 
    #the higher the MID of winning node, the more the winning node is the outlier, 
    # ie far from general rules respected by credit card customers
from pylab import bone, pcolor, colorbar, plot, show
# initialize the window that contains the map
bone()
# som.distance_map() returns a matrix of all the MID of winning nodes, .T transposes the matrix
# pcolor() assigns colors to different range values of MID
dist_map = som.distance_map().T
pcolor(som.distance_map().T)
# colorbar() adds legend for colors
colorbar()
# identify if the winning node customers got approval or not for credit card app:
    # no approval: red circles (o); yes approval: green squares (s)
markers = ['o', 's']
colors = ['r', 'g']
# mappings = som.win_map(X) returns a dictionary/hashmap of (key, value): 
    # som.win_map(X) returns key=(x,y)=(w[0],w[1]) of winning node to which value=list of rows/customers(in x's) are mapped to
mappings = som.win_map(X)
#fraud coordinate (x,y) are those whose MID are largest, close to 1 (white)
# mappings[key=(x,y)] returns value=list of rows/customers(in customer features x's) that are mapped to winning node (x,y)
first_fraud = True
# random initialization of frauds
frauds=np.zeros(X.shape[1])
frauds=np.reshape(frauds,(1,-1))
high_MID_winning_nodes = np.array([(0,0)])
# loop over all customers, get winning node for each customer, color that node as approved or not
# i= row index of customer, x= vector of customer features x's (each row in dataset)
# for row_index, row in enumerate(sequence):
#  w = som.winner(x): get winning node for customer/row x
#y = dataset.iloc[:, -1].values: each row has value 0 not approved or 1 approved
g=0
n=0
for i, x in enumerate(X):
    w = som.winner(x)
    vMID=dist_map[w[0]][w[1]]
    print(vMID)
    if vMID>0.979:
        row = np.reshape(np.array(x),(1,-1))
        winning_node=np.array([(w[0],w[1])])
        g=g+1
        if first_fraud==True:
            first_fraud = False
            frauds = row
            high_MID_winning_nodes = winning_node
        else:
            #concat. numpy arrays of fraud customer x's along the vertical axis (axis=0), ie each row below previous row; axis=1: horizontal concatenation
            print(winning_node)
            frauds = np.concatenate((frauds, row), axis = 0)
            if winning_node not in high_MID_winning_nodes:
                high_MID_winning_nodes = np.concatenate((high_MID_winning_nodes,winning_node),axis=0)
                print("added winning node: ")
                print(winning_node)
                n=n+1

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, #w[0]= x coordinate of winning node square at lower left corner, +0.5 to center of square
         w[1] + 0.5, #w[1]= y coordinate of winning node square at lower left corner, +0.5 to center of square
         markers[y[i]], #y[i] for ith row is 0 or 1; marker[0] is circle, marker[1] is square
         markeredgecolor = colors[y[i]], #marker edge/outline color is color[0] red or color[1] green
         markerfacecolor = 'None', #marker face/fill color is 'None'
         markersize = 10,
         markeredgewidth = 2)
show()
# inverse-scale all customer feature values, including IDs, from range btwn (0,1) to actual feature values, as configured above for sc
frauds = sc.inverse_transform(frauds)
#frauds_check=np.zeros(X.shape[1])
#frauds_check=np.reshape(frauds_check,(1,X.shape[1]))
frauds_check=[]
# mappings[key=(x,y)] returns value=list of rows/customers(in customer features x's) that are mapped to winning node (x,y)
for i,row in enumerate(high_MID_winning_nodes):
    x=row[0]
    y=row[1]
    f0=mappings[(x,y)]
    for i2,row2 in enumerate(mappings[(x,y)]):
        row2=np.reshape(row2,(1,-1))
        if i2==0:
            frauds_check=row2
        else:
            frauds_check=np.concatenate((frauds_check, row2), axis = 0)
#    frauds_check.append(mappings[(x,y)]) #for list, not numpy
frauds_check = sc.inverse_transform(frauds_check)

#compare if frauds and frauds_check array are same