# Mega Case Study - Make a Hybrid Deep Learning Model


# Part 1 - Identify the Frauds with the Self-Organizing Map
# then in part 2, label the customers who are frauds, 
#    feed all customer features and classifications (fraud or not) to train Artificial Neural Network 
#    to predict probability of fraud classification based on customer features


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
    # each observation i is labeled with its class y_i
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
# MinMaxScaler: normalize data feature to values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM- Self-Organizing Map
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = X.shape[1], sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
# in 2D SOM, the higher the MID (mean interneuron dist) of winning node (that SOM identified),
    #(MID is mean of all dist of wining node from neighboring neurons inside neighborhood defined by sigma radius) 
    #the higher the MID of winning node, the more the winning node is the outlier, 
    # ie far from general rules respected by credit card customers
from pylab import bone, pcolor, colorbar, plot, show
bone()
dist_map = som.distance_map().T
pcolor(som.distance_map().T)
colorbar()
# identify if the winning node customers got approval or not for credit card app:
    # no approval: red circles (o); yes approval: green squares (s)
markers = ['o', 's']
colors = ['r', 'g']
# mappings = som.win_map(X) returns a dictionary/hashmap of (key, value): 
    # som.win_map(X) returns key=(x,y)=(w[0],w[1]) of winning node to which value=list of rows/customers(in x's) are mapped to
mappings = som.win_map(X)
first_fraud = True
frauds=np.zeros(X.shape[1])
frauds=np.reshape(frauds,(1,-1))
high_MID_winning_nodes = np.array([(0,0)])
# loop over all customers, get winning node for each customer, color that node as approved or not
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
            print(winning_node)
            frauds = np.concatenate((frauds, row), axis = 0)
            if winning_node not in high_MID_winning_nodes:
                high_MID_winning_nodes = np.concatenate((high_MID_winning_nodes,winning_node),axis=0)
                print("added winning node: ")
                print(winning_node)
                n=n+1

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5, 
         markers[y[i]], #y[i] for ith row is 0 or 1; marker[0] is circle, marker[1] is square
         markeredgecolor = colors[y[i]], #marker edge/outline color is color[0] red or color[1] green
         markerfacecolor = 'None', #marker face/fill color is 'None'
         markersize = 10,
         markeredgewidth = 2)
show()

frauds = sc.inverse_transform(frauds)
frauds_check=[]
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

frauds_check = sc.inverse_transform(frauds_check)

#compare if frauds and frauds_check array are same




# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
# dependent variable y: 0 is fraud customer, 1 is no fraud
# initialize is_fraud array of all customers to 0, 
# then if any customer ID (column 0, got by dataset.iloc[i,0]) is in frauds array, 
# set that customer is_fraud status/value to 1
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)): #range((default starts at) 0 to (set) len(dataset)-1), upper bound len(dataset) is excluded
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
# sort numpy array by 1 column- y_pred probabilities: 
y_pred = y_pred[y_pred[:, 1].argsort()]
