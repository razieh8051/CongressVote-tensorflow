import numpy as np
import pandas as pd
import tensorflow as tf
import os.path
from sklearn.model_selection import train_test_split


def create_model(inputs, weights, biases):
    #Create 2-layered NN
    input_layer= tf.matmul(inputs,weights['input'])
    # Hidden layer 1 (relu activation)
    hidden_layer1= tf.nn.relu(input_layer + biases['input'])
    #  Hidden layer 2 (relu activation)
    hidden_layer2= tf.nn.relu(tf.matmul(hidden_layer1,weights['hidden1'])+biases['hidden1'])
    # Output layer (linear activation)
    output_layer= tf.matmul(hidden_layer2, weights['hidden2']) + biases['hidden2']

    return output_layer

#First Column is the label and others are the features
COLUMNS = ['className','handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution',
'physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban',
'aid-to-nicaraguan-contras','mx-missle','immigration','synfuels-corporation-cutback','education-spending',
'superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa']

# Read data into CSV file and convert ? to NaN
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "./house-votes-84.data.txt")
                       , names=COLUMNS
                       , skipinitialspace=True
                       , na_values="?")

#Rows with NaN values are removed from data
df.replace(["NaN"], np.nan, inplace = True)
df = df.dropna()
df = df.reset_index(drop=True)
  
#New row numbers after removing NaN values
rows_number=df.shape[0]


#Label column republican/democrat to 1 and 0
for i in range(0,len(COLUMNS)):
    cleanup_nums = {COLUMNS[i]: {"y":1, "n":0, "republican":1,"democrat":0}}
    df.replace(cleanup_nums, inplace=True)

#Make 2 binary vectors for the labels column
labels=np.zeros((rows_number, 2))
label=df['className']
for i in range(0,len(label)):
    if(label[i]==0):
        labels[i,0]=0
        labels[i,1]=1
    else:
        labels[i,0]=1
        labels[i,1]=0

inputs=df[COLUMNS[2:]]
inputs=inputs.values

x_train, x_test, y_train, y_test = train_test_split(inputs, labels)

#HyperParameters Training
learning_rate = 0.5
epochs = 50
batch_size = 50 #small dataset

n_features = 15
n_calsses=2

#HyperParameters NN
hidden_nodes_layer1=10
hidden_nodes_layer2=10

#Define weights and biases
input_weights = tf.Variable(tf.random_normal([n_features, hidden_nodes_layer1]))
input_biases = tf.Variable(tf.zeros([1,hidden_nodes_layer1]))

hidden1_weights = tf.Variable(tf.random_normal([hidden_nodes_layer1, hidden_nodes_layer2]))
hidden1_biases = tf.Variable(tf.zeros([1,hidden_nodes_layer2]))

hidden2_weights = tf.Variable(tf.random_normal([hidden_nodes_layer2, n_calsses]))
hidden2_biases = tf.Variable(tf.zeros([1,n_calsses]))

weights={'input':input_weights,'hidden1':hidden1_weights,'hidden2':hidden2_weights}
biases={'input':input_biases,'hidden1':hidden1_biases,'hidden2':hidden2_biases}


x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_calsses])

#Create the NN model
prediction=create_model(x,weights,biases)

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with tf.Session() as sess:
    sess.run(init)    
    #Training cycle
    # Loop epochs
    for epoch in range(epochs):
        avg_cost = 0       
        total_batch = int(len(x_train)/batch_size)
        X_batches = np.array_split(x_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)
        # Loop batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization and loss 
            _, cost = sess.run([optimizer, loss_function], feed_dict={x: batch_x,
                                                          y: batch_y})
            # average cost
            avg_cost += cost / total_batch
        print("Epoch:", epoch+1, "cost=", avg_cost)
    
    #accuracy
    print("Accuracy: " + str(accuracy.eval(feed_dict={x: x_test,
                                                          y: y_test})))

