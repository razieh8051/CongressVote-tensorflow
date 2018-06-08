import numpy as np
import pandas as pd
import tensorflow as tf



def create_model(inputs,nfeatures,nclasses):

    hidden_nodes_layer1=10
    hidden_nodes_layer2=10

    input_weights = tf.Variable(tf.truncated_normal([nfeatures, hidden_nodes_layer1]))
    input_biases = tf.Variable(tf.zeros([1,hidden_nodes_layer1]))

    hidden1_weights = tf.Variable(tf.truncated_normal([hidden_nodes_layer1, hidden_nodes_layer2]))
    hidden1_biases = tf.Variable(tf.zeros([1,hidden_nodes_layer2]))

    hidden2_weights = tf.Variable(tf.truncated_normal([hidden_nodes_layer2, nclasses]))
    hidden2_biases = tf.Variable(tf.zeros([1,nclasses]))

    #Create 2-layered NN
    input_layer= tf.matmul(inputs,input_weights)
    hidden_layer1= tf.nn.relu(input_layer + input_biases)

    hidden_layer2= tf.nn.relu(tf.matmul(hidden_layer1,hidden1_weights)+hidden1_biases)
    output_layer= tf.matmul(hidden_layer2, hidden2_weights) + hidden2_biases

    return output_layer

COLUMNS = ['className','handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution',
'physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban',
'aid-to-nicaraguan-contras','mx-missle','immigration','synfuels-corporation-cutback','education-spending',
'superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa']

#read data into CSV file and convert ? to NaN
df = pd.read_csv('house-votes-84.data.txt'
                       , names=COLUMNS
                       , skipinitialspace=True
                       , na_values="?")
#label column republican/democrat to 1 and 0
for i in range(0,len(COLUMNS)):
    cleanup_nums = {COLUMNS[i]: {"y":1, "n":0, "republican":1,"democrat":0}}
    df.replace(cleanup_nums, inplace=True)

#label=np.array(df[0])
lables=np.zeros((435, 2))
label=df['className']
for i in range(0,len(label)):
    if(label[i]==0):
        lables[i,0]=0
        lables[i,1]=1
    else:
        lables[i,0]=1
        lables[i,1]=0
print(lables)
inputs=df[COLUMNS[2:]]
inputs=inputs.values
print inputs
print inputs.shape

n_calsses= 2
n_features = 15
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_calsses])

prediction=create_model(x,n_features,n_calsses)

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
costs = []
batch_size = 1

with tf.Session() as sess:
    sess.run(init)    
    # Training cycle
    for epoch in range(201):
        optimizer.run(feed_dict={x: inputs, y: lables})
        if ((x+1) % 100 == 0):
            print("Training epoch " + str(x+1))
            print("Accuracy: " + str(optimizer.run(feed_dict={x: inputs, y: lables})
  

        # step = sess.run(optimizer, feed_dict={x: inputs, y: lables})
        # print(step)
