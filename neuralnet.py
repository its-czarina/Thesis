import tensorflow as tf
import numpy as np
from numpy import genfromtxt

# Convert to one hot
def convertOneHot(data):
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)

print ("getting data")
data = genfromtxt('training-data.csv',delimiter=',')  # Training data
test_data = genfromtxt('testing-data.csv',delimiter=',')  # Test data

print ("training")
x_train=np.array([ i[1::] for i in data])
y_train,y_train_onehot = convertOneHot(data)

x_test=np.array([ i[1::] for i in test_data])
y_test,y_test_onehot = convertOneHot(test_data)

print ("fix errthang")
#  A number of features, 4 in this example
#  B = 3 species of Iris (setosa, virginica and versicolor)
A=data.shape[1]-1 # Number of features, Note first is y
B=len(y_train_onehot[0])
tf_in = tf.placeholder("float", [None, A]) # Features
tf_weight = tf.Variable(tf.zeros([A,B]))
tf_bias = tf.Variable(tf.zeros([B]))
tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)

print ("thingy2")
# Training via backpropagation
tf_softmax_correct = tf.placeholder("float", [None,B])
tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))

print ("Train using tf.train.GradientDescentOptimizer")
# Train using tf.train.GradientDescentOptimizer
tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)

print ("# Add accuracy checking nodes")
tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

print( "# Initialize and run")
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print("...")
# Run the training
for i in range(30):
    sess.run(tf_train_step, feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})

# Print accuracy
    result = sess.run(tf_accuracy, feed_dict={tf_in: x_test, tf_softmax_correct: y_test_onehot})
    print ("Run {},{}",format(i,result))