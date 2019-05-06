#!/usr/bin/env python3
# Python 2 compatibility
import numpy as np
import os
import sys
from datetime import datetime
import utlty as util

##################################################
#            READ PARAMETERS FOR DATA SET             #
##################################################
if len(sys.argv) != 6:
    print("usage: script.py nt nv seed n_epochs batch_size")
    print("nt:         number of training   samples")
    print("nv:         number of validation samples")
    print("seed:       seed for selecting training samples etc. (to make runs reproducible)")
    print("n_epochs:   number of training epochs")
    print("batch_size: size of batches for training (must not be bigger than nt)")
    quit()

nt = int(sys.argv[1]) #number of training samples
nv = int(sys.argv[2]) #number of validation samples
seed = int(sys.argv[3]) #seed for rng
n_epochs = int(sys.argv[4]) #number of training epochs
batch_size = int(sys.argv[5]) #number of samples in 1 batch
nsave = n_epochs//200 #saves approximately every 10% of the training
assert nsave != 0

#load dataset
print("Loading Dataset...\n")
data = util.DataContainer(nt, nv, seed, False)
print("total number of data:", data.num_data)
print("training examples:   ", data.num_train)
print("validation examples: ", data.num_valid)
print("test examples:       ", data.num_test)
print()

#fire up tensorflow
print("Loading TensorFlow...\n")
import tensorflow as tf

##################################################
#   PARAMETERS FOR NEURAL NETWORK AND TRAINING   #
##################################################
retrain_model = False #whether or not to retrain the model from an existing file
model_parameter_save= "save/"

#######To retrain the model
#retrain_model = True #whether or not to retrain the model from an existing file
#model_parameter_save = "save/NN-seed23-nt3000000-nv300000-24-24-24-24-24-24-24-24-24-1_loss0.00032417075_2019-02-06-08-48-38-21" #Use the saved model parameter

#number of input variables
n_inputs  = data.num_features
#number of neurons in each hidden layer (and how many hidden layers)
n_hidden = [24,24,24,24,24,24,24,24]
assert len(n_hidden) >= 1 #check that we have at least 1 hidden layer
#number of output variables
n_outputs = data.num_outputs

#learning rate parameter
#learning_rate = 5e-4 # default value I use is 1e-3

global_step = tf.Variable(0, trainable=False)
zero_global_step_op = tf.assign(global_step,0)
starter_learning_rate = 1.0e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000000, 0.85, staircase=False)
# Passing global_step to minimize() will increment it at each step.

#lambda multiplier for L2 regularization (0 -> no regularization)
l2_lambda = 0.0 

#for saving logs
now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
root_logdir = "logs"
#encode NN architecture in string
NNstr = "NN-seed"+str(seed)+"-nt"+str(data.num_train)+"-nv"+str(data.num_valid)+"-"+str(n_inputs)+"-"
for i in range(len(n_hidden)):
    NNstr = NNstr + str(n_hidden[i]) + "-"
NNstr  = NNstr + str(n_outputs)
logdir = "{}/run_".format(root_logdir)+NNstr+"_{}/".format(now)

#create neural network architecture
#function to create a layer of neurons, 
#getting X as input 
#with n_out outputs
def neuron_layer(X, n_out, activation_fn=lambda x: x, scope=None, factor=2.0):
    with tf.variable_scope(scope):
        #define layer
        n_in = X.shape[1].value
        W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=tf.sqrt(factor/(n_in+n_out))), name="W")
        b = tf.Variable(tf.truncated_normal([n_out], stddev=0), name="b")
        y = activation_fn(tf.add(tf.matmul(X,W),b))

        #L2 loss term for regularization
        l2_W = tf.nn.l2_loss(W, name="l2_W") 

        #add to collections
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_W)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        tf.add_to_collection(tf.GraphKeys.BIASES, b)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, y)

        #create histogram summaries for monitoring the weights and biases
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases",  b)

        return y

#function to create a Multilayer Perceptron getting X as feature input and Z as atom idendity encoding (as integer)
#n_hidden is a list containing the number of neurons in the hidden layers (at least 1 hidden layer)
#and n_outputs is how many output variables the MLP should have.
def MLP(X, n_hidden, n_outputs, activation_fn=lambda x: x, scope=None, factor=2.0):
    assert len(n_hidden) >= 1 #check that there is at least 1 hidden layer
    with tf.name_scope(scope):
        #list that stores the hidden layers
        hidden = []
        #create first hidden layer
        hidden.append(neuron_layer(X, n_hidden[0], activation_fn=activation_fn, scope="hidden0", factor=factor))
        #create rest of hidden layers

        hidden.append(neuron_layer(hidden[0], n_hidden[1], scope="hidden1", factor=2.0))
        hidden[1]=tf.nn.relu(hidden[1])+X

        hidden.append(neuron_layer(hidden[1], n_hidden[2], activation_fn=activation_fn, scope="hidden"+str(2), factor=factor))
        hidden.append(neuron_layer(hidden[2], n_hidden[3], scope="hidden"+str(3), factor=2.0))
        hidden[3]=tf.nn.relu(hidden[3])+hidden[1]

        hidden.append(neuron_layer(hidden[3], n_hidden[4], activation_fn=activation_fn, scope="hidden"+str(4), factor=factor))
        hidden.append(neuron_layer(hidden[4], n_hidden[5], scope="hidden"+str(5), factor=2.0))
        hidden[5]=tf.nn.relu(hidden[5])+hidden[3]

        hidden.append(neuron_layer(hidden[5], n_hidden[6], activation_fn=activation_fn, scope="hidden"+str(6), factor=factor))
        hidden.append(neuron_layer(hidden[6], n_hidden[7], scope="hidden"+str(7), factor=2.0))
        hidden[7]=tf.nn.relu(hidden[7])+hidden[5]

        return tf.nn.sigmoid(neuron_layer(hidden[len(n_hidden)-1], n_outputs, scope="output", factor=2.0))*0.4

#activation functions
def tanh(x): #scaled tanh
    return 1.592537419722831*tf.tanh(x)

def asinh(x): #scaled asinh
    return 1.256734802399369*tf.log(x+tf.sqrt(x*x+1.0))

def actf(x):
    return tf.nn.softplus(x)-tf.log(2.0)

#create neural network (and placeholders for feeding)
print("Creating neural network...\n")
X = tf.placeholder(tf.float32, shape=[None, n_inputs],  name="X") 
y = tf.placeholder(tf.float32, shape=[None, n_outputs], name="y") 

#NOTE: factor = 1 is only for the self-normalizing input functions!
yhat =  MLP(X, n_hidden, n_outputs, activation_fn=asinh,  scope="neuralnetwork", factor=1.0) #ordinary NN

#define loss function: here RMSE + regularization loss
with tf.name_scope("loss"):
    l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.reduce_mean(tf.squared_difference(tf.log(y+1.0),tf.log(y+tf.abs(tf.subtract(y,yhat))+1.0))) + l2_lambda*l2_loss

#define score function (performance measure, here: MAE)
with tf.name_scope("score"):
    score = tf.reduce_mean(tf.abs(tf.subtract(y,yhat)))

with tf.name_scope("rmsd"):
    rmsd =  tf.reduce_mean(tf.squared_difference(y,yhat))

#define training method
with tf.name_scope("train"):
    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

#for logging stats
#mean absolute error
score_for_train = tf.constant(0.0)
score_for_valid = tf.constant(0.0)
score_for_best  = tf.constant(0.0)
tf.summary.scalar("score-train", score_for_train)
tf.summary.scalar("score-valid", score_for_valid)
tf.summary.scalar("score-best",  score_for_best)

#Loss function
loss_for_train = tf.constant(0.0)
loss_for_valid = tf.constant(0.0)
loss_for_best  = tf.constant(0.0)
tf.summary.scalar("loss-train", loss_for_train)
tf.summary.scalar("loss-valid", loss_for_valid)
tf.summary.scalar("loss-best", loss_for_best)

#root mean squared error
rmsd_for_train = tf.constant(0.0)
rmsd_for_valid = tf.constant(0.0)
rmsd_for_best = tf.constant(0.0)
tf.summary.scalar("rmsd-train", rmsd_for_train)
tf.summary.scalar("rmsd-valid", rmsd_for_valid)
tf.summary.scalar("rmsd-best", rmsd_for_best)

#merged summary op
summary_op = tf.summary.merge_all()
#create file writer for writing out summaries
file_writer = tf.summary.FileWriter(logdir=logdir, 
                                    graph=tf.get_default_graph(),
                                    flush_secs=120)

#define saver nodes (max_to_keep=None lets the saver keep everything)
saver_best = tf.train.Saver(name="saver_best",max_to_keep=50) #saves only the x best model
saver_step = tf.train.Saver(name="saver_step",max_to_keep=200) #saves checkpoint every few steps

#counter that keeps going up for the best models
number_best = 0 

#train the model
score_best = np.finfo(dtype=float).max #initialize best score to huge value
loss_best = np.finfo(dtype=float).max #initialize best loss to huge value
rmsd_best = np.finfo(dtype=float).max #initialize best loss to huge value

#get the complete training and validation set
X_train, y_train = data.get_train_data()
X_valid, y_valid = data.get_valid_data()

print("Starting training...\n")
with tf.Session() as sess:
    #initialize variables
    if retrain_model:
        saver_step.restore(sess, model_parameter_save)
        sess.run(zero_global_step_op)

    else:
        tf.global_variables_initializer().run()
        
    for epoch in range(n_epochs):
        for iteration in range(data.num_train // batch_size):
            X_batch, y_batch = data.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
       
        score_train, loss_train, rmsd_train = sess.run([score,loss,rmsd], feed_dict={X: X_train, y: y_train})
        score_valid, loss_valid, rmsd_valid = sess.run([score,loss,rmsd], feed_dict={X: X_valid, y: y_valid})

        #calculate current models average score
        score_avg = (data.num_train*score_train + data.num_valid*score_valid)/(data.num_train+data.num_valid)
        #save best model
        if(score_valid < score_best):
            score_best = score_valid

        if(loss_valid < loss_best):
            loss_best = loss_valid
            saver_best.save(sess, "./save/"+NNstr+"_loss"+str(np.sqrt(loss_best))+"_"+now,global_step=number_best)
            number_best += 1

        if(rmsd_valid < rmsd_best):
            rmsd_best = rmsd_valid

        #save every few epochs
        if epoch%(n_epochs//nsave) == 0:
            saver_step.save(sess, "./save/"+NNstr+"_step",global_step=epoch)
            print("saved model at epoch " + str(epoch))

        #print progress to console
        lrate=sess.run(learning_rate)
        print(epoch,"/",n_epochs, "loss:",np.sqrt(loss_best), "score train:", score_train, "validation:", score_valid, "best:", score_best, "rmsd train:",np.sqrt(rmsd_train), "validation:",np.sqrt(rmsd_valid), "best:", np.sqrt(rmsd_best),"Lrate:",lrate)
        #log process
        summary_str = summary_op.eval(feed_dict={score_for_train: score_train, score_for_valid: score_valid, score_for_best: score_best, loss_for_train: np.sqrt(loss_train), loss_for_valid: np.sqrt(loss_valid), rmsd_for_train: np.sqrt(rmsd_train), rmsd_for_valid: np.sqrt(rmsd_valid), loss_for_best: np.sqrt(loss_best), rmsd_for_best: np.sqrt(rmsd_best)})
        file_writer.add_summary(summary_str, epoch)
        sys.stdout.flush()
    
    #cleanup
    file_writer.close()
    saver_step.save(sess, "./save/"+NNstr+"_step",global_step=n_epochs)
