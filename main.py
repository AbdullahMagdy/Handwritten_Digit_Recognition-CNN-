import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import preparing_data
import CNN_archi
import results_in_plots

# hyperparameters

input_N = 28
batch_size = 128
No_classes = 10
No_epochs = 10
LR = 0.001

x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,No_classes])

X_train ,  Y_train, X_test , Y_test = preparing_data.prepare_data()


# prediction part

pred = CNN_archi.conv_net(x, CNN_archi.weights, CNN_archi.biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start with tensorflow session

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(No_epochs):
        for batch in range(len(X_train) // batch_size):
            batch_x = X_train[batch * batch_size:min((batch + 1) * batch_size, len(X_train))]
            batch_y = Y_train[batch * batch_size:min((batch + 1) * batch_size, len(Y_train))]
            opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
        print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("Optimization Finished!")

        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: X_test, y: Y_test})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    summary_writer.close()


#plotting

results_in_plots.between_training_and_testing_loss(train_loss,test_loss)

results_in_plots.between_training_and_testing_acc(train_loss , test_loss , train_accuracy , test_accuracy)




