#!/usr/bin/python3
import time
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from datetime import datetime

def show_image(pixels, labels, ind, shape):

    pixels = pixels[ind].reshape(shape)
    
    plt.imshow(pixels, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.title(int(labels[ind]))
    plt.axis("off")
    plt.show()
    
def fetch_batch(X, Y, m, epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)  
    X_batch = X[indices] 
    y_batch = Y[indices] 
    return X_batch, y_batch

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')

    return dict

## Assumes RGB layers in [R...G...B] format in one image per row
## img_size is defined such that image is img_size x img_size
def reshape_image(X_bx, m, img_size):

    tot_img_size = img_size * img_size
    
    X_bx_R = X_bx[:, 0:tot_img_size].reshape((m, img_size, img_size))
    X_bx_G = X_bx[:, tot_img_size:(2 * tot_img_size)].reshape((m, img_size, img_size))
    X_bx_B = X_bx[:, (2 * tot_img_size):(3 * tot_img_size)].reshape((m, img_size, img_size))

    X_bx = np.stack((X_bx_R, X_bx_G, X_bx_B), axis = -1)

    return X_bx


if __name__ == "__main__":


    data_dir = 'cifar-10-batches-py/'

    b1 = unpickle(data_dir + 'data_batch_1')
    b2 = unpickle(data_dir + 'data_batch_2')
    b3 = unpickle(data_dir + 'data_batch_3')
    b4 = unpickle(data_dir + 'data_batch_4')
    b5 = unpickle(data_dir + 'data_batch_5')

    tb = unpickle(data_dir + 'test_batch')

    label_names = unpickle(data_dir + 'batches.meta')[b'label_names']

    #print(label_names)
    
    # X: 1024 R x 1024 G x 1024 B
    # image is 32 x 32
    X_b1, Y_b1 = b1[b'data'], b1[b'labels']
    X_b2, Y_b2 = b2[b'data'], b2[b'labels']
    X_b3, Y_b3 = b3[b'data'], b3[b'labels']
    X_b4, Y_b4 = b4[b'data'], b4[b'labels']
    X_b5, Y_b5 = b5[b'data'], b5[b'labels']

    X_tb, Y_tb = tb[b'data'], b5[b'labels']
    
    n_batch1 = len(X_b1)
    n_batch2 = len(X_b2)
    n_batch3 = len(X_b3)
    n_batch4 = len(X_b4)
    n_batch5 = len(X_b5)

    n_testbatch = len(X_tb)

    img_size = 32

    X_b1 = reshape_image(X_b1, n_batch1, img_size)
    X_b2 = reshape_image(X_b2, n_batch2, img_size)
    X_b3 = reshape_image(X_b3, n_batch3, img_size)
    X_b4 = reshape_image(X_b4, n_batch4, img_size)
    X_b5 = reshape_image(X_b5, n_batch5, img_size)

    X_tb = reshape_image(X_tb, n_testbatch, img_size)

    # plt.imshow(X_tb[500, :, :])
    # plt.show()

    # Convert list into numpy array
    Y_b1 = np.asarray(Y_b1)
    Y_b2 = np.asarray(Y_b2)
    Y_b3 = np.asarray(Y_b3)
    Y_b4 = np.asarray(Y_b4)
    Y_b5 = np.asarray(Y_b5)

    Y_tb = np.asarray(Y_tb)
    
    print("Size of batches:", n_batch1, n_batch2, n_batch3, n_batch4, n_batch5)
    print()
    
    ## Define training, validation, test datasets
    # Training
    X_train = np.concatenate((X_b1, X_b2, X_b3, X_b4))
    Y_train = np.concatenate((Y_b1, Y_b2, Y_b3, Y_b4))

    ## For debugging
    # X_train = X_train[0:200, :]
    # Y_train = Y_train[0:200]
    
    # Validation
    X_val = X_b5
    Y_val = Y_b5

    # Test
    X_test = X_tb
    Y_test = Y_tb

    # ## Define useful numbers
    m_train = len(X_train)
    m_val = len(X_val)
    m_test = len(X_test)

    print("m_train:", m_train)
    print("m_val:", m_val)
    print("m_test:", m_test)
    print()
    
    ## Define parameters
    learning_rate = 0.01
    n_inputs = X_train.shape[1]
    n_outputs = len(label_names)
    n_epochs = 250
    batch_size = 200

    print("learning_rate:", learning_rate)
    print("n_epochs:", n_epochs)
    print("batch_size:", batch_size)
    

    ## Setup TensorBoard
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    X = tf.placeholder(tf.float32, shape = (None, img_size, img_size, 3), name = "X")
    Y = tf.placeholder(tf.int32, shape = (None), name = "Y")

    
    # AlexNet (-ish)
    with tf.name_scope("AlexNet"):

        # C1 - Convolution
        C1 = tf.layers.conv2d(X, filters = 96, kernel_size = [11, 11], strides = [4, 4],
                              padding = 'same', activation = tf.nn.relu, name = "C1")

        # N1 - Local response normalization
        N1 = tf.nn.local_response_normalization(C1, depth_radius = 2, bias = 1,
                                                alpha = 0.00002, beta = 0.75, name = "N2")
        
        # S2 - Max Pooling
        S2 = tf.layers.max_pooling2d(N1, pool_size = [3, 3], strides = [2, 2],
                                     padding = 'valid', name = "S2")

        # C3 - Convolution
        C3 = tf.layers.conv2d(S2, filters = 256, kernel_size = [5, 5], strides = [1, 1],
                              padding = 'same', activation = tf.nn.relu, name = "C3")

        # N3 - Local response normalization
        N3 = tf.nn.local_response_normalization(C3, depth_radius = 2, bias = 1,
                                                alpha = 0.00002, beta = 0.75, name = "N2")

        # S4 - Max Pooling
        S4 = tf.layers.max_pooling2d(N3, pool_size = [3, 3], strides = [2, 2],
                                     padding = 'valid', name = "S4")

        # C5 - Convolution
        C5 = tf.layers.conv2d(S4, filters = 384, kernel_size = [3, 3], strides = [1, 1],
                              padding = 'same', activation = tf.nn.relu, name = "C5")

        # C6 - Convolution
        C6 = tf.layers.conv2d(C5, filters = 384, kernel_size = [3, 3], strides = [1, 1],
                              padding = 'same', activation = tf.nn.relu, name = "C6")
        
        # C7 - Convolution
        C7 = tf.layers.conv2d(C6, filters = 256, kernel_size = [3, 3], strides = [1, 1],
                              padding = 'same', activation = tf.nn.relu, name = "C7")

        # F8 - Fully Connected
        F8 = tf.layers.dense(C7, units = 256, activation = tf.nn.relu, name = "F8")

        # D8 - Dropout at 50%
        D8 = tf.layers.dropout(F8, rate = 0.5)
        
        # F9 - Fully Connected
        F9 = tf.layers.dense(D8, units = 256, activation = tf.nn.relu, name = "F9")

        # D9 - Dropout at 50%
        D9 = tf.layers.dropout(F9, rate = 0.5)

        
        # Output - Fully Connected
        logits = tf.layers.dense(F9, units = n_outputs, name = "logits")

        logits = tf.layers.flatten(logits)

    with tf.name_scope("loss"):

        loss = tf.losses.sparse_softmax_cross_entropy(labels = Y, logits = logits)

    with tf.name_scope("train"):

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):

        correct = tf.nn.in_top_k(logits, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        prediction = tf.argmax(logits, axis = 1)
        precision, precision_op = tf.metrics.precision(Y, tf.argmax(logits, axis = 1))
        recall, recall_op = tf.metrics.recall(Y, prediction)

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)

        write_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    saver = tf.train.Saver()
    
    start_time = time.time()
    
    with tf.Session() as sess:

        init.run()
        init_local.run()

        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(logdir + '/val', sess.graph)

        n_batches = int(m_train/batch_size)

        print("n_batches", n_batches)
        print()

        for epoch in range(n_epochs):

            for b in range(n_batches):
                
                X_batch, Y_batch = fetch_batch(X_train, Y_train, m_train, epoch, b, batch_size)

                _, prec_train, recall_train, accuracy_train = sess.run([train_op, precision_op, recall_op, accuracy], feed_dict = {X: X_batch, Y: Y_batch})

                train = write_op.eval(feed_dict = {X: X_batch, Y: Y_batch})
                train_writer.add_summary(train, epoch)

            prec_val, recall_val, accuracy_val = sess.run([precision_op, recall_op, accuracy], feed_dict = {X: X_val, Y: Y_val})

            validation = write_op.eval(feed_dict = {X: X_val, Y: Y_val})
            val_writer.add_summary(validation, epoch)
                
            print("Epoch:", epoch, "Train precision:", prec_train, "Train recall:", recall_train, "Train accuracy:", accuracy_train)
            print("Epoch:", epoch, "Val precision:", prec_val, "Val recall:", recall_val, "Val accuracy:", accuracy_val)
            print()

            save_path = saver.save(sess, "/home/ivy/Documents/CIFAR-10/AlexNet_model")
            
        train_writer.close()
        val_writer.close()

        
        
    print("Time elapsed", (time.time() - start_time)/60, "minutes")
    print()
    
        



        
