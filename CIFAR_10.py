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

def flip_images(X_imgs, Y_labels, img_size):

    X_flip = []
    Y_flip = []

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (img_size, img_size, 3))
    
    tf_lr = tf.image.flip_left_right(X)
    tf_ud = tf.image.flip_up_down(X)
    #tf_tr = tf.image.transpose_image(X)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        init.run()

        for img in range(len(X_imgs)):
            flipped = sess.run([tf_lr, tf_ud], feed_dict = {X: X_imgs[img, :, :, :]})

            X_flip.extend(flipped)
            Y_flip.extend(2 * [Y_labels[img]])


    X_flip = np.asarray(X_flip, dtype = np.float32)
    Y_flip = np.asarray(Y_flip)

    np.random.seed(5423)  
    indices = np.random.randint(len(X_flip), size = int(0.25 * len(X_flip)))
    X_flip = X_flip[indices, :, :, :]
    Y_flip = Y_flip[indices]

    return X_flip, Y_flip
        

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

    # plt.imshow(X_b1[500, :, :])
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
    
    # Flip images    
    FX_b1, FY_b1 = flip_images(X_b1, Y_b1, img_size)
    FX_b2, FY_b2 = flip_images(X_b2, Y_b2, img_size)
    FX_b3, FY_b3 = flip_images(X_b3, Y_b3, img_size)
    FX_b4, FY_b4 = flip_images(X_b4, Y_b4, img_size)
    FX_b5, FY_b5 = flip_images(X_b5, Y_b5, img_size)
        
    ## Define training, validation, test datasets
    # Training
    X_train = np.concatenate((X_b1, FX_b1, X_b2, FX_b2, X_b3, FX_b3, X_b4, FX_b4))
    Y_train = np.concatenate((Y_b1, FY_b1, Y_b2, FY_b2, Y_b3, FY_b3, Y_b4, FY_b4))

    # ## For debugging
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
    learning_rate = 0.001
    n_inputs = X_train.shape[1]
    n_outputs = len(label_names)
    n_epochs = 150
    batch_size = 50

    print("learning_rate:", learning_rate)
    print("n_epochs:", n_epochs)
    print("batch_size:", batch_size)
    

    ## Setup TensorBoard
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    X = tf.placeholder(tf.float32, shape = (None, img_size, img_size, 3), name = "X")
    Y = tf.placeholder(tf.int32, shape = (None), name = "Y")

    
    # CNN
    with tf.name_scope("CNN"):

        # C1 - Convolution
        C1 = tf.layers.conv2d(X, filters = 16, kernel_size = 5, strides = 1,
                                   padding = 'same', activation = tf.nn.relu, name = "C1")

        # S2 - Max Pooling
        S2 = tf.layers.max_pooling2d(C1, pool_size = 2, strides = 2,
                                         padding = 'valid', name = "S2")

        # C3 - Convolution
        C3 = tf.layers.conv2d(S2, filters = 20, kernel_size = 5, strides = 1,
                                   padding = 'same', activation = tf.nn.relu, name = "C3")

        # S4 - Max Pooling
        S4 = tf.layers.max_pooling2d(C3, pool_size = 2, strides = 2,
                                         padding = 'valid', name = "S4")

        # C5 - Convolution
        C5 = tf.layers.conv2d(S4, filters = 20, kernel_size = 5, strides = 1,
                                   padding = 'same', activation = tf.nn.relu, name = "C5")

        F6 = tf.layers.dense(C5, units = 120, activation = tf.nn.relu, name = "F6")

        D7 = tf.layers.dropout(F6, rate = 0.5, seed = 1568, name = "D7")

        # Output - Fully Connected
        logits = tf.layers.dense(D7, units = n_outputs, name = "logits")
        logits = tf.layers.flatten(logits)

    with tf.name_scope("loss"):

        loss = tf.losses.sparse_softmax_cross_entropy(labels = Y, logits = logits)

    with tf.name_scope("train"):

        # optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate,
        #                                        rho = 0.0001)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate,
        #                                       decay = 0.7,
        #                                       momentum = 0.3)

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        
        train_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):

        correct = tf.nn.in_top_k(logits, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        prediction = tf.argmax(logits, axis = 1)
        precision, precision_op = tf.metrics.precision(Y, tf.argmax(logits, axis = 1))
        recall, recall_op = tf.metrics.recall(Y, prediction)

        tf.summary.scalar('loss', loss)
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
    
        



        
