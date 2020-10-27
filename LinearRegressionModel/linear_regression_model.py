"""
Building a Linear Regression Model with TensorFlow
"""

# Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Model hyperparameters
learning_rate = 0.01
training_epochs = 2000
display_step = 200

# Training Dataset
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# Testing Dataset
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

# Placeholders for the predictor variables (x) and for the target variable (y)
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Model weights and biases
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Building the linear model
# Formula of the linear model: y = W*X + b
linear_model = W * X + b

# Mean squared error
cost = tf.reduce_sum(tf.square(linear_model - y)) / (2 * n_samples)

# Gradient Descent optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Defining variables initialization
init = tf.global_variables_initializer()

# Starting the session
with tf.Session() as sess:
    # Starting variables
    sess.run(init)

    # Model training
    for epoch in range(training_epochs):

        # Gradient Descent optimization
        sess.run(optimizer, feed_dict={X: train_X, y: train_y})

        # Display of each epoch
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, y: train_y})
            print("Epoch:{0:6} \t Cost (Error):{1:10.4} \t W:{2:6.4} \t b:{3:6.4}".format(epoch + 1, c, sess.run(W), sess.run(b)))
                                                                        
    # Printing the final parameters of the model
    print("\nOptimization Complete!")
    training_cost = sess.run(cost, feed_dict={X: train_X, y: train_y})
    print("Final Training Cost:", training_cost, " - W Final:", sess.run(W), " - b Final:", sess.run(b), '\n')

    # Viewing the result
    plt.plot(train_X, train_y, 'ro', label='Original Data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Regression Line')
    plt.legend()
    plt.show()

    # Testing the model
    testing_cost = sess.run(tf.reduce_sum(tf.square(linear_model - y)) / (2 * test_X.shape[0]),
                            feed_dict={X: test_X, y: test_y})

    print("Final Cost in Test:", testing_cost)
    print("Absolute Square Mean Difference:", abs(training_cost - testing_cost))

    # Test Display
    plt.plot(test_X, test_y, 'bo', label='Testing Data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Regression Line')
    plt.legend()
    plt.show()

sess.close()
