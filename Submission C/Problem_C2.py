# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Halts the training when the loss falls below 0.4

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    # Check the loss
    if(logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91):

      # Stop if threshold is met
      print("\nAccuracy is above 0.91 so cancelling training!")
      self.model.stop_training = True
# Instantiate class
callbacks = myCallback()
def solution_C2():
    mnist = tf.keras.datasets.mnist

    # Discard test set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # DEFINE YOUR MODEL HERE
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28,1)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        # Compile the model
    model.compile(optimizer=tf.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Train the model with a callback
    model.fit(x_train, y_train, epochs=10,
              validation_data=(x_test, y_test),
              callbacks=[callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
