# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

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
def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(bbc.text,
                                                                                bbc.category,
                                                                                train_size=training_portion,
                                                                                shuffle=False
                                                                                )

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_tok
                          )
    tokenizer.fit_on_texts(training_sentences)

    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_sentences = pad_sequences(train_sequences,
                                   maxlen=max_length,
                                   truncating=trunc_type,
                                   padding=padding_type
                                   )

    test_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_sentences = pad_sequences(test_sequences,
                                  maxlen=max_length,
                                  truncating=trunc_type,
                                  padding=padding_type
                                  )

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc.category)

    training_labels = np.array(label_tokenizer.texts_to_sequences(training_labels))
    validation_labels = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.summary()

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy']
                  )

    model.fit(training_sentences,
              training_labels,
              epochs=50,
              validation_data=(validation_sentences, validation_labels),
              callbacks=[callbacks],
              verbose=2
              )
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
