import json
import tensorflow as tf
from keras.models import load_model
import pickle


# load the trained model
model = load_model('chat_model')

# load the training data
with open('intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

# encode the training labels
with open('label_encoder.pickle', 'rb') as ecn_file:
    lbl_encoder = pickle.load(ecn_file)

training_labels_encoded = lbl_encoder.transform(training_labels)

# tokenize and pad the training sentences
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = tf.keras.utils.pad_sequences(sequences, truncating='post', maxlen=20)

# evaluate the model
loss, accuracy = model.evaluate(padded_sequences, training_labels_encoded)

# print the accuracy
print('Accuracy:', accuracy)

# evaluate the model
loss, accuracy = model.evaluate(padded_sequences, training_labels_encoded)

# print the accuracy
print('Accuracy:', accuracy)
