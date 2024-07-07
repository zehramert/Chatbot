#tokenizing, limitazing and removing stop words
import json
import random
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer= WordNetLemmatizer()



intents = json.loads(open('../intents.json').read())



words = []
tags = []
word_tag = []
ignoreLetters = ['.', '?', '!', ',']   #These will be neglected from the dataset

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern) #Tokenizes the pattern into individual words
        words.extend(wordList)
        word_tag.append((wordList, intent['tag'])) #A tupple contains the wordlist and corresponding tag
                                                 #This will be used for training example with the input (wordList) and the output (intent['tag'])
        if intent['tag'] not in tags:
            tags.append(intent['tag'])  #Collects all unique tags


words =[lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters] #Reduce words to their root
words = sorted(set(words))  #Set is used for avoid duplication


pickle.dump(words, open('words.pkt', 'wb'))
pickle.dump(tags, open('tags.ptk', 'wb'))

#Creating training dataset
train = []
outputEmpty = [0] * len(tags)

for sample in word_tag:
    bag = [] #it will use for determining the index of word
    wordPattern = sample[0]
    wordPattern = [lemmatizer.lemmatize(word.lower()) for word in wordPattern]
    for word in words:
        bag.append(1) if word in wordPattern else bag.append(0)

    outputRow = list(outputEmpty)  #Copy of outputEmpty has created
    outputRow[tags.index(sample[1])] = 1  #Finding the tags index
    train.append(bag + outputRow)  #contains two vectors: input pattern and one-hot encoded representation of output class

random.shuffle(train)
train = np.array(train)

#Splitting the train array to bags and output class
trainX = train[:,:len(words)] #keeps bag list
trainY = train[:,len(words):] #keeps output classes (tag)

#Creating the model
model = tf.keras.Sequential()

# Input layer with ReLU activation
model.add(tf.keras.layers.Dense(128, 'relu', input_shape=(len(trainX[0]),)))
model.add(tf.keras.layers.Dropout(0.5))

# Hidden layer with ReLU activation
model.add(tf.keras.layers.Dense(64,  'relu'))
model.add(tf.keras.layers.Dropout(0.5))


# Output layer with Softmax activation
model.add(tf.keras.layers.Dense(len(trainY[0]), 'softmax'))


# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Print model summary
model.summary()

hist = model.fit(np.array(trainX), np.array(trainY), epochs=600, batch_size=3, verbose=1)

model.save('chatbot2.h5', hist)


print("Training progress is completed")









