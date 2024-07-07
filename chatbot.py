import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import datetime
import requests

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('../intents.json').read())

words = pickle.load(open('words.pkt', 'rb'))
tags = pickle.load(open('tags.ptk', 'rb'))

model = load_model('chatbot2.h5')

WEATHER_API_KEY = '4e49b17a80c9dddda24190e606e40f9e'
#A function for tokenizing a sentence into individual words and lemmitize
def tokenizing(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#A function that gets a list of lemmitized words and creates bag of words that represents the sentence
def bag_of_words(sentence):
    sentence_words = tokenizing(sentence)
    bag = [0]*len(words) #list that contains 0's for each word
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#Returns a list of intents with probabilities with descending order
def predictTag(sentence):
    bow = bag_of_words(sentence) #numerical vector that represents the absence of known words
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25 #probabilities below that will be eliminated
    results = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]  #i represents index, r represents prob.

    results.sort(key=lambda x: x[1], reverse=True)
    result_list=[]
    for r in results:
        result_list.append({'intent': tags[r[0]], 'probability': str(r[1])})

    return result_list

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    weather_data = response.json()

    if weather_data.get('main'):
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        return f"The current temperature in {city} is {temp}°C with {description}."
    else:
        return "I'm unable to retrieve the weather data right now."

def get_response(intents_list, intents_file):
    tag = intents_list[0]['intent']
    list_of_intents = intents_file['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            if tag == "date":
                response = response.replace("{date}", datetime.datetime.now().strftime("%Y-%m-%d"))
            elif tag == "time":
                response = response.replace("{time}", datetime.datetime.now().strftime("%H:%M:%S"))
            elif tag == "weather":
                city = input("Enter the city for weather informations: ")
                response = get_weather(city)
            break
    return response


print("Chatbot is running...")
running=True
while (running):
    message = input("You: ")
    if(message=="EXIT"):
        running=False
        print("Exiting...")
        break
    intents_list = predictTag(message)
    if intents_list:
        response = get_response(intents_list,intents)
    else:
        response = "I'm not sure how to respond to that."
    print ("Chatbot: " + response)













