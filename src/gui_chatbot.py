import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np

from keras.models import load_model
import json
import random
import os
import tkinter
from tkinter import *

lemmatizer = WordNetLemmatizer()

data_path = os.path.dirname(os.path.dirname(__file__))
path_model = os.path.join(data_path, "data\\hatbot_model.keras")
model = load_model(path_model)
intents_path = os.path.join(data_path, "data\\intents.json")
words_path = os.path.join(data_path, "data\\words.pkl")
classes_path = os.path.join(data_path, "data\\classes.pkl")
intents = json.loads(open(intents_path).read())
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))

# print(model.input_shape)
# initial_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#       0, 0, 0, 0, 0,
#       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
#       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
#       0, 0, 1, 0, 0, 1, 0, 0])
# print(initial_data.shape)
# print(model.predict(np.array([initial_data])))


def clean_up_sentence(sentence: str) -> list:
    """
    Cleans up a sentence by tokenizing it into words and lemmatizing each word.

    Args:
        sentence (str): The sentence to be cleaned up.

    Returns:
        list: A list of cleaned up words.

    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str,
                 words: pickle,
                 show_details: bool = True) -> np.array:
    """
    Converts a sentence into a bag of words representation
    using a given set of words.

    Args:
        sentence (str): The input sentence to convert.
        words (pickle): The set of words to use
        for the bag of words representation.
        show_details (bool, optional): Whether to print details
        about the words found in the bag. Defaults to True.

    Returns:
        np.array: The bag of words representation of the sentence.
    """
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {word}")
    return np.array(bag)


def predict_class(sentence: str, ERROR_THRESHOLD:  float = 0.05) -> list:
    """
    Predicts the class of a given sentence using a trained model.

    Args:
        sentence (str): The input sentence to classify.
        ERROR_THRESHOLD (float, optional): The threshold value
        for classifying the sentence. Defaults to 0.05.

    Returns:
        list: A list of dictionaries containing
        the predicted intent and its probability.
    """
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json) -> str:
    """
    Retrieves a response based on the predicted intent
    from the chatbot's intents JSON file.
    Args:
        ints (list): List of predicted intents and their probabilities.
        intents_json (dict): JSON object containing the chatbot's intents.
    Returns:
        str: The selected response from the intents JSON file.
    """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def send() -> None:
    """
    Sends the user's message, processes it, and displays the response in the chat box.
    """
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state = NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#442265", font=("Verdana", 12))

        ints = predict_class(msg)
        res = getResponse(ints, intents)

        ChatBox.insert(END, "Bot: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)






