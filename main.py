"""
This is the main script for the Chatbot project.
It loads the trained model, data, and GUI components to create a chatbot interface.
"""

# FILEPATH: /c:/Users/danie/OneDrive/Escritorio/Archivos_personales/Projects-DS/Chatbot/Chatbot-project/main.py

# Import necessary libraries
from src import save_model, save_data, send, getResponse, predict_class
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

# if __name__ == "__main__":
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Set the paths for model, data, intents, words, and classes
    data_path = os.path.dirname(os.path.dirname(__file__))
    path_model = os.path.join(data_path, "data\\hatbot_model.keras")
    model = load_model(path_model)
    intents_path = os.path.join(data_path, "data\\intents.json")
    words_path = os.path.join(data_path, "data\\words.pkl")
    classes_path = os.path.join(data_path, "data\\classes.pkl")

    # Load the intents, words, and classes
    intents = json.loads(open(intents_path).read())
    words = pickle.load(open(words_path, 'rb'))
    classes = pickle.load(open(classes_path, 'rb'))

    # Create the GUI window
    root = Tk()
    root.title("Chatbot")
    root.geometry("400x500")
    root.resizable(width=FALSE, height=FALSE)

    # Create the chatbox
    ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)
    ChatBox.config(state=DISABLED)

    # Create the scrollbar
    scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
    ChatBox['yscrollcommand'] = scrollbar.set

    # Create the send button
    SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send",
                        width="12", height=5, bd=0, bg="#32de97",
                        activebackground="#3c9d9b", fg='#ffffff', command=send)

    # Create the entry box
    EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")

    # Place the scrollbar, chatbox, entry box, and send button in the GUI window
    scrollbar.place(x=376, y=6, height=386)
    ChatBox.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)

    # Run the GUI event loop
    root.mainloop()