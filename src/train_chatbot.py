import json
import os
import pickle
import random

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer


def save_data() -> None:
    """
    This function reads the intents.json file, processes the data, and saves the processed data as pickle files.
    It performs the following steps:
    1. Reads the intents.json file.
    2. Tokenizes the patterns in the intents and stores them in a list.
    3. Lemmatizes the words and removes ignore letters.
    4. Sorts and removes duplicates from the words list.
    5. Sorts the classes list.
    6. Prints the number of documents, classes, and unique lemmatized words.
    7. Saves the words, classes, and documents lists as pickle files.
    """

    # Run this the first time: nltk.download('punkt')
    # Run this the fist time: nltk.download('wordnet')

    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get the path to the data directory
    data_path = os.path.dirname(os.path.dirname(__file__))

    # Construct the path to the intents.json file
    new_path = os.path.join(data_path, "data\\intents.json")

    # Read the intents.json file
    intents_file = open(new_path).read()

    # Parse the intents.json file into a Python dictionary
    intents = json.loads(intents_file)

    # Initialize lists to store words, classes, and documents
    words = []
    classes = []
    documents = []
    ignore_letters = ["!", "?", ",", "."]

    # Iterate over each intent in the intents dictionary
    for intents_dict in intents["intents"]:
        # Iterate over each pattern in the current intent
        for pattern in intents_dict["patterns"]:
            # Tokenize the pattern into a list of words
            word_list = nltk.word_tokenize(pattern)
            # Add the words to the words list
            words.extend(word_list)
            # Add the word list and the intent tag to the documents list
            documents.append((word_list, intents_dict["tag"]))
            # Add the intent tag to the classes list
            # if it's not already present
            if intents_dict["tag"] not in classes:
                classes.append(intents_dict["tag"])

    # Lemmatize the words and remove the ignore letters
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in ignore_letters
    ]
    # Sort and remove duplicates from the words list
    words = sorted(set(words))

    # Sort the classes list
    classes = sorted(set(classes))

    # Print the number of documents, classes, and unique lemmatized words
    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    # Construct the paths to save the words and classes lists as pickle files
    new_path_words = os.path.join(data_path, "data\\words.pkl")
    new_path_classes = os.path.join(data_path, "data\\classes.pkl")
    new_path_documents = os.path.join(data_path, "data\\documents.pkl")
    # Save the words and classes lists as pickle files
    pickle.dump(words, open(new_path_words, "wb"))
    pickle.dump(classes, open(new_path_classes, "wb"))
    pickle.dump(documents, open(new_path_documents, "wb"))


def save_model() -> None:
    """
    Save the trained chatbot model.

    This function loads the training data, creates a sequential model,
    compiles and trains the model, and saves the trained model to a file.

    Args:
        None

    Returns:
        None
    """
    lemmatizer = WordNetLemmatizer()
    data_path = os.path.dirname(os.path.dirname(__file__))
    new_path_words = os.path.join(data_path, "data\\words.pkl")
    new_path_classes = os.path.join(data_path, "data\\classes.pkl")
    new_path_documents = os.path.join(data_path, "data\\documents.pkl")

    words = pickle.load(open(new_path_words, "rb"))
    classes = pickle.load(open(new_path_classes, "rb"))
    documents = pickle.load(open(new_path_documents, "rb"))

    # Initialize an empty list to store the training data
    training_data = []

    # Create a list of zeros with the length of the classes list
    output_empty = [0] * len(classes)

    # Iterate over each document in the documents list
    for document in documents:
        # Initialize an empty list to store the bag of words
        bag = []

        # Get the word patterns from the current document
        word_patterns = document[0]

        # Lemmatize and lowercase the word patterns
        word_patterns = [
            lemmatizer.lemmatize(word.lower()) for word in word_patterns]

        # Create a bag of words representation for the current document
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        # Create the output row for the current document
        output_row = output_empty[:]
        output_row[classes.index(document[1])] = 1

        # Add the bag of words and output row to the training data
        training_data.append([bag, output_row])

    # Shuffle the training data
    random.shuffle(training_data)

    # Split the training data into input and output arrays
    training_data_x = [train[0] for train in training_data]
    training_data_y = [train[1] for train in training_data]

    # Convert the training data arrays to numpy arrays
    training_data_x = np.array(training_data_x)
    training_data_y = np.array(training_data_y)

    # Convert the training data arrays to lists
    train_x = list(training_data_x)
    train_y = list(training_data_y)
    print(train_x)
    print(len(train_x))
    print(train_y)
    print(len(train_y[0]))
    # Create a sequential model
    model = Sequential()

    # Add a dense layer with 128 units and ReLU activation function
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))

    # Add a dense layer with 64 units and ReLU activation function
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))

    # Add a dense layer with the number of units equal to the number of classes
    # and softmax activation function
    model.add(Dense(len(train_y[0]), activation="softmax"))

    # Create an SGD optimizer with specified parameters
    sgd = SGD(
        learning_rate=0.01,
        weight_decay=1e-6,
        momentum=0.9,
        nesterov=True)

    # Compile the model with categorical crossentropy loss function
    # and the SGD optimizer
    model.compile(
        loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # Train the model on the training data
    hist = model.fit(
        np.array(train_x),
        np.array(train_y),
        epochs=200,
        batch_size=10,
        verbose=1)

    # Construct the path to save the trained model
    new_path_model = os.path.join(data_path, "data\\hatbot_model.keras")

    # Save the trained model
    model.save(new_path_model, hist)

    # Print the summary of the model
    # print(model.summary())
