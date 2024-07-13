oading and Preprocessing:

Load the spam.csv file.
Preprocess the data by dropping unnecessary columns and renaming the relevant ones.
Directory Creation:

Create directories to store classified spam and ham messages.
Message Processing:

Define the process_message function for tokenizing, stemming, and removing stopwords.
Word Counting:

Define functions to count words in spam and ham messages.
Spam Detection:

Use the spam function to classify messages.
Message Classification and Storage:

Loop through each message, classify it, and store it in the appropriate directory based on the classification result.
This approach integrates the original dataset, processes each message, classifies it using the spam detection model, and stores the messages in separate folders based on their classification.
