# Overview
Simple chatbot in python with deep learning

# How to run locally
'pip install --user -U nltk'

'pip install --user -U numpy'

'python3 chatbot.py'


# Steps:

- Prepare a dictionary  of intents (patterns, tags, responses)
- Create a vocabulary of all of the words used in the patterns, associated tags and classes
- Text Cleaning and Pre-processing of vocabulary: Tokenization (nltk) & Lemmatization (nltk.stem) & Noise Removal (punctuation)
- Data processing: convert the data to numerical values using bag-of-words technique (Weighted Words)
- Train the deep learning model: Sequential Model for Text Classification
- Use the model to predict class of user input features (pre-processed)



# Details

## Data preparation

The dictionary = input data

used a dictionary to represent an intents JSON file
intents(json)= tags -> patterns(the queries posed by the user) -> responses
 Prepare tags in advance 

### Create a vocabulary 

1. Loop through all the intents, tokenize each pattern and append tokens to words, the patterns and the associated tag to their associated list
2. add the tag to the classes if it's not there already 


* words: vocabulary of all of the words used in the patterns
* doc_X: list of all the patterns within the intents file
* doc_y: list of all the associated tags to go with each pattern in the intents file
* classes: the tags of each intent (unique)

## Lemmatize the vocab

lemmatize all the words in the vocab and convert them to lowercase if the words don't appear in punctuation
sorting the vocab and classes in alphabetical order and ensure no duplicates occur


* Words : lemmatized and sorted in alphabetical order
* Classes: sorted


 train deep learning model with training data

Neural Networks expect numerical values (we have words) Specifically, vectors of numbers.

-> process data (convert our data to numerical values):  bag of words technique == cleaning text (reduce noise by removing unnecessary data) and representing text as numerical values

the deep learning model
Sequential


## Data processing: Bag-of-Words Model

A bag-of-words is a representation of text that describes the occurrence of words within a document.


* training: list for training data: for each pattern(Bow,output_row)
* Bow: associated word count of all tokens of a pattern
* output_row: associated classes/tags of all tokens of a pattern 

reorganize the order of training array items
split the features and target labels (using numpy)

* train_X: np.array(list(training[:, 0]))  all rows of column 0
* train_y: np.array(list(training[:, 1])) all rows of column 1

-> train_X & train_y are used to train the deep learning model 


# Example of execution

pattern1: Hello there !

pattern2: old

words: ['hello', 'old', 'there']

doc_X ['Hello there', 'old'] = patterns

doc_y ['greeting', 'age'] = associated tags

classes: ['age', 'greeting']

--------- For pattern1

bow1: [1, 0, 1] # = [hello not in pattern1 so 1, old in pattern1 so 0, there in pattern1 so 1]

output_row1: [0, 1] # =1 when class = pattern


--------- For pattern2

bow2: [0, 1, 0]
output_row2: [1, 0]

---------

training: [ [ [1, 0, 1], [0, 1] ], [ [0, 1, 0], [1, 0] ] ] # = [bow, output_row] for each pattern

---------

training after shuffle: [
						 [list([0, 1, 0]) list([1, 0])]
						 [list([1, 0, 1]) list([0, 1])]
						 								]

---------

train_X: [[0 1 0]
 		  [1 0 1]]  # = np.array(list(training[:, 0])) all rows of column 0 --> features

train_y: [[1 0]
 		  [0 1]]	# = np.array(list(training[:, 1])) all rows of column 1 --> tags




