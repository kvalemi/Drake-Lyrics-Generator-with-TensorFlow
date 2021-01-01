## Project Description

In this project I aimed to build a model that could autonomously create Drake lyrics. I cleaned the lyrics of about 290 Drake songs, and parsed them into the correct format so that our model could be trained. I chose a Sequential Recurrent Neural Network (RNN) as my model of choice. I treated each lyrical line, from any given Drake song, as a standalone peice of data that my model could be trained on. After training the model, I then build a pipeline script that would take as input from the user one single word and would output a lyrical line that should theoretically begin with that word. The intuition here is that I want to capture the most probable lyrical line that Drake would say that starts with any arbritrary line. Here are some examples:

**Example 1:**

Input: 'Ice'

Output: 'Ice all i am i am i am i am i' 

**Example 2:**

Input: Hate

Output: 'Hate when i am i am i am i am i'

**Example 3:**

Input: 'clouds'

Output: 'Clouds on some dudes get it is there is there is'

* As we can see, there is a lot of self-referal in the form of 'i am'. The model appears to default to high occuring words like 'i', 'am', 'there', and 'is' which is interesting. Furthermore, if you would like to see the progress made on this project please first look at the Jupyter Notebook!

## How to Build the Project

1) Ensure the dependancies are correctly installed and built from source.

2) For training the model, I have provided the following two options in the script `Train_Model.py`:

- Training with Google Cloud TPUs: Change the training function call to `train_with_TPUs`.
- Training with Non TPU Architecture: Change the training function call to `train_without_TPUs`.

Also, this is a pretty heavy duty training process, so update the Epoch parameter of the model fit step according to the results you want to acheive and the time you have available for training.

3) Once the model has finished training launch the model interface script `Drake_Lyrics_Generator.py`, and follow the instructions on screen.


## Credits:

Data obtainde from: https://www.kaggle.com/juicobowley/drake-lyrics
