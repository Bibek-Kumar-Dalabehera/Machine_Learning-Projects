import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# # Define the CSS to set the background color to black
# background_color = """
# <style>
# body {
#     background-color: black;
#     color: white;  /* Change text color to white for better visibility */
# }
# </style>
# """

# # Inject the CSS into the Streamlit app
# st.markdown(background_color, unsafe_allow_html=True)


# Load the custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text) # TOKENIZER:It create a list by converting the words into lower case
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)# It create a list by removing special character after the tokkenising 
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            # It create a list by removing dtp words and punctuation markr after abobe operation
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))# steming operation

    return " ".join(y)

tfdt=pickle.load(open('Vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email Spam Classifier')

input_sms=st.text_area("Enter the message:")

if st.button('Predict'):

    #1.Preprocess
    transformed_sms=transform_text(input_sms)

    #2.Vectorize
    vector_input= tfdt.transform([transformed_sms])

    #3.Predict
    result=model.predict(vector_input)[0]

    #4.Display
    if result==1:
        st.header("Spam Message")
    else:
        st.header("Not Spam Message")