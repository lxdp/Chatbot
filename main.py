from chatterbot import ChatBot
import nltk
from chatterbot.trainers import ListTrainer
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel


nltk.download('punkt_tab')

chatbot = ChatBot("Chatbot")

url = "https://stackoverflow.com/questions/1735109/setting-python-interpreter-in-eclipse-mac"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# clean_text = re.sub(r'\s+', ' ', ' '.join(paragraphs))
# clean_text = re.sub(r'[^\w\s]', '', clean_text)

# sentences = sent_tokenize(clean_text)
#
# tokens = word_tokenize(clean_text.lower())

trainer = ListTrainer(chatbot)
trainer.train([
    "hello",
    "Welcome, what program error can I help you solve?",
])

def answer_questions(query):
    
    heading = soup.find('h1').get_text()
    
    print(heading)
    
    documents = [heading, query]
    
    vectorizer = TfidfVectorizer()
    tfid_matrix = vectorizer.fit_transform(documents)
    
    cos_simularity = cosine_similarity(tfid_matrix[0:1], tfid_matrix[1:2])
    
    if cos_simularity >= 0.2:
        div = soup.find('div', class_='s-prose js-post-body')
        
        if div:
            text = div.get_text(strip = True)
    
    if text:
        return text
    else:
        return "We could not find what you are looking for"
    

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    
    response = answer_questions(query)
    if response != None:
        print(response)