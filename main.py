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

paragraphs = [p.get_text() for p in soup.find_all('p')]

clean_text = re.sub(r'\s+', ' ', ' '.join(paragraphs))
clean_text = re.sub(r'[^\w\s]', '', clean_text)

sentences = sent_tokenize(clean_text)

tokens = word_tokenize(clean_text.lower())

vectoriser = TfidfVectorizer()
sentence_vectors = vectoriser.fit_transform(sentences)

trainer = ListTrainer(chatbot)
trainer.train([
    "hello",
    "Welcome, what program error can I help you solve?",
])
trainer.train(paragraphs)

def answer_questions(query):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    input_ids = tokenizer.encode(query, return_tensors="pt")
    
    output = model.erate(input_ids, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response;
    

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    
    response = answer_questions(query)
    if response != None:
        print(response)