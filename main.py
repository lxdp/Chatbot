from chatterbot import ChatBot
import nltk
from chatterbot.trainers import ListTrainer
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download('punkt_tab')

chatbot = ChatBot("Chatbot")

url = "https://stackoverflow.com/questions/1735109/setting-python-interpreter-in-eclipse-mac"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

paragraphs = [p.get_text() for p in soup.find_all('p')]

clean_text = re.sub(r'\s+', ' ', ' '.join(paragraphs))
clean_text = re.sub(r'[^\w\s]', '', clean_text)

tokens = word_tokenize(clean_text.lower())

stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

trainer = ListTrainer(chatbot)
trainer.train([
    "hello",
    "Welcome, what program error can I help you solve?",
])

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        print(f"{chatbot.get_response(query)}")