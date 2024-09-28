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
trainer = ListTrainer(chatbot)
trainer.train([
    "hello",
    "Welcome, what program error can I help you solve?",
])

def answer_questions(query):
    
    heading = soup.find('h1').get_text()
    
    documents = [heading, query]
    
    vectorizer = TfidfVectorizer()
    tfid_matrix = vectorizer.fit_transform(documents)
    
    cos_simularity = cosine_similarity(tfid_matrix[0:1], tfid_matrix[1:2])
    
    if cos_simularity >= 0.2:
        
        verified_answer = soup.find('div', class_= 'js-accepted-answer-indicator flex--item fc-green-400 py6 mtn8')
        
        
        if verified_answer:
            container = verified_answer.find_parent('div', class_='answer js-answer accepted-answer js-accepted-answer')
            
            
        if container:
            
            for remove in container.select('.user-details, .share, .footer, .post-signature'):
                remove.decompose()
                
            removed_phrases = ["Share", "Improve this answer", "Follow", "Add a comment", "|"]
            
            answer = container.get_text(strip=True)
            
            for phrase in removed_phrases:
                answer = answer.replace(phrase, '')
            
            answer = re.sub(r'^\d+', '', answer)  
            answer = re.sub(r'\s{2,}', ' ', answer)
            
            return answer
                
        else:
            return "There are no verified answers but we can give you some suggestions made"
    

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    
    response = answer_questions(query)
    if response != None:
        print(response)