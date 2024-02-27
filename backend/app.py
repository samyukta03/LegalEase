from email import header
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import text
import pandas as pd
import difflib
import model
import nltk
from langdetect import detect
from deep_translator import GoogleTranslator
from googletrans import Translator

TEMPLATES_AUTO_RELOAD = True
use_reloader=True
app = Flask(__name__,static_folder="F:/Project-Legal.ly/build",static_url_path='/')
app.debug = True
app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config.update(
    TEMPLATES_AUTO_RELOAD=True
)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

data = pd.read_csv('A2015-22.csv')

topics = list(data['title'])
text = list(data['text'])

greetings = ['Hi', 'Hello', "Hey There", 'hi']

translator = GoogleTranslator(source='auto', target='en')

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route("/api", methods=['POST'])
@cross_origin()
def get_bot_response():
    data = request.get_json()  # Extract the JSON data from the request body
    query = data.get('msg', '')  # Get the value of 'msg' parameter, defaulting to an empty string if not found

    cleaned = " ".join([word for word in query.split()
                       if word not in nltk.corpus.stopwords.words('english')])

    # if difflib.get_close_matches(query, greetings):
    #     return "Hi There! Welcome to Legal.ly\nPlease type 'topics' to get a list of the topics I have knowledge on."

    # if query.lower().strip() in ['topics', 'topic']:
    #     response = 'You can ask me anything about the following topics:\n' + \
    #         " | ".join(topics)
    #     return response

    input_lang = detect(query)
    if input_lang is not None:
        if input_lang == 'ta':
            query_english = translator.translate(query)
            print(query_english)
        else:
            query_english = query
        
    translator1 = GoogleTranslator(source='auto', target='ta')
    if input_lang == 'ta':
        ans_temp="Hi There! Welcome! Ask me your legal queries"
        ans_translated = translator1.translate(ans_temp)
        return ans_translated
    else:
        ans_temp="Hi There! Welcome to LegalEase! You can ask your legal queries"
        return ans_temp
        # match = difflib.get_close_matches(query_english, topics, n=1)
        # if match:
        #     match = match[0]
        #     print(query_english, " fetching data")
        #     ans = model.get_answer(query_english, text[topics.index(match)])
        #     translator1 = GoogleTranslator(source='auto', target='ta')
        #     if input_lang == 'ta':
        #         topic_match=text[topics.index(match)]
        #         ans_temp = "I believe the answer to your query is:" +ans+" . \n For more context, please refer to the following text:  "+topic_match
        #         ans_translated = translator1.translate(ans_temp)
        #         print("in tamil",ans)
        #         return ans_translated
        #     else:
        #         topic_match=text[topics.index(match)]
        #         ans_temp = "I believe the answer to your query is:" +ans+" . \n For more context, please refer to the following text:  "+topic_match
        #         return ans_temp
        
    
    # return "Sorry, I didn't understand that."

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)

# # Makeshift script for chatbot backend as the Model couldn't train in time :')
# # Please read the README file to check our actual approach towards the problem

# from email import header
# from flask import Flask, request, render_template
# from flask_cors import CORS, cross_origin
# from matplotlib.pyplot import text
# import pandas as pd
# import difflib
# import model
# import nltk

# app = Flask(__name__,static_folder="F:/Project-Legal.ly/build",static_url_path='/')
# app.config["TEMPLATES_AUTO_RELOAD"] = True
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


# data = pd.read_csv('A2015-22.csv')

# topics = list(data['title'])
# text = list(data['text'])


# greetings = ['Hi', 'Hello', "Hey There", 'hi']

# # @app.route("/", methods=['GET'])
# # def index():
# #     return "Welcome to Legal.ly Bot API"
# @app.errorhandler(404)
# def not_found(e):
#     return app.send_static_file('index.html')


# @app.route('/')
# def index():
#     return app.send_static_file('index.html')

# @app.route("/api", methods=['POST'])
# @cross_origin()
# def get_bot_response():
#     print("heyyyyloo")
#     data = request.get_json()  # Extract the JSON data from the request body
#     query = data.get('msg', '')  # Get the value of 'msg' parameter, defaulting to an empty string if not found
#     print(query)
#     # query = request.form['msg']

#     cleaned = " ".join([word for word in query.split()
#                        if word not in nltk.corpus.stopwords.words('english')])

#     if difflib.get_close_matches(query, greetings):
#         return "Hi There! Welcome to Legal.ly\nPlease type 'topics' to get a list of the topics I have knowledge on."

#     if query.lower().strip() in ['topics', 'topic']:
#         response = 'You can ask me anything about the following topics:\n' + \
#             " | ".join(topics)
#         return response

#     match = difflib.get_close_matches(cleaned, topics, n=1)
#     if match:
#         match = match[0]
#         ans = model.get_answer(query, text[topics.index(match)])
#         return "I believe the answer to your query is: {}. \n For more context, please refer to the following text: {}".format(ans, text[topics.index(match)])
#     else:
#         return "Sorry, I didn't understand that."


# if __name__ == "__main__":
#     app.run("0.0.0.0", debug=True)
