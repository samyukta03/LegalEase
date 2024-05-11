# LegalEase - _Legal Assistance for all_
<br/>
**Problem Statement - Chatbot - Law details, Relevant Supreme Court case Details & Legal Advice, Multi-Lingual and Voice Support**
<br/>

![legalease_main](https://github.com/samyukta03/LegalEaseFork/blob/main/forgithub.png)

<br/>**LegalEase**, is a chatbot that provides factual and appropriate information to legal queries from central acts. 
<br/>
<br/> The chatbot integrates key features to enhance accessibility and legal awareness. It offers multilingual support, automatic language detection, translation capabilities, and speech recognition functionalities, ensuring inclusivity for users of various language proficiency. Through natural language processing, the chatbot analyzes user scenarios, matches them with relevant laws, regulations, similar cases, and provides valuable legal insights.

 Our proposed legal assistant chatbot project is a direct response to the urgent need for improved legal accessibility, offering a holistic solution. Utilizing state-of-the-art technologies like natural language processing, machine learning, and web scraping, the project aims to deliver personalized legal guidance, multilingual support, and automatic speech recognition. This initiative seeks to empower individuals to make informed decisions regarding their legal matters, thereby advancing fairness, equality, and justice for all.

The prime objective is to structure an intelligent chatbot, which is not only data-driven but also incorporates machine learning and deep learning algorithms to map input sequences to output sequences.


<br/>LegalEase responds to both voice and text along with multi linguality for increased accessibility.


# Methodology
This project is implemented using a combination of machine learning algorithms like the Siamese Neural Network model for finding the most similar laws that are relevant and applicable to a scenario posted by a user from the dataset.

To provide the user with more guidance, supreme court cases that happened across the years are scraped and reformatted to a dataset are stored, and using a graph-based similarity ranking approach the most relevant and similar cases to the scenario posted by the user are chosen among the cases and their key points and outcome are presented to the user for them to get more idea of how to proceed further.

In addition to the above, using a HyBrid BERT-BiLSTM model predicts the likelihood of favorability to a party in a case based on the outcomes of similar cases scraped.



## UI Developed:
![UI](https://github.com/samyukta03/LegalEaseFork/blob/main/UI.png)
## Deployment

To deploy this project run

```bash
  npm start
```

After cloning the project steps to run:
1. Create a Python 3.8 environment 
 ```python
  conda create --name py8
```
2. Go to the project folder, Activate the environment: conda activate envt_name
 ```python
  conda activate py8
```
3. Go to backend->app.py file - 
 ```python
  python app.py
```
4. For the frontend, open the legalease folder in a node cmd and run, (if u update anything, run cmd again for the update to be rendered)  
 ```bash
  npm run build
```

## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```
