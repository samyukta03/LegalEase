import os
import urllib
import urllib.request
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import progressbar
base = 'https://indiankanoon.org'
# Path of the JSON file containing links
path_of_Supreme_court_json_file = "links_Supreme_Court.json"

# Load the JSON file
with open(path_of_Supreme_court_json_file, 'r') as file:
    links_json = json.load(file)

# Specify the category
category = 'Supreme Court of India'
print("Started file ... {0} with docs = {1}\n\n\n".format(category, len(links_json[category])))
category_links = links_json[category]
year = path_of_Supreme_court_json_file[len(path_of_Supreme_court_json_file) - 9: len(path_of_Supreme_court_json_file) - 5]  # Extract the year
ndocs = 0
os.makedirs('Yearwise_data/' + year, exist_ok=True)

while len(category_links) > 0:
    time.sleep(2)
    links_done_in_this_loop = []
    for i in progressbar.progressbar(range(len(category_links))):
        BASE_URL = category_links[i]
        try:
            req = urllib.request.Request(BASE_URL, headers={'User-Agent': 'Mozilla/5.0'})
            html = urllib.request.urlopen(req).read()
            print(html)

            if html is None:
                print("NONE")
                continue  # Skip to the next link if html is None
        except urllib.error.HTTPError:
            print("Ocuured at doc", ndocs)
            time.sleep(2)
        else:
            soup = BeautifulSoup(html, "lxml")
            data_html = soup.find("div", attrs={"class": "browselist"})
            # print(data_html)
            doc_links = data_html.find_all("a")
            print(doc_links)
            for doc_link in doc_links:
                doc_url = doc_link.get("href")
                doc_url= base+doc_url
                print(doc_url)
                doc_url = doc_url.replace(" ", "%20")
                req_doc = urllib.request.Request(doc_url, headers={'User-Agent': 'Mozilla/5.0'})
                try:
                 html_doc = urllib.request.urlopen(req_doc).read()
                except:
                 time.sleep(10)
                print(html_doc)
                if html_doc is None:
                    print("NONE")
                    continue  # Skip to the next document if html is None
                soup_doc = BeautifulSoup(html_doc, "lxml")
                text = soup_doc.find("div", attrs={"class": "result_title"})
                # data1= soup_doc.find("div", attrs={"class": "browselist"})
                # print(text)
                # print(data1)
                case_links=text.find_all("a")
                print(case_links,"whoo")
                for case_link in case_links:
                    case_url=case_link.get("href")
                    case_url=base+case_url
                    print(case_url)
                    case_url = case_url.replace(" ", "%20")
                    req_case= urllib.request.Request(case_url, headers={'User-Agent': 'Mozilla/5.0'})
                    try:
                     case_html=urllib.request.urlopen(req_case).read()
                    except:
                     time.sleep(10)
                    print(case_html)
                    soup_j=BeautifulSoup(case_html, "lxml")
                    print(soup_j.prettify()) 
                    res=soup_j.find("div", attrs={"class": "judgments"})
                    print(res)
                    data_res=res.find_all("p")
                    text_data = ""
                    for tag in data_res:
                          text_data += tag.get_text() + "\n"
                path = 'Yearwise_data/' + year + '/' + year + '_' + str(ndocs)
                with open(path, "w+") as f:
                    f.write(text_data)
                ndocs += 1
            links_done_in_this_loop.append(BASE_URL)
            if ndocs % 100 == 0:
                time.sleep(2)
    for link in links_done_in_this_loop:
        category_links.remove(link)
    print("Docs that were downloaded: {0}\n\n\n".format(ndocs))
# #!/usr/bin/env python
# # coding: utf-8

# # In[ ]:


# import os
# import urllib
# import urllib.request
# import json
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# import progressbar


# # In[ ]:


# path_of_Supreme_court_json_file = "links_Supreme_Court.json" #path of Links of Supreme Court json file


# # In[ ]:


# links_json = json.loads(path_of_Supreme_court_json_file)
# category = 'Supreme Court of India'
# print("Started file ... {0} with docs = {1}\n\n\n".format(json_file,len(links_json[category])))
# category_links = links_json[category]
# year = json_file[len(json_file)-9:len(json_file)-5] # Obtain the name of year like '1947' or '2002'
# ndocs = 0
# os.mkdir('Yearwise_data/'+ year)

# while(len(category_links)>0):
#   time.sleep(2) 
#   links_done_in_this_loop = []
#   for i in progressbar.progressbar(range(len(category_links))):
#       BASE_URL = category_links[i]
#       try:
#           html = urllib.request.urlopen(BASE_URL).read()
#       except urllib.error.HTTPError:
#           print("Ocuured at doc",ndocs)
#           time.sleep(2)

#       else:
#           soup = BeautifulSoup(html, "lxml")
#           data_html = soup.find("div", attrs={"class": "judgments"})
#           text = data_html.get_text()
#           path =  'Yearwise_data/' + year + '/' + year +'_' +str(ndocs) # Path where the yearwise data file will save
#           f= open(path,"w+")
#           f.write(text)
#           ndocs = ndocs+1
#           links_done_in_this_loop.append(BASE_URL)
#           if(ndocs%100==0):
#               time.sleep(2)          # If number of documents downloaded is 100 then go for sleep for 2 sec
#   for link in links_done_in_this_loop:
#       category_links.remove(link)
#   print("Docs that were downloaded: {0}\n\n\n".format(ndocs))

