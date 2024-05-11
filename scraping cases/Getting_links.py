import requests
from bs4 import BeautifulSoup
import time
import json

URL = 'https://indiankanoon.org/browse/'  # URL of the site we are scraping
base = 'https://indiankanoon.org'
no_of_pages = list(range(100))

# Send a request to the URL and parse the HTML content
with requests.Session() as session:
    page = session.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

# Find all <td> elements which contain the links to different courts
results = soup.find_all('td')

# Dictionary to store the links for each court
links = {}

for result in results:
    # Extract the court name and URL
    court_link = result.find('a')
    # print(court_link)
    if court_link:
        court_name = court_link.text
        court_url = base + court_link['href']
        # print(court_url)
        # Initialize the list of links for the current court
        links[court_name] = []

        # Send a request to the court's URL and parse the HTML content
        with requests.Session() as session:
            court_page = session.get(court_url)
        court_soup = BeautifulSoup(court_page.content, 'html.parser')
        # print(court_soup)
        # Find all <a> elements with class 'result_url' which contain the links to judgments
        judgment_links = court_soup.find_all(class_='browselist')
        # print(judgment_links)
        for judgment_link in judgment_links:
            # Append the absolute URL to the list of links for the current court
            links[court_name].append(base + judgment_link.find('a')['href'])
        # print(links[court_name])
# Filter the links to include only judgments from 1947 to 2020
print(links)
# filtered_links = {}
# for court_name, court_links in links.items():
#     filtered_links[court_name] = [link for link in court_links if '1980' <= link <= '2024']
#     print(filtered_links[court_name])
# print(filtered_links)
# Save the filtered links in a JSON file
with open("links_Supreme_Court.json", "w") as outfile:
    json.dump(links, outfile, indent=4)

# import requests
# import re
# from bs4 import BeautifulSoup
# import time
# import json

# URL = 'https://indiankanoon.org/browse/' # url of site from where we are scraping
# with requests.Session() as session:
#     page = session.get(URL)
# soup = BeautifulSoup(page.content, 'html.parser')
# print(soup)
# results = soup.find_all('td')
# # results = list(results[0:1])
# print(results)
# links = {}
# no_of_pages = list(range(100))
# base = 'https://indiankanoon.org'

# for link in results: # loop for multiple courts, if scraping data from multiple courts
#     linkd = link.find('a')['href']
#     court_name = link.find('a').text
#     URL = base+linkd
#     page = requests.get(URL)
#     soup = BeautifulSoup(page.content, 'html.parser')
#     result_new = soup.find_all(class_ = 'browselist')
#     links[court_name] = []
#     for link_new in result_new: # loop for every year
#         print((link_new.find('a').text) + " Year Started .....\n")
#         if((int)(link_new.find('a').text) < 1947 or (int)(link_new.find('a').text) > 2020):
#             continue
#         URL = base + link_new.find('a')['href']
#         page = requests.get(URL)
#         soup = BeautifulSoup(page.content, 'html.parser')
#         result_new2 = soup.find_all(class_ = 'browselist')
#         for link_new2 in result_new2: # loop for every month
#             for page_in in no_of_pages: # loop for every page
#               time.sleep(1)
#               URL = base + link_new2.find('a')['href']
#               URL = URL + '&pagenum={}'.format(page_in)
#               page = requests.get(URL)
#               soup = BeautifulSoup(page.content, 'html.parser')
#               result_new3 = soup.find_all(class_ = 'result_url')
#               if(len(result_new3) == 0):
#                 break
#               for link_new3 in result_new3: # finally appending the url in the list
#                 URL = base + link_new3['href']
#                 links[court_name].append(URL)
#         print("Current Year Completed\n")


# valid_court_name = ['Supreme Court of India']

# final_list = {}
# for court_name in valid_court_name:
#   final_list[court_name] = links[court_name]


# json_object = json.dumps(final_list, indent = 4) 

# # saving the dictionary with links in a json file
# with open("links_Supreme_Court.json", "w") as outfile:  
#     outfile.write(json_object)

