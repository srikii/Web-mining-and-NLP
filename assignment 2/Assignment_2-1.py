#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Web Scraping

# ## Q1. Scrape Book Catalog 
# - Scape content of http://books.toscrape.com 
# - Write a function getData() to scrape **title** (see (1) in Figure), **rating** (see (2) in Figure), **price** (see (3) in Figure) of all books (i.e. 20 books) listed in the page.
#   * For example, the figure shows one book and the corresponding html code. You need to scrape the highlighted content. 
#   * For star ratings, you can simply scrape One, Two, Three, ... 
# - The output is a list of 20 tuples, e.g. [('A Light in the ...','Three','£51.77'), ...] 
#     <img src='assign3_q1.png' width='80%'>
# 

# ## Q2. Data Analysis 
# - Create a function preprocess_data which 
#   * takes the list of tuples from Q1 as an input
#   * converts the price strings to numbers 
#   * calculates the average price of books by ratings 
#   * plots a bar chart to show the average price by ratings. 

# ### Q3 (Bonus) Expand your solution to Q1 to scrape the full details of all books on http://books.toscrape.com
# - Write a function getFullData() to do the following: 
#    * Besides scraping title, rating, and price of each book as stated in Q1, also scrape the **full title** (see (4) in Figure), **description** (see (5) in Figure), and **category** (see (6) in Figure) in each individual book page. 
#      * An example individual book page is shown in the figure below.
#        <img src='assign3_q3a.png' width='60%'>
#    
#    * Scape all book listing pages following the "next" link at the bottom. The figure below gives an screenshot of the "next" link and its corresponding html code. 
#    * <b>Do not hardcode page URLs </b>(except http://books.toscrape.com) in your code. 
#       <img src='assign3_q3.png' width='80%'>
#    * The output is a list containing 1000 tuples, 
#      - e.g. [('A Light in the ...','Three','£51.77', 'A Light in the Attic', "It's hard to imagine a world without A Light in the Attic. This now-classic collection ...",'Poetry'), ...]
#     

# In[52]:


import requests
from bs4 import BeautifulSoup  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(data):
    
    # add your code
    one=[]
    two=[]
    three=[]
    four=[]
    five=[]
    for i in data:
        p1=i[2]
        p2=float(p1[1:])
        if i[1].lower()=='one':
            one.append(p2)
        elif i[1].lower()=='two':
            two.append(p2)
        elif i[1].lower()=='three':
            three.append(p2)
        elif i[1].lower()=='four':
            four.append(p2)
        elif i[1].lower()=='five':
            five.append(p2)

    avg1=sum(one)/len(one)
    avg2=sum(two)/len(two)
    avg3=sum(three)/len(three)
    avg4=sum(four)/len(four)
    avg5=sum(five)/len(five)
    
    plt.bar("one",avg1,color='blue')
    plt.bar("two",avg2,color='blue')
    plt.bar("three",avg3,color='blue')
    plt.bar("four",avg4,color='blue')
    plt.bar("five",avg5,color='blue')
    plt.xlabel('ratings in stars')
    plt.ylabel('average price')
    plt.title('average price by rating')
    plt.show()

    
def getData():
    
    
    data=[]  # variable to hold all reviews
    
    # add your code
    
    page = requests.get("http://books.toscrape.com")
    soup = BeautifulSoup(page.content, 'html.parser')

    for i in soup.findAll("article", class_ = "product_pod"):
        star=i.p.get('class')[1]
        title=i.h3.a.get_text()
        #title=i.h3.a.get('title')
        price=i.h3.next_sibling.next_sibling.p.get_text()
        
        data.append((title,star,price))
        
    return data

def getFullData():
    
    
    data=[]  # variable to hold all book data
    
    description=[]
    title=[]
    category=[]
    rating=[]
    price=[]
    page = requests.get("http://books.toscrape.com")
    soup = BeautifulSoup(page.content, 'html.parser')

    for i in soup.findAll("article", class_ = "product_pod"):
        
        star=i.p.get('class')[1]
        rating.append(star)

        cash=i.h3.next_sibling.next_sibling.p.get_text()
        price.append(cash)
        
        page="http://books.toscrape.com/"
        book=i.a.get('href')
        booklink=page+book
        pages = requests.get(booklink)
        soups = BeautifulSoup(pages.content, 'html.parser')      
        
        for i in soups.findAll('ul', class_="breadcrumb"):
            j=i.select('ul > li')[2].get_text()
            category.append(j.strip())
            
        for i in soups.findAll('div', class_="col-sm-6 product_main"):
            j=i.h1.get_text().strip()
            title.append(j)
            
        desc=[]
        for i in soups.findAll('p'):
            desc.append(i)
        description.append(desc[3].get_text())
    x=0
    for i in range(len(price)):
        
        data.append((title[x], rating[x], price[x], description[x],category[x]))
        x+=1


    
    # add your code
            
    return data


# In[53]:


if __name__ == "__main__":  
    
    # Test Q1
    data=getData()
    print(20,"\ttitle\t\trating\t\tprice")
    t=0
    for i in data:
        print(t,"  ",i[0],"   ",i[1],"\t",i[2])
        t+=1
    # Test Q2
    preprocess_data(data)
    
    # Test Q3
    data=getFullData()
    print(data)
    print(len(data))
    
    # randomly select one book
    #print(data[899])


# In[ ]:





# In[6]:





# In[ ]:




