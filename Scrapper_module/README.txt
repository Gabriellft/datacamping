Liste_readme
---------------------
This is the readme for the scrapping module. You can find the readme for the streamlit part in the main folder.

CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Installation
 * Configuration
 * Maintainers


INTRODUCTION
------------
Module to scrape tripadvisor and google reviews connected to an API.

 * For a full description of the module, check the project pdf:
	20221012_Gpe_7_DE2_Laffitte_Marchandise


 
REQUIREMENTS
------------
 *torch==1.12.0
 *transformers==4.20.1
 *pandas==1.4.2
 *deep-translator==1.8.3
 *streamlit
 *bertopic	
 *sklearn
 *nltk
 *wordcloud
 *joblib==1.1.0
 *bs4
 *requestium
 *selenium
 *keybert
 *gensim
 *fastapi



INSTALLATION
------------
 
 * pip install -r requirement.txt

 * pip install --upgrade joblib==1.1.0

 

CONFIGURATION
-------------

 * To run Webdriver, you need to install chromedriver for your current version of chrome https://chromedriver.chromium.org/downloads and to place it on your google chrome folder.

 * To run the anti-scrapping extension, you need to put the extension_1_6_8_0.crx in the same folder than the chromedriver

 * uvicorn app:app --reload

MAINTAINERS
-----------

Current maintainers:
 * Briac Marchandise
 * Gabriel Laffitte

