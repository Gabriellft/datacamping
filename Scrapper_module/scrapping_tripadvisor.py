from selenium import webdriver
import time
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time



def htmlstring_df(htmlstring):
    soup = BeautifulSoup(htmlstring, 'html.parser')
    Liste_Reviews = []
    compteur = 0
    textingfi = ''
    for i in range(len(soup.body.find_all(class_="reviewSelector"))):

        ii = soup.body.find_all(class_="reviewSelector")[i].text
        oo = soup.body.find_all(class_="noQuotes")[i].text



        if 'Date of visit' in ii:
            compteur += 1
            indexpremier = ii.index(oo)
            indexfinito = ii.index('Date of visit')
            indexfin = indexpremier + len(oo)
            textingfi = ii[indexfin:indexfinito]
            if textingfi[-4:] == 'More':
                Liste_Reviews.append(textingfi[:-4])
            else:
                if textingfi[-9:] == 'Show less':
                    Liste_Reviews.append(textingfi[:-9])

                else:
                    Liste_Reviews.append(textingfi)

            # print('----')
        # except ValueError:
        else:
            indexpremier = ii.index(oo)

            indexfin = indexpremier + len(oo)
            textingfi = ii[indexfin:]
            if textingfi[-4:] == 'More':
                Liste_Reviews.append(textingfi[:-4])
            else:
                Liste_Reviews.append(textingfi)

            # print('----')

    Liste_dates = []
    Liste_nom = []
    for date in soup.body.find_all(class_="prw_rup prw_reviews_stay_date_hsx"):
        texting = date.text[15:]
        Liste_dates.append(texting)

    for nom_reviewer in soup.body.find_all(class_="info_text pointer_cursor"):
        Liste_nom.append(nom_reviewer.text)

    # LISTE DES NOTES

    ListeNote = []
    characters = "bubble_"

    for i in soup.body.find_all(class_='ui_column is-9'):
        if i.span.get('class')[0] == 'ui_bubble_rating':
            # print(i.span.get('class'))
            string = str(i.span.get('class')[1])
            for x in range(len(characters)):
                string = string.replace(characters[x], "")
                final_string = string[:1] + '.' + string[1:]
            ListeNote.append(final_string)

    Liste_title_review = []
    for title in soup.body.find_all(class_="noQuotes"):
        Liste_title_review.append(title.text)

    Liste_nom_main_topics = []
    for topic in soup.body.find_all(class_='ui_tagcloud'):
        Liste_nom_main_topics.append(topic.text)



    list(zip(Liste_nom, Liste_dates, Liste_title_review, ListeNote, Liste_Reviews))

    df = pd.DataFrame(list(zip(Liste_nom, Liste_dates, Liste_title_review, ListeNote, Liste_Reviews)),
                      columns=['Nom', 'Dates', 'Title_Review', 'Score_reviews', 'text'])

    return df


def selenium_click_and_get_tripadvisor(URL, nombre_page):
    options = webdriver.ChromeOptions()
    options.add_extension('C:\Program Files\Google\Chrome Beta\Application\extension_1_6_8_0.crx')
    time.sleep(2)
    driver = webdriver.Chrome('C:\Program Files\Google\Chrome Beta\Application\chromedriver', options=options)
    time.sleep(2)
    driver.switch_to.window(driver.window_handles[0])
    driver.get("http://ipinfo.io")
    time.sleep(2)
    ip = driver.page_source

    soupip = BeautifulSoup(ip, 'html.parser')
    iptext = soupip.body.find_all(class_="text-green-05")[0].text
    regiontext = soupip.body.find_all(class_="text-green-05")[2].text
    countrytext = soupip.body.find_all(class_="text-green-05")[3].text
    print('start scrapping with proxies with config:')
    print('ip: ', iptext)
    print('region: ', regiontext)
    print('country: ', countrytext)
    time.sleep(1)

    driver.get(URL)

    # click sur accepter les coockies (pas necessaire avec extension)
    time.sleep(1)
    Xpathcookies = "/html/body/div[11]/div[2]/div/div[2]/div[1]/div/div[2]/div/div[1]/button"
    # driver.find_element_by_xpath(Xpathcookies).click()
    time.sleep(1)
    # click sur reviews
    reviews_click_xpath = "/html/body/div[2]/div[1]/div/div[4]/div/div/div[2]/span[1]/a/span"
    driver.find_element_by_xpath(reviews_click_xpath).click()
    time.sleep(2)
    # click sur all langages
    all_langages_click_xpath = "/html/body/div[2]/div[2]/div[2]/div[6]/div/div[1]/div[3]/div/div[2]/div/div[1]/div/div[2]/div[4]/div/div[2]/div[1]/div[1]/label/span"
    driver.find_element_by_xpath(all_langages_click_xpath).click()

    htmlstring1 = []
    for i in range(nombre_page):
        time.sleep(2)
        selector_more = "DIV.prw_rup.prw_reviews_text_summary_hsx .partial_entry span.taLnk"
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector_more))).click()
        except:
            pass
        time.sleep(2)
        htmlstring1temp = driver.page_source
        htmlstring1.append(htmlstring1temp)
        selector_next_page = 'DIV.prw_rup.prw_common_responsive_pagination .ui_pagination .nav.next'
        time.sleep(2)
        selector_first_page = 'DIV.prw_rup.prw_common_responsive_pagination .ui_pagination .nav.next'
        button_next_first_page = driver.find_element_by_css_selector(selector_first_page)
        actions = ActionChains(driver)
        actions.move_to_element(button_next_first_page).perform()

        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector_first_page))).click()
        except:
            break

    driver.quit()

    df = htmlstring_df(htmlstring1[0])

    for i in htmlstring1[1:]:
        df1 = htmlstring_df(i)
        df = pd.concat([df, df1])

    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)

    return df