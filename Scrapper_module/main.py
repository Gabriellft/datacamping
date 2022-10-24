import time
import scrapping_tripadvisor as scrapping
import sentimentanalysis as sentiment
import to_sql_lite as to
start = time.time()

URL = 'https://www.tripadvisor.com/Restaurant_Review-g154948-d708071-Reviews-Portobello-Whistler_British_Columbia.html'

df = scrapping.selenium_click_and_get_tripadvisor(URL,99999)


df = sentiment.sentiment_analysis(df)

to.to_sql_lite(df)


end = time.time()

elapsed = end - start

print(f'Temps d\'exécution : {elapsed}')

print(df.head())