import pandas as pd
import sqlite3

def to_sql_lite(df):


    conn = sqlite3.connect('dataset_1')
    c = conn.cursor()

    c.execute('CREATE TABLE IF NOT EXISTS reviews (Nom, Dates, Title_Review, Score, text)')
    conn.commit()

    df.to_sql('reviews', conn, if_exists='replace', index=False)

    return True










def to_pd():
    conn = sqlite3.connect('Database-reviews-portobello.db')
    c = conn.cursor()
    c.execute('''  
    SELECT * FROM reviews
              ''')
    df_from_slq_serv = pd.DataFrame(c.fetchall(), columns=['Nom', 'Dates', 'Title_Review', 'Score', 'text'])

    return df_from_slq_serv

