from fastapi import FastAPI
import pandas as pd
import sqlite3
app = FastAPI(debug = True)


@app.get("/")
def root_route():
    """
    Root route of the API ;)
    :return:
        response[json] -- content of ping
    """
    return {"content" : "Ping succeeded!"}

@app.get("/get_updated_comments")
def get_updated_comments(url_google_reviews_restaurant:str):
    """
    Root route of the API ;)
    :return:
        response[json] -- content of ping
    """
    conn = sqlite3.connect('Database-reviews-portobello.db')
    c = conn.cursor()
    c.execute('''  
        SELECT * FROM reviews
                  ''')
    df_from_slq_serv = pd.DataFrame(c.fetchall(), columns=['Nom', 'Dates', 'Title_Review', 'Score_reviews', 'text','label','score','isTranslated','textInEnglish','sentiment','autorating','Topic',"Name"])


    # Prepare the actor input
    return {"all_reviews" : df_from_slq_serv.to_json(orient="records") }
