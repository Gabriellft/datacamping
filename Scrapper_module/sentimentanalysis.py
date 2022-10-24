from transformers import pipeline
from deep_translator import GoogleTranslator
import pandas as pd
import time

def sentiment_analysis(df):
    sentiment_pipeline = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')
    sentiment_pipeline_liste = []
    translated_yes_no_liste = []
    translated_text = []
    for i in df["text"]:
        o = i

        if len(i) > 512:
            i = i[:512]

        if i[:18] == 'Google Translation':
            translated = GoogleTranslator(source='auto', target='en').translate(text=o[18:])
            translated_text.append(translated)
            if len(translated) > 512:
                translated = translated[:512]
            sentiment_pipeline_liste.append(sentiment_pipeline(translated))
            translated_yes_no_liste.append('True')

        else:
            sentiment_pipeline_liste.append(sentiment_pipeline(i))
            translated_yes_no_liste.append('False')
            translated_text.append(o)

    df['Sentiments analysis'] = pd.DataFrame(sentiment_pipeline_liste)
    df = pd.concat([df.drop(['Sentiments analysis'], axis=1), df['Sentiments analysis'].apply(pd.Series)], axis=1)
    df['isTranslated'] = pd.DataFrame(translated_yes_no_liste)
    df['textInEnglish'] = pd.DataFrame(translated_text)
    return df