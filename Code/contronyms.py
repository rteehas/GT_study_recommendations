import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from google_trans_new import google_translator
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

translator = google_translator()

def get_languages(google_translate_fname):
    df_goog_lang = pd.read_csv(google_translate_fname)
    list_lang = df_goog_lang.iloc[:, 1].values
    return list_lang

def backtranslate_once(lang, text):
  en_lang = translator.translate(text, lang_src='en',lang_tgt=lang)
  lang_en = translator.translate(en_lang, lang_src=lang, lang_tgt='en')
  return (lang, en_lang, lang_en)

def threaded_longitudinal(lang, texts):
  pool = ThreadPool(8)
  backtranslate_partial = partial(backtranslate_once, lang)
  try:
    results = pool.map(backtranslate_partial, texts)
  except Exception as e:
    raise e
  pool.close()
  pool.join()

  return results

def threaded_latitudinal(langs, text):
  pool = ThreadPool(8)
  backtranslate_partial = partial(backtranslate_once, text=text)
  try:
    results = pool.map(backtranslate_partial, langs)
  except Exception as e:
    raise e
  pool.close()
  pool.join()

  return results

def process_results(tup_list, lang_list, df_goog_lang):
  #this sorts the results by language in case they get out of order from threading
  #then it splits the tuples and converts to a dataframe

  sorted_results = sorted(tup_list, key=lambda tup : lang_list.index(tup[0]))
  langs, en_lang_translated, lang_en_translated = zip(*sorted_results)
  lang_names = []
  for lang in langs:
    lang_names.append(df_goog_lang.loc[df_goog_lang.iloc[:,1]==lang].Language.values[0])
  df_lang = pd.DataFrame(data=np.stack([np.array(lang_names), np.array(en_lang_translated),np.array(lang_en_translated)], axis = -1), columns=["Language", "En2Language", "Language2En"])
  return df_lang

def run(fname, lat_output_fname, intermediate_lang_id, long_output_fname):
    sentences = ["The war cry for justice was all over - Much to the chagrin of the authorities.",
                 "The regime's decision adumbrates the underlying agenda.",
                 "The community was anxious over the passing of the much needed reforms.",
                 "The think tank  published a paper outlining their apology of capital punishment.",
                 "Aught was left of the petition's validity",
                 "The President exhorted his ministers to buckle up as key opposition support for his economic plan was about to buckle.",
                 "The voters were chuffed to see the bill passed.",
                 "The lawyer's discursive narration swayed the jury.",
                 "The court enjoined the violence!",
                 "The EU block opposed an eventual imposition of anti-dumping measures.",
                 "A fulsome eulogy was delivered by the sly counsel.",
                 "The gig-economy agency decided to garnish the refunds!",
                 "Members of the anti-theocratic revolutionary movement overthrew the peers in power!",
                 "The traitors disappointingly decided to fight with the colonialists."]

    text = "The court enjoined the violence!"
    list_lang = get_languages(fname)

    results = threaded_latitudinal(list_lang.tolist(), text)
    df_trans = process_results(results, list_lang.tolist())
    df_trans.to_csv(lat_output_fname, index=False)

    long_results = threaded_longitudinal(intermediate_lang_id, sentences)
    df_long = pd.DataFrame(data=long_results, columns=["En", "En2" + intermediate_lang_id, intermediate_lang_id + "2En"])
    df_long.to_csv(long_output_fname, index=False)