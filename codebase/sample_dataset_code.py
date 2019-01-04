import pandas as pd
import nltk

import re
import csv


def read_csv(file_path):
    return pd.read_csv(file_path)


def get_word_forms(word):
    forms = [word]
    root_words = lemma(word)
    for root_word in root_words:
        if root_word:
            forms.append(root_word)
    return forms


def compile_tag_values(file_path, key):
    data_frame = read_csv(file_path)
    words = {}

    for i in range(len(data_frame)):
        word = str(data_frame[key][i]).strip().lower()
        if word not in stop_words and word not in punct_list:
            if '-' in word:
                parts = word.split("-")
                last_part_forms = get_word_forms(parts.pop())
                words[word] = []

                for form in last_part_forms:
                    new_parts = parts + [form]
                    words[word].append("".join(new_parts))
                    words[word].append(new_parts)

            else:
                words[word] = get_word_forms(word)

    return words


def get_tokenized_header(header_value):
    return word_tokenize(str(header_value).lower())


def get_tag_and_header_intersection(tags, headers, prefix):
    assert type(tags) is dict
    assert type(headers) is list

    headers = set(headers)
    matches = []

    for key, val in tags.items():
        try:  # is single word type
            val = set(val)
            if headers & val:
                matches.append(prefix + key)

        except TypeError:  # is multiple word type
            for x in val:
                if type(x) is list:
                    x = set(x)
                    if len(headers & x) == len(x):
                        matches.append(prefix + key)
                else:
                    if x in headers:
                        matches.append(prefix + key)

    return matches


def get_tag_values_for_snippet(tags, h1, h2, h3):
        h1_words = h2_words = []
        for word in get_tokenized_header(h1):
            h1_words += get_word_forms(word)
        for word in get_tokenized_header(h2):
            h2_words += get_word_forms(word)

        if h3 == 'nan':
            return get_tag_and_header_intersection(tags, h1_words, 'h1: ') or ['any']
        else:
            return get_tag_and_header_intersection(tags, h2_words, 'h2: ') or \
                   get_tag_and_header_intersection(tags, h1_words, 'h1: ') or ['any']


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  cleanr2 = re.compile('{.*?}')
  cleantext2 = re.sub(cleanr2, '', cleantext)
  return str(cleantext2)

# Import the data using pandas
df = pd.read_csv("data/input/oldContentData1_formatted.csv")
df.dropna(axis=0,how="all",inplace=True)

#Create the master dictionary
df = df.reset_index()
df.drop("index",axis=1,inplace=True)

df.drop(["blog_post_name","blog_post_title"],inplace=True,axis=1)

df["h2_tag"] = df["h2_tag"].astype(str)
df["h2_tag"] = df["h2_tag"].apply(cleanhtml)

df["h3_tag"] = df["h3_tag"].astype(str)
df["h3_tag"] = df["h3_tag"].apply(cleanhtml)
# Mapping of post ids with the respective snippet ids
df_dict = {}

df.rename(index=str, columns={"blog_post_id":"post_id"},inplace=True)

for i in range(len(df["post_id"])):
    v = df_dict.get(int(df["post_id"][i]),0)
    df_dict[int(df["post_id"][i])] = v + 1


for k,v in df_dict.items():
    temp = df[df["post_id"] == k]
    ls = []
    temp.reset_index(inplace=True)
    temp.drop("index",axis=1,inplace=True)
    for j in range(len(temp)):
        ls.append(int(temp["snippet_id"][j]))
    df_dict[k] = ls
    

# Lemmatization and Stemming of Words
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('rslp')

def lemma(word):
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    wordnet = WordNetLemmatizer()
    
    ls = [porter.stem(word),snowball.stem(word),wordnet.lemmatize(word)]
    
    curr_set = set(ls)
    
    ls = list(curr_set)
    for w in ls:
        if w == word:
            ls.remove(w)
    return ls

#Import the subject tag types
            
accomodations =pd.read_csv("data/input/accomodations.csv")
places_to_visit = pd.read_csv("data/input/places-to-visit.csv")
things_to_do = pd.read_csv("data/input/things_to_do.csv")

#Import the stopwords list from nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punct_list = set(["-",":",".","!","_","'","&","...",","])
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def word_sep(words_list):
    ls = []
    for word in words_list:
        temp = word.lower().split("-")
        ls.append(temp)
    return ls


# Create the list of all words for their respective subject tag type
accomodations1 = []
places_to_visit1 = []
things_to_do1 = []


for i in range(len(accomodations)):
    word = str(accomodations["Accommodations"][i]).strip().lower()
    accomodations1.append(word)

for i in range(len(places_to_visit)):
    word = str(places_to_visit["Places to Visit"][i]).strip().lower()
    places_to_visit1.append(word)
    
for i in range(len(things_to_do)):
    word = str(things_to_do["Things to do"][i]).strip().lower()
    things_to_do1.append(word)
    
#---- Create a list of P1 Tags for things to do and places to visit
    
p1_things_to_do = pd.read_csv("data/input/p1_things_to_do.csv")
p1_places_to_visit = pd.read_csv("data/input/p1_places_to_visit.csv")

p1_single_word_place_types = {}

for i in range(len(p1_places_to_visit)):
    word = str(p1_places_to_visit["tags"][i]).strip().lower()
    
    if word not in stop_words and word not in punct_list:
        if "-" not in word:
            p1_single_word_place_types[word] = []
            p1_single_word_place_types[word].append(word)
            
            root_word = lemma(word)
            
            for w in root_word:
                if w != '':
                    p1_single_word_place_types[word].append(w)
                            
p1_multiple_word_place_types = {} 
for i in range(len(p1_places_to_visit)):
    word = str(p1_places_to_visit["tags"][i]).strip().lower()
    
    if word not in stop_words and word not in punct_list:
        if "-" in word:
            p1_multiple_word_place_types[word] = []
            word_split = word.split("-")
            last_word = word_split[-1]
            root_word = lemma(last_word)
            
            p1_multiple_word_place_types[word].append(["".join(word_split)])
            p1_multiple_word_place_types[word].append(word_split)
           
            for w in root_word:
                if w!='':
                    partial_list = word_split[:-1]
                    partial_list.append(w)
                    if partial_list not in p1_multiple_word_place_types.values():
                        p1_multiple_word_place_types[word].append(["".join(partial_list)])
                        p1_multiple_word_place_types[word].append(partial_list)
    
#Create a list of p1 Activity Types
p1_single_things_to_do_types = {}                            
for i in range(len(p1_things_to_do)):
    word = str(p1_things_to_do["tags"][i]).strip().lower()
    
    
    if word not in stop_words and word not in punct_list:
        if "-" not in word:
            p1_single_things_to_do_types[word] = []
            p1_single_things_to_do_types[word].append(word)
            
            root_word = lemma(word)
            
            for w in root_word:
                if w != '':
                    p1_single_things_to_do_types[word].append(w)          
                    

p1_multiple_word_things_to_do_types = {} 
for i in range(len(p1_things_to_do)):
    word = str(p1_things_to_do["tags"][i]).strip().lower()
    
    if word not in stop_words and word not in punct_list:
        if "-" in word:
            p1_multiple_word_things_to_do_types[word] = []
            word_split = word.split("-")
            last_word = word_split[-1]
            root_word = lemma(last_word)
            
            p1_multiple_word_things_to_do_types[word].append(["".join(word_split)])
            p1_multiple_word_things_to_do_types[word].append(word_split)
           
            for w in root_word:
                if w!='':
                    partial_list = word_split[:-1]
                    partial_list.append(w)
                    if partial_list not in p1_multiple_word_things_to_do_types.values():
                        p1_multiple_word_things_to_do_types[word].append(["".join(partial_list)])
                        p1_multiple_word_things_to_do_types[word].append(partial_list)    
    

# Create a set to remove the duplicates and find intersections    
accom = set(accomodations1)
things = set(things_to_do1)
places = set(places_to_visit1)
things_to_do_set = set(['things','to','do'])


# Snippet Level Mapping of the subjects tag types
df_snippet_subject_tagging = {}
df.rename(index=str, columns={"h1_tag":"h1_tag_value","h2_tag":"h2_tag_value","h3_tag":"h3_tag_value"},inplace=True)


for i in range(len(df)):
    h2 = word_tokenize(str(df["h2_tag_value"][i]).lower())
    h1 = word_tokenize(str(df["h1_tag_value"][i]).lower())
    
    key = str(df["snippet_id"][i])
    
    h2_set = set(list(h2))
    h1_set = set(list(h1))
    
    df_snippet_subject_tagging[key] = []
    
    if str(df["h3_tag_value"][i]) == "nan":

        if len(h1_set&accom) > 0:
            df_snippet_subject_tagging[key].append("Accommodation")
            
        elif len(h1_set&things) > 0:
            df_snippet_subject_tagging[key].append("Things To Do")
        elif len(things_to_do_set&h1_set)==3:
            df_snippet_subject_tagging[key].append("Things To Do")
        elif len(h1_set&places)> 0:
            df_snippet_subject_tagging[key].append("Places To Visit")
    
    elif str(df["h3_tag_value"][i]) != "nan":
        if len(h2_set&accom) > 0:
            df_snippet_subject_tagging[key].append("Accommodation")
        elif len(h1_set&accom) > 0:
            df_snippet_subject_tagging[key].append("Accommodation")
            
        elif len(h2_set&things) > 0:
            df_snippet_subject_tagging[key].append("Things To Do")
        elif len(things_to_do_set&h2_set)==3:
            df_snippet_subject_tagging[key].append("Things To Do")
        elif len(h1_set&things) > 0:
            df_snippet_subject_tagging[key].append("Things To Do")
        elif len(things_to_do_set&h1_set)==3:
            df_snippet_subject_tagging[key].append("Things To Do")
            
        elif len(h2_set&places)> 0:
            df_snippet_subject_tagging[key].append("Places To Visit")
        elif len(h1_set&places)> 0:
            df_snippet_subject_tagging[key].append("Places To Visit")

#------------ Mapping of the P1 tags w.r.t Subjects--------------- 

for k,v in df_snippet_subject_tagging.items():
    if len(v) < 1:
        key = int(k)
        temp_df = df[df["snippet_id"] == key]
        temp_df.reset_index(inplace=True)
        print(len(temp_df))
        
        h2 = word_tokenize((temp_df.iloc[0]["h2_tag_value"]).lower().strip())
        h1 = word_tokenize((temp_df.iloc[0]["h1_tag_value"]).lower().strip())
        h2_set = set(h2)
        h1_set = set(h1)
        
        h3 = word_tokenize((temp_df.iloc[0]["h3_tag_value"]).lower().strip())
        if "nan" in h3:
            for tag,value in p1_single_things_to_do_types.items():
                curr_set = set(value)
                if len(curr_set & h1_set) > 0:
                    df_snippet_subject_tagging[k].append("Things To Do")
            if len(df_snippet_subject_tagging[k]) < 1:
                for tag2,value2 in p1_multiple_word_things_to_do_types.items():
                    for words in value2:
                        curr_mul_set = set(words)
                        if len(curr_mul_set&h1_set) == len(curr_mul_set):
                            df_snippet_subject_tagging[k].append("Things To Do")
            
            if len(df_snippet_subject_tagging[k]) < 1:
                for tag3,value3 in p1_single_word_place_types.items():
                    curr_single_set2 = set(value3)
                    if len(curr_single_set2&h1_set) > 0:
                        df_snippet_subject_tagging[k].append("Places To Visit")
            
            if len(df_snippet_subject_tagging[k]) < 1:
                for tag4,value4 in p1_multiple_word_place_types.items():
                    for words in value4:
                        curr_multiple_set3 = set(words)
                        if len(curr_multiple_set3&h1_set) == len(curr_multiple_set3):
                            df_snippet_subject_tagging[k].append("Places To Visit")
        else:
            for tag,value in p1_single_things_to_do_types.items():
                curr_set = set(value)
                if len(curr_set & h2_set) > 0:
                    df_snippet_subject_tagging[k].append("Things To Do")
                elif len(curr_set & h1_set) > 0:
                    df_snippet_subject_tagging[k].append("Things To Do") 
            
            if len(df_snippet_subject_tagging[k]) < 1:
                for tag2,value2 in p1_multiple_word_things_to_do_types.items():
                    for words in value2:
                        curr_mul_set = set(words)
                        if len(curr_mul_set&h2_set) == len(curr_mul_set):
                            df_snippet_subject_tagging[k].append("Things To Do")
                        elif len(curr_mul_set&h1_set) == len(curr_mul_set):
                            df_snippet_subject_tagging[k].append("Things To Do")
            
            if len(df_snippet_subject_tagging[k]) < 1:
                for tag3,value3 in p1_single_word_place_types.items():
                    curr_single_set2 = set(value3)
                    if len(curr_single_set2&h2_set) > 0:
                        df_snippet_subject_tagging[k].append("Places To Visit")
                    elif len(curr_single_set2&h1_set) > 0:
                        df_snippet_subject_tagging[k].append("Places To Visit")
            
            if len(df_snippet_subject_tagging[k]) < 1:
                for tag4,value4 in p1_multiple_word_place_types.items():
                    for words in value4:
                        curr_multiple_set3 = set(words)
                        if len(curr_multiple_set3&h2_set) == len(curr_multiple_set3):
                            df_snippet_subject_tagging[k].append("Places To Visit")
                        elif len(curr_multiple_set3&h1_set) == len(curr_multiple_set3):
                            df_snippet_subject_tagging[k].append("Places To Visit")
            

# --- End of Subject Level Tagging --- 
        
#---- Start of place - type tags ----
            
#Import the places type and activity type data
        
places_types = pd.read_csv("data/input/places_types.csv")
activity_type = pd.read_csv("data/input/activity_type.csv")


#Create a list of Places Type
single_word_place_types = {}

for i in range(len(places_types)):
    word = str(places_types["types"][i]).strip().lower()
    
    if word not in stop_words and word not in punct_list:
        if "-" not in word:
            single_word_place_types[word] = []
            single_word_place_types[word].append(word)
            
            root_word = lemma(word)
            
            for w in root_word:
                if w != '':
                    single_word_place_types[word].append(w)
                            
multiple_word_place_types = {} 
for i in range(len(places_types)):
    word = str(places_types["types"][i]).strip().lower()
    
    if word not in stop_words and word not in punct_list:
        if "-" in word:
            multiple_word_place_types[word] = []
            word_split = word.split("-")
            last_word = word_split[-1]
            root_word = lemma(last_word)
            
            multiple_word_place_types[word].append(["".join(word_split)])
            multiple_word_place_types[word].append(word_split)
           
            for w in root_word:
                if w!='':
                    partial_list = word_split[:-1]
                    partial_list.append(w)
                    if partial_list not in multiple_word_place_types.values():
                        multiple_word_place_types[word].append(["".join(partial_list)])
                        multiple_word_place_types[word].append(partial_list)
            
                                       
#Create a list of Activity Types
single_word_activity_types = {}                            
for i in range(len(activity_type)):
    word = str(activity_type["activity_type"][i]).strip().lower()
    
    
    if word not in stop_words and word not in punct_list:
        if "-" not in word:
            single_word_activity_types[word] = []
            single_word_activity_types[word].append(word)
            
            root_word = lemma(word)
            
            for w in root_word:
                if w != '':
                    single_word_activity_types[word].append(w)          
                    

multiple_word_activity_types = {} 
for i in range(len(activity_type)):
    word = str(activity_type["activity_type"][i]).strip().lower()
    
    if word not in stop_words and word not in punct_list:
        if "-" in word:
            multiple_word_activity_types[word] = []
            word_split = word.split("-")
            last_word = word_split[-1]
            root_word = lemma(last_word)
            
            multiple_word_activity_types[word].append(["".join(word_split)])
            multiple_word_activity_types[word].append(word_split)
           
            for w in root_word:
                if w!='':
                    partial_list = word_split[:-1]
                    partial_list.append(w)
                    if partial_list not in multiple_word_activity_types.values():
                        multiple_word_activity_types[word].append(["".join(partial_list)])
                        multiple_word_activity_types[word].append(partial_list)

#Activity Type Mapping  
df_snippet_activity_tagging = {}

for i in range(len(df)):
    h2 = word_tokenize(str(df["h2_tag_value"][i]).lower())
    h3 = word_tokenize(str(df["h3_tag_value"][i]).lower())
    h1 = word_tokenize(str(df["h1_tag_value"][i]).lower())
    h2_new = []
    h3_new = []
    h1_new = []
    for word in h2:
        h2_new.append(word)
        w = lemma(word)
        for l in w:
            if l != '':
                h2_new.append(l)
    
    
    for word in h3:
        h3_new.append(word)
        w = lemma(word)
        for l in w:
            if l != '':
                h3_new.append(l)
                
    for word in h1:
        h1_new.append(word)
        w = lemma(word)
        for l in w:
            if l != '':
                h1_new.append(l)
    
    key = str(df["snippet_id"][i])
    
    h2_set = set(list(h2_new))
    h3_set = set(list(h3_new))
    h1_set = set(list(h1_new))
    
    df_snippet_activity_tagging[key] = []
    
    for k,val in single_word_activity_types.items():
        single_word_activity_types_set = set(val)         

        if len(h3_set&single_word_activity_types_set) > 0:
            if str("h3: "+str(k)) not in df_snippet_activity_tagging[key]:
                df_snippet_activity_tagging[key].append("h3: "+str(k))
            
        elif len(h2_set&single_word_activity_types_set) > 0:
            if str("h2: "+str(k)) not in df_snippet_activity_tagging[key]:
                df_snippet_activity_tagging[key].append("h2: "+str(k))
                
        elif len(h1_set&single_word_activity_types_set) > 0:
            if str("h1: "+str(k)) not in df_snippet_activity_tagging[key]:
                df_snippet_activity_tagging[key].append("h1: "+str(k))
     
        
    for k,value in multiple_word_activity_types.items():
        for val in value:
            multiple_word_activity_types_set = set(val)         
    
            if len(h3_set&multiple_word_activity_types_set) == len(multiple_word_activity_types_set):
                if str("h3: "+str(k)) not in df_snippet_activity_tagging[key]:
                    df_snippet_activity_tagging[key].append("h3: "+str(k))
                
            elif len(h2_set&multiple_word_activity_types_set) == len(multiple_word_activity_types_set):
                if str("h2: "+str(k)) not in df_snippet_activity_tagging[key]:
                    df_snippet_activity_tagging[key].append("h2: "+str(k))
                    
            elif len(h1_set&multiple_word_activity_types_set) == len(multiple_word_activity_types_set):
                if str("h1: "+str(k)) not in df_snippet_activity_tagging[key]:
                    df_snippet_activity_tagging[key].append("h1: "+str(k))         


#Place Type Mapping
                
df_snippet_place_types_tagging = {}

for i in range(len(df)):
    h2 = word_tokenize(str(df["h2_tag_value"][i]).lower())
    h3 = word_tokenize(str(df["h3_tag_value"][i]).lower())
    h1 = word_tokenize(str(df["h1_tag_value"][i]).lower())
    h2_new = []
    h3_new = []
    h1_new = []
    for word in h2:
        h2_new.append(word)
        w = lemma(word)
        for l in w:
            if l != '':
                h2_new.append(l)
    
    for word in h3:
        h3_new.append(word)
        w = lemma(word)
        for l in w:
            if l != '':
                h3_new.append(l)
                
    for word in h1:
        h1_new.append(word)
        w = lemma(word)
        for l in w:
            if l != '':
                h1_new.append(l)
    
    key = str(df["snippet_id"][i])
    
    h2_set = set(list(h2_new))
    h3_set = set(list(h3_new))
    h1_set = set(list(h1_new))
    
    df_snippet_place_types_tagging[key] = []
    
    for k,val in single_word_place_types.items():
        ex_places_type_set = set(val)

        if len(h3_set&ex_places_type_set) > 0:
            if str("h3: "+str(k)) not in df_snippet_place_types_tagging[key]:
                df_snippet_place_types_tagging[key].append("h3: "+str(k))
            
        elif len(h2_set&ex_places_type_set) > 0:
            if str("h2: "+str(k)) not in df_snippet_place_types_tagging[key]:
                df_snippet_place_types_tagging[key].append("h2: "+str(k))
                
        elif len(h1_set&ex_places_type_set) > 0:
            if str("h1: "+str(k)) not in df_snippet_place_types_tagging[key]:
                df_snippet_place_types_tagging[key].append("h1: "+str(k))
                
    for k,value in multiple_word_place_types.items():
        for val in value:
            multiple_word_place_types_set = set(val)         
    
            if len(h3_set&multiple_word_place_types_set) == len(multiple_word_place_types_set):
                if str("h3: "+str(k)) not in df_snippet_place_types_tagging[key]:
                    df_snippet_place_types_tagging[key].append("h3: "+str(k))
                
            elif len(h2_set&multiple_word_place_types_set) == len(multiple_word_place_types_set):
                if str("h2: "+str(k)) not in df_snippet_place_types_tagging[key]:
                    df_snippet_place_types_tagging[key].append("h2: "+str(k))
                    
            elif len(h1_set&multiple_word_place_types_set) == len(multiple_word_place_types_set):
                if str("h1: "+str(k)) not in df_snippet_place_types_tagging[key]:
                    df_snippet_place_types_tagging[key].append("h1: "+str(k)) 


def write_tag_data_to_csv():
    month_tags = compile_tag_values('data/input/month_types.csv', 'types')
    season_tags = compile_tag_values('data/input/season_types.csv', 'types')
    with_whom_tags = compile_tag_values('data/input/with_whom_types.csv', 'types')
    occassion_tags = compile_tag_values('data/input/occassion_types.csv', 'types')
    time_of_day_tags = compile_tag_values('data/input/timeofday_type.csv', 'types')
    budget_tags = compile_tag_values('data/input/budget_type.csv', 'types')
    theme_tags = compile_tag_values('data/input/theme_type.csv', 'types')

    csv_header = None
    result = []

    for i in range(len(df)):
        key = str(df["snippet_id"][i])
        data = {
            "snippet_id": key,
            "subject_tag": ", ".join(df_snippet_subject_tagging[key]),
            "place_tags": ", ".join(df_snippet_place_types_tagging[key]),
            "activity_tags": ", ".join(df_snippet_activity_tagging[key]),
            "month_tags": ", ".join(get_tag_values_for_snippet(month_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i])),
            "season_tags": ", ".join(get_tag_values_for_snippet(season_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i])),
            "with_whom_tags": ", ".join(get_tag_values_for_snippet(with_whom_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i])),
            "occassion_tags": ", ".join(get_tag_values_for_snippet(occassion_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i])),
            "time_of_day_tags": ", ".join(get_tag_values_for_snippet(time_of_day_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i])),
            "budget_tags": ", ".join(get_tag_values_for_snippet(budget_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i])),
            "theme_tags": ", ".join(get_tag_values_for_snippet(theme_tags, df["h1_tag_value"][i], df["h2_tag_value"][i], df["h3_tag_value"][i]))
        }
        result.append(data)
        if not csv_header:
            csv_header = list(data.keys())

    with open('data/result/result_new.csv', 'w', newline='') as csvfile:
         x = csv.DictWriter(csvfile, fieldnames=csv_header)
         x.writeheader()
         x.writerows(result)


write_tag_data_to_csv()


"""            
df_place_type = pd.DataFrame.from_dict(df_snippet_place_types_tagging,columns=["v1","v2","v3","v4","v5"],orient="index")
df_place_type.to_csv("new_place_type_tags.csv",encoding="utf8")


df_activity_type = pd.DataFrame.from_dict(df_snippet_activity_tagging,columns=["v1","v2","v3","v4"],orient="index")
df_activity_type.to_csv("new_activity_type_tags.csv",encoding="utf8")
"""


#-------------- Finding Image Alt Tag Values---------------------
from bs4 import BeautifulSoup

def image_content(s):
    soup = BeautifulSoup(s)
    
    alt_content = ""
    if soup.img:
        alt_content = soup.img['alt'] 
    
    return alt_content

df['snippet_image'] = df["snippet_image"].astype(str)
    
df["Alt_Content"] = df["snippet_image"].apply(lambda x: image_content(str(x)))
    
def clean_num_chars(s):
   s = re.sub('\d', '', s)
   s = re.sub('_', '', s)
   s = re.sub('-', ' ', s)
   return s

df["Alt_Content"] = df["Alt_Content"].apply(clean_num_chars)

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
from collections import OrderedDict

snippets_destinations3 = {}

for i in range(len(df)):
    
    curr_str = df.iloc[i]["Alt_Content"].lower().strip()
    curr_str = curr_str.replace("\n","")
    
    h3 = str(df.iloc[i]["h3_tag_value"])
    h3 = cleanhtml(h3)
    h3 = h3.lower().strip()
    h3 = h3.replace("-"," ")
    h2 = str(df.iloc[i]["h2_tag_value"])
    h2 = cleanhtml(h2)
    h2 = h2.lower().strip()
    h2 = h2.replace("-"," ")
    
    h3_word_tokens = word_tokenize(h3)
    
    h3_word_tokens2 = []
    
    for w in h3_word_tokens:
        if w not in stop_words and w not in punct_list:
            h3_word_tokens2.append(w)
    
    
   # h3_word_tokens_set = set(h3_word_tokens2
    
    
    h2_word_tokens = word_tokenize(h2)
    h2_word_tokens2 = []
    for w in h2_word_tokens:
        if w not in stop_words and w not in punct_list:
            h2_word_tokens2.append(w)
    #h2_word_tokens_set = set(h2_word_tokens2)
    key = df["snippet_id"][i]
    
    snippets_destinations3[key] = []        
    
    curr_str_tokens = word_tokenize(curr_str)
    curr_str_tokens2 =[]
    
    for word in curr_str_tokens:
        if word not in stop_words and word not in punct_list:
            curr_str_tokens2.append(word)
            
    #curr_str_set = set(curr_str_tokens2)
    
    if h3 != "nan":
        if len(intersection(curr_str_tokens2,h3_word_tokens2))>0:
            
            snippets_destinations3[key].append(list(OrderedDict.fromkeys(intersection(curr_str_tokens2,h3_word_tokens2))))
            
    elif h3 == "nan":
        if len(intersection(curr_str_tokens2,h2_word_tokens2))>0:
            
            snippets_destinations3[key].append(list(OrderedDict.fromkeys(intersection(curr_str_tokens2,h2_word_tokens2))))


#AIzaSyBvSjN-rST9HGM3aseC7IFAXyW1RT5x_G4

#--------------------- Main Destination Mapping -------------------------
        
activity_list = []
places_to_visit_list = []

for k,v in single_word_activity_types.items():
    for w in v:
        activity_list.append(w)

for k,v in multiple_word_activity_types.items():
    for w in v:
        for word in w:
            activity_list.append(word)



for k,v in single_word_place_types.items():
    for w in v:
        places_to_visit_list.append(w)
        
for k,v in multiple_word_place_types.items():
    for w in v:
        for word in w:
            places_to_visit_list.append(word)

        
blogs_mapped_destinations = {}
i= 0
df["snippet_text"] = df["snippet_text"].astype(str)
df["cleaned_text"] = df["snippet_text"].apply(cleanhtml)

while i < len(df):
    
    curr_str = df.iloc[i]["h1_tag_value"].lower().strip()
    h1 = word_tokenize(curr_str)
    h1_refined_words = []
    
    
    for j in h1:
        if "-" not in j:
            if j not in stop_words and j not in punct_list:
                h1_refined_words.append(j)
        else:
            j_list = j.split("-")
            for wj in j_list:
                if wj not in stop_words and wj not in punct_list:
                    h1_refined_words.append(wj)
    
    post_id = df.iloc[i]["post_id"]
    h1_word_dict = {}
    temp_df = df[df["post_id"] == post_id]
    
    temp_df.reset_index(inplace=True)
    

    for word in h1_refined_words:
        if word not in activity_list and word not in places_to_visit_list:
            v = h1_word_dict.get(word,0)
            h1_word_dict[word] = v+1
                
    blogs_mapped_destinations[post_id] = h1_word_dict
        
    i = i + len(temp_df)
    
    
#----------- Article Tags Mapping -------------------
   

tags = pd.read_csv("data/input/tags.csv")


tags_intersection_posts = {}


for k,v in blogs_mapped_destinations.items():
    curr_post_id = k
    tags_intersection = {}
    for key,value in v.items():
        
        curr_tags = tags[tags["post_id"]== k]
        curr_tags.reset_index(inplace=True)
        if len(curr_tags) > 0:
            tags_list = curr_tags.iloc[0]["tags"].lower().strip()

            tokens = word_tokenize(tags_list)
            tags_cleaned = []
            for word in tokens:
                if word not in stop_words and word not in punct_list:
                    tags_cleaned.append(word)

            if key in tags_cleaned:
                v = tags_intersection.get(key,0)
                tags_intersection[key] = v+1

        tags_intersection_posts[curr_post_id] = tags_intersection



#--------------- Cleaning of the Main Destinations Data-----------
    
tags_intersection_posts_cleaned = {}    

for k,v in tags_intersection_posts.items():
    temp_ls = []
    for key,value in v.items():
        if key not in places_to_visit_list and key not in activity_list and key not in accomodations1 and key not in things_to_do1 and key not in places_to_visit1:
            temp_ls.append(key)
    
    tags_intersection_posts_cleaned[k] = temp_ls
            
    
# mapped_destinations = pd.DataFrame.from_dict(tags_intersection_posts_cleaned,columns=["v1","v2","v3"],orient="index")
        
words_freq_list = {}

for k,v in tags_intersection_posts_cleaned.items():
    
    for word in v:
        count = words_freq_list.get(word,0)
        words_freq_list[word] = count + 1
      

# Import a manual exhaustive list

exhaustive_df = pd.read_csv("data/input/word_freq.csv")
exhaustive_df = exhaustive_df[exhaustive_df["Bool"]=="n"]

exhaustive_list = exhaustive_df["words"].tolist()


tags_intersection_posts_cleaned = {}  

for k,v in tags_intersection_posts.items():
    temp_ls = []
    for key,value in v.items():
        if key not in places_to_visit_list and key not in activity_list and key not in accomodations1 and key not in things_to_do1 and key not in places_to_visit1 and key not in exhaustive_list:
            temp_ls.append(key)
    
    tags_intersection_posts_cleaned[k] = temp_ls    
    

snippets_cleaned_sub_destinations = {}

for k,v in snippets_destinations3.items():
    temp_ls = []
    if len(v)>0:
        curr_ls = v[0]
        for key in curr_ls:
            #if key not in places_to_visit_list and key not in activity_list and key not in accomodations1 and key not in things_to_do1 and key not in places_to_visit1 and key not in exhaustive_list:
            temp_ls.append(key)
        snippets_cleaned_sub_destinations[k] = temp_ls

            
#------- Mapping of Main and Sub Destinations --------------

final_mappings = {}
for k,v in tags_intersection_posts_cleaned.items():
    
    main_destination = ",".join(v)
    post_id = k
    
    temp_df = df[df["post_id"]==post_id]
    temp_df.reset_index(inplace=True)
    
    temp_post_dict = {}
    temp_snippets_ls = temp_df["snippet_id"].tolist()
    temp_post_dict[post_id] = temp_snippets_ls
    
    temp_sub_ls = []
    for d in temp_snippets_ls:
        if d in snippets_cleaned_sub_destinations.keys():
            ls = snippets_cleaned_sub_destinations[d]
            t = [d,ls]
            if t not in temp_sub_ls:
                temp_sub_ls.append([d,ls])
                    
                    
    curr_ls = final_mappings.get(main_destination,[])
    final_mappings[main_destination] = curr_ls + temp_sub_ls

non_mapped_sub_dests = final_mappings['']        
non_mapped_sub_dests2 = [] 


    
for k,v in final_mappings.items():
    temp_sub_ls = []
    
    if k != '':
        for ls in non_mapped_sub_dests:
            destination = ls[1]
            destination_set = set(destination)
            if len(destination) > 0:
                for dests in v:
                    curr_dest = dests[1]
                    curr_dest_set = set(curr_dest)
                    if len(destination_set&curr_dest_set) == len(destination_set):
                        if ls not in temp_sub_ls:
                            temp_sub_ls.append(ls)

        if len(temp_sub_ls) > 0:
            
            non_mapped_sub_dests2.append(temp_sub_ls) 
            curr_ls = final_mappings.get(k)
            final_mappings[k] =curr_ls + temp_sub_ls
"""            
mls = []
    
for k,v in final_mappings.items():
    if len(k) > 0:
        for d in v:
            mls.append([k,d])
            
df_mls = pd.DataFrame(mls,columns =["main_dest","sub_dest"])
    
df_mls.to_csv("mapped_sub_destinations.csv",encoding="utf8")    
"""

with open('data/result/result_mapped_dest.csv', 'w', newline='') as csvfile:
     x = csv.writer(csvfile)
     for key, value in final_mappings.items():
         x.writerow([key, value])





#---------------------Probabilty Model--------------------------
















