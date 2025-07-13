
import argparse
import os
import sys
new_path = os.getcwd()
sys.path.append(new_path)
from utils.llama3_parameter import args
from utils.llama3 import Llama3Model
import time
from tqdm import tqdm


import json
def HumanMessage(content):
    return {"role": "user", "content": content}
def AIMessage(content):
    return {"role": "assistant", "content": content}
def SystemMessage(content):
    return {"role": "system", "content": content}


chat = Llama3Model(model=args.model_path, tensor_parallel_size=1)


with open('./data/SNIPS/test', 'r', encoding='utf-8') as fin:
    inputs_text =[]
    for line in fin.readlines():
        token = line.strip().split()
        text = ' '.join([x.rsplit(':', maxsplit=1)[0] for x in token[:-2]])
        intent= token[-1]
        inputs_text.append({
            'text':text,
            'gold_intent': intent
        })
  
prompt = """
You need to annotate some sentences I gave you in the following, which includes intent and slots.

Given following sentences, first choose the intent of the sentences from the following intent list: [AddToPlaylist ; BookRestaurant ; GetWeather ; PlayMusic ; RateBook ; SearchCreativeWork ; SearchScreeningEvent].

then annotate given sentences with slots from following slot list, the description of each slot is given.
album: Name of the album that user want to play e.g. Like A Hurricane, The Happy Blues, Subconscious Lobotomy
artist: Name of musical artist mentioned in the sentences e.g. Too Poetic, baro ferret, David Gahan
best_rating: Max rating stars/points of the mentioned book e.g. 6
city: Name of the city request by the user e.g. Nicodemus, Sacaton, Cle Elum
condition_description: Weather condition queried e.g. humidity, storm, blizzard
condition_temperature: Temperature condition when quering weather e.g. colder, hot, warmer
country: Name of the country whose weather is asked e.g. Tanzania, Chad, Montenegro
cuisine: Type of dishes the user asks when booking restaurant e.g. creole, gluten free, turkish
current_location: The specific location of which the weather is asked e.g. here, current position, current spot
entity_name: Name of the song to be added into playlist e.g. A Very Cellular Song, Young at Heart, Recalled to Life
facility: Facility that the requested restaurant need to have e.g. spa, indoor, pool
genre: Genre of the music e.g. Rock Symphonique, Progressive Metal, blues britânico
geographic_poi: Geographic Position of Interest e.g. Tahquamenon Falls State Park, Monument of Lihula, Park Narodowy Brimstone Hill Fortress
location_name: The name of location where the movie schedule is asked  e.g. Santikos Theatres, AMC Theaters, Harkins Theatres
movie_name: Name of the movie e.g. Babar King of the Elephants, The Ghost, Bartok the Magnificent
movie_type: Type of the movie e.g. movies, animated movies, films"
music_item: The type of item that user want to play e.g. album, song, track
object_location_type: The type of location where the movie schedule is asked e.g. cinema, movie theatre, movie house
object_name: The name of the mentioned object  e.g. Wilco Learning How to Die, A Twist in the Tale, The children of Niobe
object_part_of_series_type: Object part of series type e.g. series, chronicle, saga
object_select: Words used to select the object to be rated e.g. this, current, this current
object_type: Type of the object to be rated e.g. novel, book, album
party_size_description: Members that will eat in the Restaurant e.g. my babies and I, my mom and I, me and my grandfather
party_size_number: Number of members that will eat in the Restaurant e.g. 5, 4, five
playlist: Name of the playlist e.g. Flow Español, virales de siempre, laundry
playlist_owner: Owner of the playlist e.g. my, jerry's, beryl's
poi: Position of interests e.g. West Reading, naomi's hostel, Brooklawn
rating_unit: Unit of the rating e.g. stars, points
rating_value: Value of the rating e.g. four, 1, zero
restaurant_name: Name of the restaurant e.g. City Tavern, Oregon Electric Station, Albany Pump Station
restaurant_type: Type of the restaurant e.g. restaurant, pub, brasserie
served_dish: Dish served in requested restaurant e.g. seafood, noodle, hog fry
service: Software service used to play music e.g. Zvooq, Last Fm, Youtube
sort: Sorting criterion used to select item e.g. good, best, top-rated
spatial_relation: Spatial relation indicator of the item e.g. nearby, close by, nearest
state: Name of the state e.g. NE, MA, New Jersey 
timeRange: Time Range e.g. now, five pm, next year
track: Name of the track e.g. All Things Must Pass, Femme Fatale, Have You Met Miss Jones
year: Year Number. e.g. seventies, fifties, thirties

You need to output the annotations in the form of "Intent=INTENT_NAME;Slot1=VALUE1;Slot2=VALUE2;..."
You must not output anything other than the annotations.
You must not miss any possible slot-value pairs

Here is the the sentence:



"""

# testing
with open(args.output_path, 'w', encoding='utf-8') as fout:
    prompt_list = []
    for idx, sentence in enumerate(tqdm(inputs_text)):
        history = [
            HumanMessage(content=prompt+sentence['text'])
        ]
        prompt_list.append(chat.tokenizer.apply_chat_template(history, tokenize=False))

    outputs = chat.chat_complete(prompt_list, use_tqdm=True, temperature=args.temperature)
    for idx, sentence in enumerate(inputs_text):
        sentence['pred_slot'] = outputs[idx]
    json.dump(inputs_text, fout, ensure_ascii=False, indent=2)


