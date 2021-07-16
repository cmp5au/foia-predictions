#!/usr/bin/env python3
# -- coding: utf-8 --

import requests
import utils, time
from math import ceil
from pymongo import MongoClient

url = utils.API_URL
token = utils.get_api_key()
headers = utils.get_headers(token)

client = MongoClient('localhost', 27016)
db = client['foia_requests']
foias = db['foia_jsons']

def make_filter(status_str):
    return f'?user=&title=&status={status_str}&embargo=unknown&jurisdiction=10&agency=&projects=&tags=&has_datetime_submitted=unknown&has_datetime_done=unknown&ordering='

statuses = ['rejected', 'fix', 'no_docs', 'partial', 'done']

for status in statuses:

    page = 1

    next_ = url + "foia/" + make_filter(status)

    while next_ is not None: # Handling at the page level
        r = requests.get(next_, headers=headers)
        try:
            json_data = r.json()
            for request in json_data['results']:
                request['body'] = request['communications'][0]['communication']
                del request['communications']
                foias.insert_one(request)
            foias.insert_one(json_data)
            print('Page %d of %d for %s requests'
                    % (page, ceil(json_data['count'] / 50), status))
            next_ = json_data['next']
            page += 1
            time.sleep(1)
        except:
            print("There was an error of unkown origin")