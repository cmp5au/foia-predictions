import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# requires docker setup:
# $ docker run --name mongoserver -p 27016:27017 -v "$PWD":/home/data -d mongo
client = MongoClient('localhost', 27016)
db = client['foia_requests']
foia_data = db['foia_jsons']

foias = list(foia_data.find({'body' : {"$exists": True}}))

print(f"There are {foia_data.count_documents({})} total documents")

statuses = ['rejected', 'fix', 'no_docs', 'partial', 'done']

for status in statuses:
    print(f"There are {len([foia for foia in foias if foia['status'] == status])} "
            f"documents with status '{status}'")

df = pd.read_csv("agency ids.csv")
df.set_index('id', inplace=True)
df = df.append(pd.DataFrame(data=[['Agency not found']],
                       columns=df.columns,
                       index=[0]))

agencies = pd.Series(index=df.index, data=df['name'])

df['count'] = df.index * 0

statuses = ['rejected', 'fix', 'no_docs', 'partial', 'done']

for status in statuses:
    df[status] = df['count']

for result in foias:
    try:
        df.loc[result['agency'], 'count'] += 1
        df.loc[result['agency'], result['status']] += 1
    except KeyError:
        df.loc[0, 'count'] += 1
        df.loc[0, result['status']] += 1

top_agencies = df.sort_values(by='count', ascending=False)[:10]['name']

xs = ['FBI', 'N/A', 'CIA', 'NSA', 'DHS', 'FCC', 'ICE', 'FTC', 'DEA', 'State']
ys = df['count'][top_agencies.index]

fig, ax = plt.subplots()

ax.bar(xs, ys, color='blue')
ax.set_title('Top 10 Agencies by Volume of FOIA Requests')
ax.set_ylabel('Number of Requests')
plt.savefig('../images/requests-by-agency.png')

xs = ['All'] + xs
ys = [[df[status].sum() / df['count'].sum()] + list(df[status][top_agencies.index] / df['count'][top_agencies.index])
      for status in statuses]
colors = ['crimson', 'orange', 'black', 'yellow', 'green']
labels = ['Rejected', 'Fix Required', 'No Relevant Documents', 'Partially Complete', 'Complete']

fig, ax = plt.subplots()

for i in range(len(statuses)):
    ax.bar(xs, ys[i], bottom=np.array(ys[:i]).sum(axis=0), color=colors[i], label=labels[i])

ax.set_title('FOIA Request Outcomes by Agency')
ax.set_ylabel('Percentage of Requests by Status')
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper right')
plt.savefig('../images/results-by-agency.png')

fig, ax = plt.subplots()

dates = sorted([foia['datetime_submitted'][:7] for foia in
             foia_data.find({'datetime_submitted' : {'$exists' : True, '$ne' : None}})])
xs = sorted(list(set(dates)))
ys = [dates.count(x) for x in xs]

ax.plot(xs, ys)
ax.set_title('Request Dates')
ax.set_ylabel('Number of requests per month')
ax.set_xticks(xs[::12] + [xs[-1]])
ax.set_xticklabels([x[:-3] for x in (xs[::12] + [xs[-1]])], fontsize=9)
plt.savefig('../images/requests-by-date.png')