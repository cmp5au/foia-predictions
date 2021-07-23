# Predicting Success of Freedom of Information Act Requests

_________________________

## Table of Contents
_________________________

1. [Background](#background)
2. [Data](#data)
    * [Using the API](#using-the-api)
    * [Preprocessing](#preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Hypothesis Testing](#hypothesis-testing)
    * [Digging Deeper](#digging-deeper)
        * [Two Rooks v. Queen](#two-rooks-v.-queen)
        * [Bishop pair v. Knight pair](#bishop-pair-v.-knight-pair)
        * [Knight & Bishop v. Rook & Pawn](#knight-&-bishop-v.-rook-&-pawn)
5. [Sources and Further Reading](#sources-and-further-reading)

<div align="center">
    <img width="1200" src="./images/foia-crest.png" alt="chmod Options">
</div>

## Background
The Freedom of Information Act (FOIA) was signed into law in 1967, and it requires government agencies to fully or partially disclose previously unreleased information upon request. There are 9 current exemptions that address issues of security and personal rights. It was initially targeted to improve government transparency with businesses and media, but recently law firms and individuals have been the most frequent users. FOIA has been amended, reformed, or expanded 8 times since its inception, and it can be difficult for a private citizen to know whether their request will be accepted or rejected. The goal is to use publicly available past FOIA requests to create a text classifier that can let a potential requester estimate their probability of success before submission.

## Data

### Using the API
MuckRock is a collaborative news site whose goal is to make politics more transparent and democracies more informed. They have a free [API](https://www.muckrock.com/api/) where you can access their FOIA request data programmatically. It is rate-limited at one page (50 requests) per second, and there are 19004 labelled federal FOIA requests, so it will take at least 6 minutes to download the requests. Code to do so is provided in the source folder (src/export_foia_to_mongo.py), but you must edit the credentials.py file to include your personal access token.

### Preprocessing
Each FOIA request contains many communications between requester and agency, but *a priori* our model only sees the initial request, so the code creates a new 'body' field and discards the rest of the communications.

The requests are loaded into MongoDB using the `mongo` Docker image. The body of the request is vectorized with scikit-learn's TfidfVectorizer, and the request agency is added as a feature to the transformed corpus to create our feature matrix.

_________________________

## Exploratory Data Analysis
Thanks to MuckRock's API and our choice of preprocessing, the dataset is clean and complete. Looking into request bodies at random, I saw similar phrasing with around 200-300 words in each request body (though some were longer). I also hunted for interesting FOIA requests, and found fan mail from J. Edgar Hoover, a death threat letter to Roberto Clemente that was opened 2 months <strong>after</strong> the attack was supposed to have happened, and rejected vanity plates:

<br>



<br>
<div align="center">
    <img width="400" src="images/licenses.jpg" alt="chmod Options">
</div>
<br>
<br>



<br>
<div align="center">
    <img width="1200" src="" alt="chmod Options">
</div>
<br>
<br>



_________________________

## Model Fitting

<br>
<div align="center">
    <img width="430" src="" alt="chmod Options">
</div>
<br>
<br>


<br>
<div align="center">
    <img width="600" src="" alt="chmod Options">
</div>
<br>
<br>


<br>
<div align="center">
    <img width="200" src="" alt="chmod Options">
</div>
<br>
<br>

<center>

<br>

| Model               	| F1 Score     	| Accuracy      |
|--------------------	|-----------	|----------     |
| Average Point Value   | 0.551400  	| 0             |
| Sample Size        	| 59,698 	    | 0             |
| p value               | 0 (underflow) | 0             |

<br>
</center>
<br>

_________________________

### Digging Deeper

#### Another thing

<br>
<div align="center">
    <img width="1200" src="" alt="chmod Options">
</div>
<br>
<br>


<br>
<div align="center">
    <img width="1200" src="" alt="chmod Options">
</div>
<br>
<br>



<br>

#### Thing #2


<br>
<div align="center">
    <img width="1200" src="" alt="chmod Options">
</div>
<br>
<br>

<br>

#### Thing #3


<br>
<div align="center">
    <img width="600" height="700" src="" alt="chmod Options">
</div>
<br>
<br>

___________________________________

## Conclusion and Next Steps
Final points

* point 1
* point 2
* point 3
    * subpoint
<br>

This study generates as many questions as it answers:
1. In the case of the 2/3 of games that reach unequal material after the opening, how many feature the side with the material disadvantage winning?
2. Does having 1 bishop v. 1 knight give the side with the bishop more than, less than, or exactly half of the advantage of 2 bishops v. 2 knights?
3. What is the relationship between player rating and the advantage size of these imbalances?
    * Is the effect size of these advantages bigger at the master level?
4. Does having one side of a material imbalance affect the win rate by checkmate vs. the win rate by time forfeit?

<br>

From this small study, I've gained a better understanding of ???

___________________________________

## Sources and Further Reading

1. []()
2. []()
3. []()
4. []()