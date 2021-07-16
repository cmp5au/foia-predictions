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
    <img width="1200" src="./images/foia-crest.jpg" alt="chmod Options">
</div>

## Background
Chess is a two-person strategy game where each player attempts to coordinate their pieces -- pawns, knights, bishops, rooks, queen, and king -- to attack the opposing king and prevent its escape, resulting in checkmate and the end of the game. A win for White is recorded as 1-0, denoting 1 point for White and 0 points for Black. Similarly, a win for Black is 0-1, and a draw is 0.5-0.5. I will use the term <strong>point value</strong> to refer to the overall result of the game. Each of the pieces has a relative <strong>material value</strong>, commonly given as:

<br>

| Piece  | Value |
|  :-:   | :-: |
| Pawn   | 1 |
| Knight (N) | 3 |
| Bishop (B) | 3 |
| Rook (R)   | 5 |
| Queen (Q)  | 9 |

<br>

Based on this table, there are material exchanges that are nominally equal but create an imbalance in the game: bishop for knight, 2 bishops for 2 knights, bishop and knight for rook and pawn, and queen for two rooks, for example. Chess masters, teachers, and coaches generally prefer one side of these exchanges over the other despite the seemingly equal trade. For example, masters <strong>strongly</strong> prefer to have a pair of bishops over a pair of knights, a knight and bishop over a rook and pawn, and 2 rooks over a queen. That being said, the size of the advantage always depends on the state of the chessboard.

My goal is to answer the following questions:
1. Does the material imbalance confer an advantage, even to amateur players?
2. Can we isolate the characteristics that make one side of the imbalance favorable?

## Data

### Raw Dataset

The online free chess website [lichess.org](http://www.lichess.org/) features a database of every game ever played from January 2013 to May 2021. The data is in Portable Game Notation (PGN) format, as shown in the game below.

<br>

```text
[Event "Rated Blitz game"]
[Site "https://lichess.org/yfq9u4fm"]
[Date "????.??.??"]
[Round "?"]
[White "tiggran"]
[Black "treehugger"]
[Result "1-0"]
[WhiteElo "1599"]
[BlackElo "1487"]
[ECO "B00"]
[Opening "Owen Defense"]
[TimeControl "300+0"]
[UTCDate "2013.01.01"]
[UTCTime "04:13:22"]
[Termination "Normal"]
[WhiteRatingDiff "+8"]
[BlackRatingDiff "-8"]

1. e4 b6 2. Nf3 Bb7 3. Nc3 e6 4. d3 Bb4 5. Bd2 Bxc3 6. Bxc3 Ne7 7. Be2 f6
8. e5 Ng6 9. exf6 gxf6 10. Qd2 Bxf3 11. Bxf3 c6 12. O-O-O Qe7 13. Rhe1 e5
14. Bd4 c5 15. Bc3 Nc6 16. Bxc6 dxc6 17. Qh6 Rg8 18. g3 Qg7 19. Qh3 Ke7 20.
d4 Rad8 21. dxe5 Rxd1+ 22. Rxd1 fxe5 23. Rd7+ 1-0
```

<br>

This features metadata of the game followed by a list of moves in modern algebraic chess notation that is difficult to read and requires parsing. In this example, it isn't immediately clear that this game features two amateur players that reach a bishop pair vs. knight pair position that White converted into a victory by resignation.

The unabridged raw dataset of over 2.2 billion games is available at [database.lichess.org](https://database.lichess.org/). Each month's games are available to download as bzip2 files. Of these, about 15 million were played in 2013 or 2014.

### Refining the Dataset

In blitz chess, there is an added component of a clock that can cause a player to lose if they run out of time. This often results in a player making ill-considered and suboptimal moves. Blitz is also the preferred method of many players online, so for the sake of consistency and applicability I limited my dataset to blitz games (3 - 5 minutes per player) of amateur players (lichess.org rating between 1200 and 2000), played in 2013 or 2014. My dataset is the 5,261,428 games that fit this criteria. In order to process the large data volume, I used the open source command line tool [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) to match on time control and player blitz rating.

_________________________

## Exploratory Data Analysis
The dataset was very clean, with no incomplete games, illegal moves, or null / missing tags. There was only one game that required further examination: the PGN featured a checkmate by White, but showed Black winning the game by time forfeit. After examining the game in depth, I was able to determine that White had pre-moved (a move request that doesn't get executed until the opponent makes their move) the checkmate move but didn't have enough time on their clock to register the move on the lichess server, which takes about 5ms. Therefore, it was correctly marked as a win by Black by time forfeit.

<br>

After checking for missing information, the first thing I looked at was win rate for White and Black vs. draw rate:

<br>
<div align="center">
    <img width="400" src="./images/win_loss_draw_bar.png" alt="chmod Options">
</div>
<br>
<br>

We see that the games are usually decisive; rarely drawn. There is also evidence of White's first move advantage showing in a win rate 5% higher than Black's. From here, I was curious to see how many games featured completely equal material balance <strong>after the opening</strong>, which I defined as each side having lost at least 3 points of material and the position being stable to a depth of 3 moves.

<br>
<div align="center">
    <img width="1200" src="./images/total_games_bar.png" alt="chmod Options">
</div>
<br>
<br>

A shockingly large number of amateur blitz games feature one side winning a pawn or more out of the opening, about 2 in 3 games. Some of this could be from <strong>gambits</strong> -- sacrificing a pawn for the ability to quickly bring pieces into play -- but these too can equalize after the gambiteer converts their positional advantage into winning back material.

_________________________

## Hypothesis Testing
At this point, the games featuring the interesting material imbalances have been isolated and it seems reasonable to ask: is there any advantage for the amateur player to have two bishops over two knights, a knight and bishop over a rook and pawn, or two rooks over a queen?

The metric I chose to answer this question is the average point value:

<br>
<div align="center">
    <img width="430" src="./images/avg_pt_val_def.png" alt="chmod Options">
</div>
<br>
<br>

This can be shown to be equivalent to using the difference between win rate and loss rate:

<br>
<div align="center">
    <img width="600" src="./images/winr_lossr.png" alt="chmod Options">
</div>
<br>
<br>

Because average point value is the weighted mean of proportions that sum to 1, it also behaves as a proportion and allows us to perform a simple 1-sample test of proportions. <strong>The null hypothesis in each case is that the average point value is 0.5 (win rate = loss rate), and the alternative hypothesis is that the average point value is not 0.5, giving us a two-tailed test.</strong> Our alpha value -- p-value threshold for rejecting the null hypothesis -- is 5%, and after applying the Bonferroni correction to the 3 simultaneous tests we have an effective rejection threshold of 1.66667%. 

<br>
<div align="center">
    <img width="200" src="./images/hypotheses.png" alt="chmod Options">
</div>
<br>
<br>

<center>

Sample 1 (Bishop pair v. Knight pair):
<br>

| Statistic          	| Value     	|
|--------------------	|-----------	|
| Average Point Value   | 0.551400  	|
| Sample Size        	| 59,698 	    |
| p value               | 0 (underflow) |

<br>
<br>

Sample 2 (Knight & Bishop v. Rook & Pawn):
<br>

| Statistic          	| Value     	|
|--------------------	|-----------	|
| Average Point Value   | 0.553601  	|
| Sample Size        	| 24,953 	    |
| p value               | 0 (underflow) |

<br>
<br>

Sample 3 (Two Rooks v. Queen):
<br>

| Statistic          	| Value     	|
|--------------------	|-----------	|
| Average Point Value   | 0.508331  	|
| Sample Size        	| 4,441 	    |
| p value               | 0.266747      |

</center>
<br>

While each of these cases had an effect direction that corresponded to masters' preferences, only the Bishop pair v. Knight pair and Knight & Bishop v. Rook & Pawn had a statistically significant effect. We choose to reject the null hypothesis for these first two cases, but fail to reject the null hypothesis for the final case of Two Rooks v. Queen.
_________________________

### Digging Deeper

#### Two Rooks v. Queen

Let's start with the most difficult case of Two Rooks v. Queen, the sample with by far the smallest effect size and smallest sample size. The main imbalance here is that the two rooks are better at coordinating to attack a single target by providing more attackers than the defenders, while the queen is more mobile and can attack both sides of the board at once. To examine this in depth, I grouped the games into 9 buckets based on the number of pawns left on the board when the imbalance appeared, 0 to 8 pawns. The buckets for 0 pawns and 8 pawns were extremely small, so I lumped them in with their neighboring bucket to get a final 7 buckets: "1 or fewer" to "7 or more" pawns. I then examined the win rate, draw rate, and loss rate of each of these buckets:

<br>
<div align="center">
    <img width="1200" src="./images/qv2r_by_pawns.png" alt="chmod Options">
</div>
<br>
<br>

Some quick observations: the fewer pawns on the board the more likely a draw is to occur, and overall having fewer pawns on the board seems to favor the side with the 2 rooks. The "Average point result" line is clearly trending downwards, but unfortunately our dataset is <strong>heteroskedastic</strong> -- the variance of the residuals is not constant -- because our bucket size is not constant, which violates the assumptions of linear regression and prevents us from computing a p-value in this way. However, we can still use the nonparametric method of Spearman rank-order correlation:

<br>
<div align="center">
    <img width="1200" src="./images/qv2r_by_pawns_best_fit_with_spearmanr.png" alt="chmod Options">
</div>
<br>
<br>

This shows a statistically significant monotonic relationship between the number of pawns remaining on each side and the average point result of a Queen v. Two Rooks imbalance. The takeaway: <strong>for amateur players, it is worth trading your queen for two rooks in an equal-material situation if and only if you have fewer than 4 pawns on the board.</strong>

<br>

#### Bishop pair v. Knight pair
Next we can look at Bishop pair v. Knight pair, which has the largest sample size and a strong effect. With many ways of splitting the dataset (with or without queens, number of pawns on the board) this effect persists, but one way of justifying the advantage is that the bishops are more mobile than the knights and can attack both sides of the board at once. In this light, the advantage of the bishops should be maximized when there is more material on the board and minimized in an endgame. To visualize this the bishop pair v. knight pair games were placed into 28 buckets based on total material value on the board after the imbalance appeared. Many of the buckets corresponding to lower material value were small and similarly-behaved, so they were grouped into a single "28 or fewer" bucket for a cleaner visual:

<br>
<div align="center">
    <img width="1200" src="./images/bbnn_by_material.png" alt="chmod Options">
</div>
<br>
<br>

Win rate and loss rate are mostly reflections of each other, but again they both decrease in the limit of very little material remaining which indicates an increase in draw rate. Interestingly, the advantage of the bishop pair seems to disappear when the material remaining reaches 29 (58 total). This is still most of the material on the board, so the takeaway in this case is that <strong>for amateur players, the bishop pair advantage is worthwhile in the opening when almost all material remains on the board. Otherwise, the imbalance doesn't provide a signficiant advantage to either side.</strong>

<br>

#### Bishop & Knight v. Rook & Pawn

Lastly the Bishop & Knight v. Rook & Pawn had the largest effect size. The side with the two minor pieces has more pieces available for attack, so I believe the presence of the strongest attacking piece -- the queen -- should make a large difference in the evaluation of this imbalance:

<br>
<div align="center">
    <img width="600" height="700" src="./images/nbvrp_by_queens.png" alt="chmod Options">
</div>
<br>
<br>

As we have seen before with a decrease in material, win rate decreases and draw rate increases. However, this graph also shows loss rate increasing when the queens are traded which indicates a loss of advantage. Visually the 50% mark is close to the middle of the "Queens off" bar, and this confirms the loss of advantage when the queens are traded. The remaining advantage may be statistically significant but it is practically insignificant. The takeaway: <strong>for amateur players, a knight and bishop are better than a rook and pawn, and the side with the two minor pieces should avoid trading queens while the side with the rook and pawn should try to trade queens.</strong>

___________________________________

## Conclusion
Amateur players are not as adept at maximizing advantages as chess experts and masters, and playing blitz chess limits this ability even further. However, there was still a clear and significant advantage in certain circumstances:

* The bishop pair is stronger than the knight pair with all (or nearly all) other material still remaining
* Two minor pieces are stronger than a rook and pawn before the queens are traded
* Two rooks are stronger than a queen if there are fewer than 4 pawns remaining on each side
    * The queen is stronger than the rooks in the rarer case of more than 5 pawns remaining on each side

<br>

This study generates as many questions as it answers:
1. In the case of the 2/3 of games that reach unequal material after the opening, how many feature the side with the material disadvantage winning?
2. Does having 1 bishop v. 1 knight give the side with the bishop more than, less than, or exactly half of the advantage of 2 bishops v. 2 knights?
3. What is the relationship between player rating and the advantage size of these imbalances?
    * Is the effect size of these advantages bigger at the master level?
4. Does having one side of a material imbalance affect the win rate by checkmate vs. the win rate by time forfeit?

<br>

From this small study, I've gained a better understanding of the practical dynamics of material imbalances at the amateur level. I'm well-equipped with the tools I've developed here to answer these remaining questions in a future study.

___________________________________

## Sources and Further Reading

1. [Lichess database](https://database.lichess.org/)
2. [pgn-extract documentation](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
3. ["The Evaluation of Material Imbalances" by International Master Larry Kaufman](https://www.chess.com/article/view/the-evaluation-of-material-imbalances-by-im-larry-kaufman)
4. ["Beware the Bishop Pair" by Mark Sturman](http://www.chesscomputeruk.com/Computer_Chess_Reports_Vol5_No2_-pgs_37-72-.pdf)