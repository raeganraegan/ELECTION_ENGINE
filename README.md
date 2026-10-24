# Election Engine
Election Engine, made by Alec in 2027, attempts to use Bayesian hierarchical distributions to estimate the outcome of an election.

Calibrated for the NSW 2027 state election, but will attempt to use this in other elections to see how well it copes.

Currently on version 2 of the model.

## Key Components:

There exists three components to the engine, ENKI, INANNA and Executor.
- ENKI deals with primary vote estimation, and is where most of the mathematics occur.
- INANNA deals with the OPV flows and final estimation therefrom. It is split from ENKI so that in future elections with different electoral systems, I can replace INANNA with whatever system is relevant to that election.
- Executor is as the name suggests - it's the handler for both ENKI and INANNA.

The whole thing requires two explicit inputs, namely: 
- booth data (see _sample_booth_results.csv_ for formatting)
- opinion polling data (see _sample_polls.csv_ for formatting)

You can find relevant data for the above on the internet relatively easily.

It also has some additional params that can be entered in each of the individual scripts (ENKI, INANNA). Relevant params can also be fed to executor via _--argParse_ for certain things.

There are some critical params I still need to factor out as dynamic inputs like number of registered voters, so tread carefully please.

## Modelling Basics (V1)

V1 of this started off with a relatively simple, linear model that went along the lines of:

$$
s(b,t) = m + a_b + y_t + \epsilon_b
$$

**Where:**
* $s(b,t)$ &ndash; The predicted score or outcome for booth $b$ at time $t$
* $m$ &ndash; The baseline global (seatwide) mean
* $a_b$ &ndash; The specific booth-level effect (intercept modifier for booth $b$)
* $y_t$ &ndash; The time-specific trend or effect for year/cycle $t$
* $\epsilon_b$ &ndash; The residual error term for booth $b$, sampling error, local jitter, etc.

With the second equation of: 

$$
y_t = \beta \times S_t + d_t
$$

**Where:**
* $y_t$ &ndash; is term from previous equation
* $\beta$ &ndash; is the sensitivity of an electorate to change for party $p$
* $d_t$ &ndash; is local change (not due to statewide polling shifts etc)
* $S_t$ &ndash; is the change in vote-share of party $p$ for the upcoming election based on polls.

The V1 linear model is really simple and straightforward. You have a baseline for a booth, $m+a_b$, you have the term that is driving the change which is $y_t$, and you have $\epsilon_b$ which represents temporary jitter/noise/effects.

The second equation describes $y_t$ in more detail, indicating that a seat will follow the trends of polling (with it's own sensitivity, since say Labor heartland will be effected less by a negative swing in the polls) as well as other local factors in $d_t$. Local factors could include local scandals from the candidate, incumbency and name recognition, etc.

## Modelling V2

I got tired of having to keep notes on everything so moved it over to a word doc. The word doc is NOT definitive - it is merely meant to establish the changes made from one version to another. The summary of the changes from V1 to V2 was to literally throw as many distributions & their uncertainty into the model and do a MCMC simulation to determine outcomes.

I occasionally mention what distributions are being used in the word doc, but if you're uncertain, just read the code.

I was gonna draw how the code flows but I genuinely cba doing that. The programming is fun, the documentation is not.

## Modelling V2, for dummies:

I have a document that dumbs down what has a distribution applied to it so if you want it let me know. It's an incredibly crude document hence why i'm not putting it here.

## One nation modelling

One nation in the 2027 election is a wildcard only present in the polls. They currently are not running in some seats (esp inner city seats) so they are set as a toggleable thing in ModelConfig. You can set how intense they are, you can set how their votes flow, you can set if they're even there at all. One nation can't be predicted based on existing evidence since there is none for this unprecedented election, it's up to you to determine how the chips fall for them.

## Checking Model Accuracy

The two components of the model (ENKI and INANNA) both have backtesting capabilities, and INANNA has a full-throughput testing for this sorta thing.

Backtesting tries to determine a previous election using all previous election data (i.e., 2019 using 2015, 2011, 2008 data, whatever) and confirms the margin of error on prediction. 

Mostly get between 1-2% backtesting, sometimes higher. Generally the OPV flow has a higher error than the primary vote counter i've found, kinda pointing to how volatile OPV is

## Want to use this?

Feel free, go ahead and fork this. Just idk if this goes somewhere be kind and credit where you got it from :)
