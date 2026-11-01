# Election Engine
Election Engine, made by Alec in 2027, attempts to use Bayesian hierarchical distributions to estimate the outcome of an election.

Calibrated for the NSW 2027 state election, but will attempt to use this in other elections to see how well it copes.

## Current Model Metadata

**Overall name:** ENKI Election Engine
- Idk how I feel about the name now that I think on it? Also it makes no sense given that the engine is in two parts, ENKI and INANNA?
- Subject to change based on my whims.

**Version:** V3.0
- I'll see if I can remember to change the above version num, i'm sure i'll remember to change it for the big changees.

**Purpose:** Twofold, predicts the primary vote of an election based on historical data and then applies OPV preferences to them.
- This models niche is that i'm throwing as many bayesian distributions at this as I can to model uncertainty. This results in really wide bands of probability making it virtually guarunteed that any result is within the 95% CI interval


## Why is this readme so short?

I moved the older more detailed readme to [another file](/OlderReadme.md) for now. Another [document](/ModelDescription.pdf) contains more thorough rundown of what i'm doing (though may be a bit older than I'd care to admit).

All subject to change, the reason this is so short is because I intend to refactor this into a more useful readme eventually so its a bit of a mess.

## V2 Changes

Cba writing here, just go to the older readme.

## V3 Changes

All my changes are currently geared towards better defining what is going on in ENKI. ENKI predicts the primary vote and so the OPV flows kinda depend on it. It is the foundation of which the models' reality is built on, so its really important that what's going on in ENKI is clear as crystal.

I'm currently working on two things to the above end.
- **Process Flow Diagram for ENKI**: I am intending on making ENKI spitout a simple [process flow diagram](https://asq.org/quality-resources/flowchart?srsltid=AYCX9vIusrYvqunG4h8_QuA-vRr4EV05yenmEM4xUvJyED7qOLBBL8ri) that highlights each of the steps taken by ENKI in transforming the input data into the output data. The intention is to have a final PDF output for 'logging' purposes where the first page is this process flow diagram linked with the values being generated/inputted on each step. This means that you should be able to trace each step manually via excel from this one PDF alone if you wished - it's an accountability thing.
- **Laplace Process Implementation for ENKI**: ENKI being represented as a process flow diagram allows for relative process functions such that we can measure broadly: $U(s)G(s)=Y(s)$ such that we can mathematically see how different inputs in $U(s)$ effect $Y(s)$. I.e., we can see how One Nation's input $U(s)$ effects the primary vote outcome $Y(s). This step is totally experimental its just applying some chem eng stuff to this process to see if its useful at all, and we may as well given that the above step is already taken.

Once thats done its a matter of doubling down into INANNA and making that ironclad.