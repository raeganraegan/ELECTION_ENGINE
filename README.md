# ELECTION ENGINE

This readme is a wip and is definitely not up to convention on how to write a readme, but you'll have to suck it up and just read whats going on.

---
## Project Metadata

#### **Project Name:** 
ENKI Election Engine

#### **Version:** 
V3.1 (Hierarchical Local Slopes Update)

#### **Author:**
Alec, aka [raeganraegan](https://github.com/raeganraegan) on github.

#### **What does this do:**
It aims to predict the primary vote of an electorate seat by booth and aggregate that into boothwide data predicted for the seat. From this, the aim is to then simulate the OPV preference flows present within the NSW state election.

Note, this has been tuned/calibrated to the upcoming [NSW state election](https://elections.nsw.gov.au/elections/state-elections/2027-nsw-state-election). I'm sure it can be tuned to other elections but that's for another version to implement. Aim is to get this version working before this election and to get actionable booth level data from the twofold prediction of this engine. A bold aim indeed.


#### **How this works:**
This is divided into two parts. 
- [x] Part 1: [ENKI](/ENKI_V3.py) Predicts the primary vote of an election based on historical data
- [ ] Part 2: [INANNA](/INANNA_V2.py) Uses above data and applies OPV preferences to them.

So the overall flow is

$$
In \rightarrow ENKI(In) \rightarrow Out_1 \rightarrow INANNA(Out_1) \rightarrow Out_2
$$

The reason this is split is so that I can program the primary vote calculation distinctly from the OPV flow preference vote. It also eventually if I want to, allows me to substitute INANNA for a different voting system.

--- 
# Election Engine (ENKI_V3.py) V3.1 Model Specifications

This part of the document outlines the step-by-step mathematical model and equation pipeline for [ENKI Election Engine V3.1](/ENKI_V3.py). Below begins with some basic assumptions and the basic variables at play. Other variables are should be explained along the way if they're present in the following equations.


## Core Assumptions

* **Booth Continuity:** Polling places maintain consistent catchments over time; historical venue changes are manually merged upfront.
* **Sum-to-One Consistency:** Booth party shares are modeled via softmax/multinomial-logit link functions, guaranteeing all booth and seat totals strictly sum to 100%.
* **Hierarchical Local Swing Elasticity:** Major parties (`ALP`, `LNP`, `GRN`) track statewide polling trends via booth-specific hierarchical sensitivities ($\beta_{kb}$). This allows individual polling catchments to vary in volatility based on historical bottom-up dataset discovery, rather than forcing an arbitrary, top-down uniform seat constraint. `OTHER`/independents remain structurally decoupled from statewide shifts ($\mu_{\beta} = 0.0$).
* **Seat-Wide Local Shift:** Local candidate effects and scandals form an electorate-wide autoregressive trend ($d_t$) that persists moderately ($\text{Beta}(2,2)$) across elections.
* **Poll Decay & Error Budget:** Older polls decay exponentially (2-month half-life). Total polling uncertainty combines sample variance, a calibrated 1.50 pt house-effect, and time-horizon decay ($\gamma \sqrt{\Delta t}$).
* **Vote Volume Conservation:** Booth volumes model voter shift toward early/postal voting while conserving overall electorate turnout.
* **Wildcard Entry (ONP):** One Nation vote shifts cannot be inferred from standard history, relying instead on explicit transfer rules (e.g., *BrianMode* taking 20% LNP / 5% ALP).



### Parameter & Input Definitions

* **Inputs & Random Variables:**
  * $\mathbf{Y}$ – Historical booth result matrices
  * $\mathbf{S}$ – Dirichlet-aggregated opinion polling vector
  * $\mathbf{z} = \{z_T, z_d, z_\varepsilon, z_\beta\}$ – Standard normal draws ($\mathcal{N}(0,1)$) driving MCMC posterior simulation
  * $\boldsymbol{\theta}$ – Fitted model hyperparameters ($\mu_\beta, \sigma_\beta, \rho, \sigma_d, \sigma_\varepsilon, \mu_g, \sigma_g$)

* **Numerator Components (Party Vote Volume):**
  * $\text{Multinomial}_b(\dots)$ – Simulated turnout volume ($V_b$) at booth $b$
  * $(1 - f_k)$ – One Nation (ONP) leakage/flow fraction modifier for party $k$
  * $\text{softmax}_k(\dots)$ – Converts booth latent utility ($\boldsymbol{\eta}_b$) into vote share probability ($p_{kb}$)
  * $m + \alpha_b$ – Global baseline mean $m$ and booth-specific intercept modifier $\alpha_b$
  * $\beta_{kb}(\ln S - \text{mean}_t \ln S)$ – Localized polling swing predictor scaled by booth-specific sensitivity coefficient $\beta_{kb}$
  * $(\rho d_T + \sqrt{1-\rho^2}\sigma_d z_d)$ – Autoregressive seat-wide local shift ($d^*$)
  * $\sigma_\varepsilon z_\varepsilon$ – Residual local noise/booth jitter

* **Denominator (Total Seat Turnout Volume):**
  * $\sum_b V_b$ – Sum of all booth turnout volumes across the electorate

* **Turnout & Volume Parameters:**
  * $T_T$ – Baseline total seat turnout volume
  * $a$ – Intercept vector for early voting proportion calculations
  * $\tau'$ – Coefficient parameter for early voting accessibility
  * $\kappa$ – Dirichlet concentration hyperparameter
  * `early` – Local booth indicator metric representing early voting accessibility or historical proportion

* **Polling Decay & Variance Parameters:**
  * $H$ – Recency decay half-life parameter (set to 2 months)
  * $\gamma$ – Time-decay variance factor for the polling error budget (calibrated to 1.5)
  * $\tau$ – Elapsed time interval from the polling date

---
### Unified ENKI V3.1 Master Model Equation

The full ENKI V3.1 primary vote estimation pipeline—combining Bayesian baseline fitting, decentralized hierarchical polling shifts, local autoregressive trends, turnout volume estimation, and One Nation adjustments into a single function—is represented below:


#### Abstracted Model

$$
\text{Seat Primary Vote} = \text{Aggregate}\left(\text{Volumes}(\text{Enrolment}, \text{Turnout}) \times \Big[\text{Bayesian Fit}(\text{History}, \text{Priors}) + \text{Decentralized Local Tide}(\text{Polls}) + \text{Wildcards}(\text{ONP})\Big]\right)
$$

#### Mathematical Model

$$
s_k\Big(\mathbf{Y}, \mathbf{S}, \mathbf{z}, \boldsymbol{\theta}\Big) = \frac{\sum_b \text{Multinomial}_b\!\Big(T_T e^{k \mu_g + v k \sigma_g z_T},\, \text{Dirichlet}\big(\kappa \cdot \text{softmax}(a + \tau' \text{early})\big)\!\Big) \cdot \Big[\big(1 - f_k\big) \cdot \text{softmax}_k\Big(m + \alpha_b + \beta_{kb} \Big(\ln S - \text{mean}_t \ln S\Big) + \big(\rho d_T + \sqrt{1 - \rho^2}\sigma_d z_d\big) + \sigma_\varepsilon z_\varepsilon\Big)\Big]}{\sum_b \text{Multinomial}_b\!\Big(T_T e^{k \mu_g + v k \sigma_g z_T},\, \text{Dirichlet}\big(\kappa \cdot \text{softmax}(a + \tau' \text{early})\big)\!\Big)}
$$


---
## ENKI V3.1 Step by Step

### Pipeline Overview

| Step | Input $\rightarrow$ Output | Equations |
| :--- | :--- | :--- |
| **0. Prep** | Raw counts $\rightarrow$ $Y, x$ | Data cleaning & baseline log-ratios |
| **1. Fit** | $Y, x \rightarrow$ Posterior draws $\theta^{(i)}$ | Non-centered hierarchical parameter fitting & random slopes |
| **2. Polls** | Poll table $\rightarrow$ $S^{(i)}$ | Weighted polling aggregation & Dirichlet variance |
| **3. Tide** | $\theta, S \rightarrow$ Localized booth shift | Decentralized polling trend projection & booth-level tide calculation |
| **4. Booth shares** | $\rightarrow p_b$ | Booth-level latent vote shares |
| **5. Volumes** | $\rightarrow V_b$ | Turnout and volume estimations across booths |
| **6. ONP** | $p_b \rightarrow p'_b$ | Dynamic One Nation (ONP) re-allocation layer |
| **7. Aggregate** | $\rightarrow y$ | Final seat-wide aggregate score and percentiles |

---

## Detailed Model Equations

### 0. Prep

* **What it does:** Standardizes raw booth vote counts into usable log-ratio formats, filters informal ballots, aggregates minor third parties into the `OTHER` category, and calculates historic log-poll baseline averages.
* **Why we do it:** Raw vote counts cannot be compared directly across booths with varying turnout sizes. Transforming vote shares into centered log-ratios prepares the data for multinomial logit modeling and removes scale distortions across elections.

- **Total valid votes:** $n = \text{total} - \text{informal}$
- **Other vote total:** $y_{\text{OTHER}} = n - \sum y_{\text{named}}$
- **Normalized log vote-share predictor:**

$$
x_{kt} = \ln S_{kt} - \text{mean}_t \ln S_{kt}
$$

---

### 1. Model Fitting & Priors

* **What it does:** Fits a hierarchical Bayesian model to past election data using MCMC, estimating booth-level baseline intercepts ($\alpha_{kb}$), historical electorate shifts ($d_{kt}$), localized booth-level polling sensitivities ($\beta_{kb}$), and variance parameters.
* **Why we do it:** Polling alone is too coarse to capture hyper-local booth behavior. Fitting this model extracts every booth's underlying partisan baseline and models random slopes to discover how uniquely each neighborhood reacts to broader state-wide tides.

#### Likelihood & Core Equation:

$$
\boldsymbol{y}_{bt} \sim \text{Multinomial}\big(n_{bt},\, \text{softmax}(\boldsymbol{\eta}_{bt})\big)
$$

$$
\eta_{kbt} = m_k + \alpha_{kb} + \beta_{kb} x_{kt} + d_{kt} + \varepsilon_{kbt} \quad \text{where } m_K = 0
$$

#### Hierarchical Prior Distributions (Non-Centered Parameterization):

- **Hierarchical Random Slopes (Sensitivity Prior):**
$$
\mu_{\beta,k} \sim \mathcal{N}(\text{Preset Mean},\, \text{Preset SD}), \quad \sigma_{\beta,k} \sim \text{HalfNormal}(0.20)
$$
$$
\beta_{kb} = \mu_{\beta,k} + \sigma_{\beta,k} \cdot z_{\beta,b} \quad \text{where } z_{\beta,b} \sim \mathcal{N}(0,1)
$$
- **Booth Intercept & Noise Priors:** $\alpha = \sigma_\alpha z, \quad \varepsilon = \sigma_\varepsilon z$
- **Time-Series / Autoregressive Local Shift Prior:**

$$
d_{k,1} = \sigma_d z, \quad d_{kt} = \rho \, d_{k,t-1} + \sqrt{1 - \rho^2} \, \sigma_d z
$$

- **Hyperparameter Coefficient Distributions:**

$$
\rho \sim \text{Beta}(2, 2)
$$

$$
\sigma_d \sim \text{LogNormal}\left(\ln 0.3 - \frac{1}{2} \ln 2,\, 0.7\right)
$$

---

### 2. Polling Aggregation

* **What it does:** Aggregates published statewide opinion polls into a unified Dirichlet prior distribution, weighting recent polls higher and inflating variance for house effects and time decay.
* **Why we do it:** Individual polls contain sample noise, house bias, and decay over time. Aggregating them via a Dirichlet distribution provides a stable, probabilistic picture of statewide party momentum leading up to election day.

- **Recency Decay Weighting:** $w_i = 0.5^{\frac{\text{age}_i}{H}}$
- **Weighted Mean Shift:** $\hat{s} = \frac{\sum w s}{\sum w}$
- **Combined Variance / Standard Error:**

$$
\text{se}^2 = \text{se}_{\text{poll}}^2 + \sigma_{\text{house}}^2 + \gamma^2 \tau
$$

- **Concentration Parameter:**

$$
\text{conc} = \max\!\left(\frac{p(1-p)}{\text{se}^2} - 1,\, 10\right)
$$

- **Dirichlet Polling Prior:**

$$
S \sim \text{Dirichlet}(\text{conc} \cdot \hat{s})
$$

---

### 3. Decentralized Local Tide

* **What it does:** Combines the statewide polling swing ($x^*$) with the seat's persistent local autoregressive trend ($d^*$) to produce a decentralized linear party shift ($\text{lin}_{kb}$) unique to each booth.
* **Why we do it:** Election trends are two-dimensional: statewide political tides push major parties up or down, but individual neighborhoods absorb or resist that swing at drastically different rates depending on local elasticity.

- **Shift Calculation:** $x^* = \ln S - \text{mean}_t \ln S$
- **Projected Autoregressive Shift:** $d^* = \rho \, d_T + \sqrt{1 - \rho^2} \, \sigma_d z_d$
- **Localized Linear Party Shift:**

$$
\text{lin}_{kb} = \beta_{kb} x^{*}_{k} + d^{*}_{k}
$$

---

### 4. Booth Shares

* **What it does:** Computes booth-specific latent utility scores ($\eta_{kb}$) by combining baseline intercepts, decentralized linear booth shifts, and local noise, then converts them to probabilities via a softmax function.
* **Why we do it:** Translates abstract seat-wide trends and booth baselines into explicit party primary vote probabilities ($\boldsymbol{p}_b$) that are mathematically guaranteed to sum to 100% at every single polling place.

- **Latent Utility / Score:**

$$
\eta_{kb} = m_k + \alpha_{kb} + \text{lin}_{kb} + \sigma_{\varepsilon,k} z
$$

- **Softmax Probability Conversion:**

$$
\boldsymbol{p}_b = \text{softmax}(\boldsymbol{\eta}_b)
$$

---

### 5. Turnout & Volumes

* **What it does:** Simulates turnout volume ($V_b$) across all physical and early/pre-poll voting centers using multinomial and Dirichlet distributions.
* **Why we do it:** Primary vote percentages must be weighted by actual voter numbers. Since early and postal voting proportions fluctuate significantly, modeling turnout volumes ensures larger booths and pre-poll centers correctly dominate the electorate total.

- **Adjusted Expected Turnout:**

$$
T' = T_T \exp\left(k \mu_g + v k \, \sigma_g z_T\right)
$$

- **Early Voting Probability Vector:**

$$
\boldsymbol{\pi} = \text{softmax}(a + \tau' \text{early})
$$

- **Booth Volume Distribution:**

$$
V \sim \text{Multinomial}\big(T',\, \text{Dirichlet}(\kappa \boldsymbol{\pi})\big)
$$

---

### 6. One Nation (ONP) Adjustment Layer

* **What it does:** Applies explicit vote transfer rules (e.g., *BrianMode*) to reallocate vote shares from major parties (`LNP`, `ALP`) to One Nation when ONP enters a contest.
* **Why we do it:** Third-party wildcards like One Nation run sporadically and cannot be predicted purely from historical trends. Explicit transfer rules allow manual or scenario-based modeling of wildcard candidate entries.

- **BrianMode Allocation:**

$$
p'_k = (1 - f_k)\, p_k, \quad p'_{\text{ONP}} = \sum f_k p_k
- **Alternative Layer:**

$$
s_{\text{ONP}} = \kappa \cdot S_{\text{ONP}}
$$

---

### 7. Final Aggregation

* **What it does:** Multiplies booth-level party shares ($p'_{kb}$) by booth turnout volumes ($V_b$), sums them across all booths in the electorate, and divides by total electorate turnout to output final seat-wide primary vote predictions and uncertainty intervals.
* **Why we do it:** Aggregates booth-level granular predictions into the final electorate primary vote distribution, providing point estimates and confidence intervals for ENKI's primary vote output.

- **Weighted Global Score:**

$$
s_k = \frac{\sum V_b \, p'_{kb}}{\sum V_b}
$$

*Percentiles are subsequently calculated over posterior draws $\theta^{(i)}$.*

---
# What does this output?

Other than the boring ah excel spreadsheets you can sift through after each generation, ENKI generates beautiful plots like such (example only):

![image1](/Showcase/image1.png)

![image2](/Showcase/image2.png)

![image3](/Showcase/image3.png)

As should be somewhat self evident, the program produces all the probability distributions for election volume, primary vote share by party, the electoral tide based on polling data, as well as other local parameters that are evident. 

---
# V3 Planned Changes

All my changes are currently geared towards better defining what is going on in ENKI. ENKI predicts the primary vote and so the OPV flows kinda depend on it. It is the foundation of which the models' reality is built on, so its really important that what's going on in ENKI is clear as crystal.

I'm currently working on the following (ticked off if completed):
- [x] **Refactor Helpers/Backtests out of ENKI** to keep the main script clean, to allow for backtesting to occur independently of ENKI. Should make the program as a whole a bit more well designed.
- [x] **Implement Hierarchical Booth Random Slopes ($\beta_{kb}$)** to eliminate arbitrary top-down uniform assumptions and let the sampler discover local swing elasticity natively.
- [ ] **Process Flow Diagram for ENKI**: I am intending on making ENKI spitout a simple [process flow diagram](https://asq.org/quality-resources/flowchart?srsltid=AYCX9vIusrYvqunG4h8_QuA-vRr4EV05yenmEM4xUvJyED7qOLBBL8ri) that highlights each of the steps taken by ENKI in transforming the input data into the output data. The intention is to have a final PDF output for 'logging' purposes where the first page is this process flow diagram linked with the values being generated/inputted on each step. This means that you should be able to trace each step manually via excel from this one PDF alone if you wished - it's an accountability thing.
- [ ] **Laplace Process Implementation for ENKI**:       
  - Firstly, this needs a serious conceptual reality check before I even touch this. Need to know if this is applicable.
  - ENKI being represented as a process flow diagram allows for relative process functions such that we can measure broadly: U(s)G(s)=Y(s) such that we can mathematically see how different inputs in U(s)G(s) effect Y(s). I.e., we can see how One Nation's input U(s) effects the primary vote outcome Y(s). This step is totally experimental its just applying some chem eng stuff to this process to see if its useful at all, and we may as well given that the above step is already taken.

Once thats done its a matter of doubling down into INANNA and making that ironclad.


# Why so little information on INANNA?

Cos i will get to that in time, rn i need to establish that ENKI is working fine. I partitioned the program like this so I can work on it in distinct parts.
