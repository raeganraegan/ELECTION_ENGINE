# Election Engine Overview
Election Engine, made by Alec in 2027, attempts to use Bayesian hierarchical distributions to estimate the outcome of an election.

Calibrated for the NSW 2027 state election, but will attempt to use this in other elections to see how well it copes.

### Current Model Metadata

**Overall name:** ENKI Election Engine

**Version:** V3.0


**Purpose:** Twofold, predicts the primary vote of an election based on historical data and then applies OPV preferences to them.


# Election Engine (ENKI_V3.py) V3 Model Specifications

This document outlines the step-by-step mathematical model and equation pipeline for [ENKI Election Engine V3](/ENKI_V3.py).

---
## Assumptions: 

To be added.

---

#### Parameter & Input Definitions

* **Inputs & Random Variables:**
  * $\mathbf{Y}$ – Historical booth result matrices
  * $\mathbf{S}$ – Dirichlet-aggregated opinion polling vector
  * $\mathbf{z} = \{z_T, z_d, z_\varepsilon\}$ – Standard normal draws ($\mathcal{N}(0,1)$) driving MCMC posterior simulation
  * $\boldsymbol{\theta}$ – Fitted model hyperparameters ($\beta, \rho, \sigma_d, \sigma_\varepsilon, \mu_g, \sigma_g$)

* **Numerator Components (Party Vote Volume):**
  * $\text{Multinomial}_b(\dots)$ – Simulated turnout volume ($V_b$) at booth $b$
  * $(1 - f_k)$ – One Nation (ONP) leakage/flow fraction modifier for party $k$
  * $\text{softmax}_k(\dots)$ – Converts booth latent utility ($\boldsymbol{\eta}_b$) into vote share probability ($p_{kb}$)
  * $m + \alpha_b$ – Global baseline mean $m$ and booth-specific intercept modifier $\alpha_b$
  * $\beta(\ln S - \text{mean}_t \ln S)$ – Polling swing predictor ($x^*$) scaled by party sensitivity $\beta$
  * $(\rho d_T + \sqrt{1-\rho^2}\sigma_d z_d)$ – Autoregressive seat-wide local shift ($d^*$)
  * $\sigma_\varepsilon z_\varepsilon$ – Residual local noise/booth jitter

* **Denominator (Total Seat Turnout Volume):**
  * $\sum_b V_b$ – Sum of all booth turnout volumes across the electorate

---
### Unified ENKI V3 Master Model Equation

The full ENKI V3 primary vote estimation pipeline—combining Bayesian baseline fitting, polling shift dynamics, local autoregressive trends, turnout volume estimation, and One Nation adjustments into a single function—is represented as:

$$
s_k\Big(\mathbf{Y}, \mathbf{S}, \mathbf{z}, \boldsymbol{\theta}\Big) = \frac{\sum_b \text{Multinomial}_b\!\Big(T_T e^{k \mu_g + v k \sigma_g z_T},\, \text{Dirichlet}\big(\kappa \cdot \text{softmax}(a + \tau' \text{early})\big)\!\Big) \cdot \Big[\big(1 - f_k\big) \cdot \text{softmax}_k\Big(m + \alpha_b + \beta \Big(\ln S - \text{mean}_t \ln S\Big) + \big(\rho d_T + \sqrt{1 - \rho^2}\sigma_d z_d\big) + \sigma_\varepsilon z_\varepsilon\Big)\Big]}{\sum_b \text{Multinomial}_b\!\Big(T_T e^{k \mu_g + v k \sigma_g z_T},\, \text{Dirichlet}\big(\kappa \cdot \text{softmax}(a + \tau' \text{early})\big)\!\Big)}
$$

---

## ENKI V3 Step by Step

Because the above equation is ludicriously long, and very hard to parse (even for a model like this) - the following goes through what the script does... step by step (i.e., what steps are mathematically applied to the inputs at each step). This is the same as doing individual components step by step manually and then aggregating them at the end.

### Pipeline Overview

| Step | Input $\rightarrow$ Output | Equations |
| :--- | :--- | :--- |
| **0. Prep** | Raw counts $\rightarrow$ $Y, x$ | Data cleaning & baseline log-ratios |
| **1. Fit** | $Y, x \rightarrow$ Posterior draws $\theta^{(i)}$ | Hierarchical Bayesian parameter fitting & priors |
| **2. Polls** | Poll table $\rightarrow$ $S^{(i)}$ | Weighted polling aggregation & Dirichlet variance |
| **3. Tide** | $\theta, S \rightarrow$ Party-level shift | Polling trend projection & seat-wide tide calculation |
| **4. Booth shares** | $\rightarrow p_b$ | Booth-level latent vote shares |
| **5. Volumes** | $\rightarrow V_b$ | Turnout and volume estimations across booths |
| **6. ONP** | $p_b \rightarrow p'_b$ | Dynamic One Nation (ONP) re-allocation layer |
| **7. Aggregate** | $\rightarrow y$ | Final seat-wide aggregate score and percentiles |

---

## Detailed Model Equations

### 0. Prep

- **Total valid votes:** $n = \text{total} - \text{informal}$
- **Other vote total:** $y_{\text{OTHER}} = n - \sum y_{\text{named}}$
- **Normalized log vote-share predictor:**

$$
x_{kt} = \ln S_{kt} - \text{mean}_t \ln S_{kt}
$$

---

### 1. Model Fitting & Priors

#### Likelihood & Core Equation:

$$
\boldsymbol{y}_{bt} \sim \text{Multinomial}\big(n_{bt},\, \text{softmax}(\boldsymbol{\eta}_{bt})\big)
$$

$$
\eta_{kbt} = m_k + \alpha_{kb} + \beta_k x_{kt} + d_{kt} + \varepsilon_{kbt} \quad \text{where } m_K = 0
$$

#### Prior Distributions:

- **Booth Intercept & Noise Priors:** $ \alpha = \sigma_\alpha z, \quad \varepsilon = \sigma_\varepsilon z $
- **Time-Series / Autoregressive Local Shift Prior:**

$$
d_{k,1} = \sigma_d z, \quad d_{kt} = \rho \, d_{k,t-1} + \sqrt{1 - \rho^2} \, \sigma_d z
$$

- **Coefficient & Parameter Distributions:**

$$
\beta_k \sim \mathcal{N}(\mu_k, \, \text{sd}_k)
$$

$$
\rho \sim \text{Beta}(2, 2)
$$

$$
\sigma_d \sim \text{LogNormal}\left(\ln 0.3 - \frac{1}{2} \ln 2, \, 0.7\right)
$$

---

### 2. Polling Aggregation

- **Recency Decay Weighting:** $w_i = 0.5^{\frac{\text{age}_i}{H}}$
- **Weighted Mean Shift:** $\hat{s} = \frac{\sum w s}{\sum w}$
- **Combined Variance / Standard Error:**

$$
\text{se}^2 = \text{se}_{\text{poll}}^2 + \sigma_{\text{house}}^2 + \gamma^2 \tau
$$

- **Concentration Parameter:**

$$
\text{conc} = \max\!\left(\frac{p(1-p)}{\text{se}^2} - 1, \, 10\right)
$$

- **Dirichlet Polling Prior:**

$$
S \sim \text{Dirichlet}(\text{conc} \cdot \hat{s})
$$

---

### 3. Seat-Wide Tide

* **Shift Calculation:** $x^* = \ln S - c$[cite: 1]
* **Projected Autoregressive Shift:** $d^* = \rho \, d_T + \sqrt{1 - \rho^2} \, \sigma_d z$[cite: 1]
* **Total Linear Party Shift:**
  $$\text{lin}_k = \beta_k x^*_k + d^*_k$$[cite: 1]

---

### 4. Booth Shares

- **Latent Utility / Score:**

$$
\eta_{kb} = m_k + \alpha_{kb} + \text{lin}_k + \sigma_{\varepsilon,k} z
$$

- **Softmax Probability Conversion:**

$$
\boldsymbol{p}_b = \text{softmax}(\boldsymbol{\eta}_b)
$$

---

### 5. Turnout & Volumes

- **Adjusted Expected Turnout:**

$$
T' = T_T \exp(k \mu_g + v k \, \sigma_g z)
$$

- **Early Voting Probability Vector:**

$$
\boldsymbol{\pi} = \text{softmax}(a + \tau' \text{early})
$$

- **Booth Volume Distribution:**

$$
V \sim \text{Multinomial}\big(T', \, \text{Dirichlet}(\kappa \boldsymbol{\pi})\big)
$$

---

### 6. One Nation (ONP) Adjustment Layer

- **BrianMode Allocation:**

$$
p'_k = (1 - f_k)\, p_k, \quad p'_{\text{ONP}} = \sum f_k p_k
$$

- **Alternative Layer:**

$$
s_{\text{ONP}} = \kappa \cdot S_{\text{ONP}}
$$

---

### 7. Final Aggregation

- **Weighted Global Score:**

$$
s_k = \frac{\sum V_b \, p'_{kb}}{\sum V_b}
$$

*Percentiles are subsequently calculated over posterior draws $\theta^{(i)}$.*
# V3 Planned Changes

All my changes are currently geared towards better defining what is going on in ENKI. ENKI predicts the primary vote and so the OPV flows kinda depend on it. It is the foundation of which the models' reality is built on, so its really important that what's going on in ENKI is clear as crystal.

I'm currently working on the following (ticked off if completed):
- [x] **Refactor Helpers/Backtests out of ENKI** to keep the main script clean, to allow for backtesting to occur independently of ENKI. Should make the program as a whole a bit more well designed.
- [ ] **Process Flow Diagram for ENKI**: I am intending on making ENKI spitout a simple [process flow diagram](https://asq.org/quality-resources/flowchart?srsltid=AYCX9vIusrYvqunG4h8_QuA-vRr4EV05yenmEM4xUvJyED7qOLBBL8ri) that highlights each of the steps taken by ENKI in transforming the input data into the output data. The intention is to have a final PDF output for 'logging' purposes where the first page is this process flow diagram linked with the values being generated/inputted on each step. This means that you should be able to trace each step manually via excel from this one PDF alone if you wished - it's an accountability thing.
- [ ] **Laplace Process Implementation for ENKI**:       
  - Firstly, this needs a serious conceptual reality check before I even touch this. Need to know if this is applicable.
  - ENKI being represented as a process flow diagram allows for relative process functions such that we can measure broadly: $U(s)G(s)=Y(s)$ such that we can mathematically see how different inputs in $U(s)$ effect $Y(s)$. I.e., we can see how One Nation's input $U(s)$ effects the primary vote outcome $Y(s)$. This step is totally experimental its just applying some chem eng stuff to this process to see if its useful at all, and we may as well given that the above step is already taken.

Once thats done its a matter of doubling down into INANNA and making that ironclad.

# Why is this readme so short?

I moved the older more detailed readme to [another file](/OlderReadme.md) for now. Another [document](/ModelDescription.pdf) contains more thorough rundown of what i'm doing (though may be a bit older than I'd care to admit). 

Made a new readme, thats shorter with less info and intend to expand it, I kept the old readme just incase you want to dig into it.

# Why so little information on INANNA?

Cos i will get to that in time, rn i need to establish that ENKI is working fine. I partitioned the program like this so I can work on it in distinct parts.