---
bibliography: ./paper.bib
title: "Climate-Driven Doubling of U.S. Maize Loss Probability: Interactive Simulation with Neural Network Monte Carlo"
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
date: 2024-12-05
affiliations:
  - id: 1
    name: Eric and Wendy Schmidt Center for Data Science and Environment, University of California Berkeley, Berkeley 94720, CA, USA
  - id: 2
    name: Department of Agricultural Economics and Agribusiness, University of Arkansas, Fayetteville 72701, AR, USA
  - id: 3
    name: Department of Environmental Science, Policy & Management, University of California Berkeley, Berkeley 94720, CA, USA
author:
  - name: A Samuel Pottinger \orcidlink{0000-0002-0458-4985}
    affil-id: 1
    correspondence: yes
    email: sam.pottinger@berkeley.edu
  - name: Lawson Connor \orcidlink{0000-0001-5951-5752}
    affil-id: 2
  - name: Brookie Guzder-Williams \orcidlink{0000-0001-6855-8260}
    affil-id: 1
  - name: Maya Weltman-Fahs
    affil-id: 1
  - name: Nick Gondek \orcidlink{0009-0007-7431-4669}
    affil-id: 1
  - name: Timothy Bowles \orcidlink{0000-0002-4840-3787}
    affil-id: 3
output:
  pdf_document:
    number_sections: yes
    template: default.tex
---

**Abstract:** Climate change not only threatens agricultural producers but also strains related public agencies and financial institutions. These important food system actors include government entities tasked with insuring grower livelihoods and supporting response to continued global warming. We examine future risk within the U.S. Corn Belt geographic region for one such crucial institution: the U.S. Federal Crop Insurance Program. Specifically, we predict the impacts of climate-driven crop loss at a policy-salient approximated "risk unit" scale. Built through our presented neural network Monte Carlo method, these simulations anticipate both more frequent and more severe losses in both near and medium-term projections. Futhermore, these concerning changes would result in a financially onerous doubling of the annual probability of maize Yield Protection insurance claims at mid-century. We also provide an open source pipeline and interactive visualization tools to further explore these results with configurable statistical treatments. Altogether, we fill an important gap in current understanding for climate adaptation by bridging existing historic yield estimation and climate projection to predict crop loss metrics at policy-relevant granularity.

# Introduction
Public institutions such as government-supported crop insurance play an important role in agricultural stability across much of the world [@mahul_government_2010]. We add to existing work regarding global warming impacts to these essential food systems actors [@diffenbaugh_historical_2021] by examining the U.S. Federal Crop Insurance Program inside the U.S. Corn Belt geographic region. More specifically, we build upon prior climate projections [@williams_high_2024] and remote sensing yield estimations [@lobell_scalable_2015] to generate maize loss projections through Monte Carlo simulations on top of neural network regressors. As further described below, this enables prediction of future insurance claims rates under climate change at an institutionally-relevant spatial scale. These results may help inform climate adaptation efforts.

## Background
Global warming threatens production of key staple crops, including maize [@rezaei_climate_2023]. Climate variability already drives a substantial proportion of year-to-year crop yield variation [@ray_climate_2015] and continued climate change may reduce planet-wide maize yields by up to 24% by the end of this century [@jagermeyr_climate_2021]. The growing frequency and severity of stressful weather conditions [@dai_increasing_2013] to which maize is increasingly susceptible [@lobell_changes_2020] pose not only a threat to farmers' revenue [@sajid_extreme_2023] but also strain the institutions established to safeguard those producers [@hanrahan_crop_2024]. These important organizations are also often tasked with supporting the food system through evolving growing conditions and the impacts of climate change [@rma_climate_2022].

Within this context, the United States of America is the world's largest maize producer and exporter [@ates_feed_2023]. Its government-backed Federal Crop Insurance Program covers a large share of this growing risk [@tsiboe_crop_2023]. The costs of crop insurance in the U.S. have already increased by 500% since the early 2000s with annual indemnities reaching $19B in 2022 [@schechinger_crop_2023]. Furthermore, retrospective analysis attributes 19% of "national-level crop insurance losses" between 1991 and 2017 to climate warming, an estimate rising to 47% during the drought-stricken 2012 growing season [@diffenbaugh_historical_2021]. Looking forward, @li_impact_2022 show progressively higher U.S. maize loss rates as warming elevates.

## Prior work
Modeling possible changes in frequency and severity of crop loss events that trigger indemnity claims is an important step to prepare for the future impacts of global warming. Related studies have predicted changes in crop yields at broad scales such as county-level aggregation [@leng_predicting_2020] and have estimated climate change impacts to U.S. maize within whole-sector or whole-economy analysis [@hsiang_estimating_2017]. These efforts include traditional statistical models [@lobell_statistical_2010] as well as an increasing body of work favoring machine learning approaches [@leng_predicting_2020]. Finally, the literature also considers how grower practices intersect with insurance [@connor_crop_2022; @wang_warming_2021; @chemeris_insurance_2022] and resilience [@renwick_long-term_2021; @manski_diversified_2024].

Despite these prior contributions, important programs and policies often include producer-level elements [@rma_crop_2008] and previous studies often do not include more granular models of insurer-focused loss severity at policy-relevant spatial scales. Of particular interest, the "risk" or "insured" unit refers to a set of agricultural fields that are insured together within an individual policy [@fcic_common_2020]. While modeling at this producer level may provide important institutional insight, many prior studies either do not offer this granularity [@leng_predicting_2020] or only focus on estimating historic yields instead of predicting future insurer-relevant metrics [@lobell_scalable_2015; @ma_qdann_2024].

## Contribution
We present a neural network Monte Carlo method to address this need for institutionally-relevant future loss prediction. Our granular simulations provide these projections at the informative risk unit scale, probabilistically forecasting institution-relevant outcome metrics under climate change. We focus on the important U.S. Corn Belt, a 9 state region within the United States essential to the nation's maize crop [@green_where_2018]. Within this agriculturally important area, we simulate the Multiple Peril Crop Insurance Program, "the oldest and most common form of federal crop insurance" [@chite_agricultural_2006]. We specifically model changes to risk under the Yield Protection plan. Furthermore, by contrasting results to a "counterfactual" which does not include further climate warming, we quantitatively highlight the insurer-relevant effects of climate change in near (2030) and medium-term (2050) timeframes [@williams_high_2024].

# Methods
We first build predictive models of maize yield distributions using a neural network at an insurer-relevant spatial scale. We then estimate changes to yield losses under different climate conditions with Monte Carlo simulation in order to estimate crop loss probability and severity. Finally, changes to yield estimate insurance claims rates.

## Definitions
Before modeling these systems, we articulate mathematical definitions of domain-specific concepts and policy instruments. First, insurers pay out based on the magnitude of a yield loss across the aggregation of all of the fields in an insured unit. This loss ($l$) is defined as the difference between actual yield ($y_{actual}$) and a guarantee threshold set by a coverage level ($c$) typically described as a precentage of an expected yield ($y_{expected}$) [@rma_crop_2008].

$l = max(c * y_{expected} - y_{actual}, 0)$

Growers submit production histories for the covered crop ($y_{historic}$) and typically the average of the 10 ($d=10$) most recent years of yield in which the crop was grown define expected ($y_{expected}$) yield [@rma_crop_2008]. This may span more than 10 calendar years due to, for example, crop rotations.

$y_{expected} = \frac{y_{historic}[-d:]}{d}$

Next, we define probability of experiencing a loss that may incur a Yield Protection claim ($p_{l}$) as claims rate.

$p_{l} = P(l > 0) = P(c * y_{expected} - y_{actual} > 0) = P(\frac{y_{actual} - y_{expected}}{y_{expected}} < c - 1) = P(y_{\Delta\%} < c - 1)$

Genearlly, the severity ($s$) of a loss when it occurs defines the size of the claim.

$s = \frac{l}{y_{expected}} = \max(c - \frac{y_{actual}}{y_{expected}}, 0) = \max(-1 * y_{\Delta\%} - (1 - c), 0)$

Note that this paper presents results using the more common 75% coverage level ($c=0.75$) per Federal Crop Insurance Corporation guidelines [@fcic_crop_2023] though our interactive tools allow for further exploration.

## Data
As Yield Protection operates at the level of a risk unit, modeling these formulations requires highly local yield and climate information. Therefore, we use SCYM [@lobell_scalable_2015] which provides remotely sensed yield estimations from 1999 to 2022 at 30m resolution across the US Corn Belt. SCYM benefits from substantial validation efforts [@deines_million_2021]. Meanwhile, we use climate data from CHC-CMIP6 [@williams_high_2024] which, at daily 0.05 degree or approximately 5km scale, offers both historic data on growing conditions from 1983 to 2016 as well as future projections with a 2030 and a 2050 series each containing multiple years. In choosing from its two available scenarios, we prefer the "intermediate" SSP245 within CHC-CMIP6 over SSP585 per the advice of @hausfather_emissions_2020. This offers the following climate variables for modeling: precipitation, temperature (min and max), relative humidity (average, peak), heat index, wet bulb temperature, vapor pressure deficit, and saturation vapor pressure. Note that we prefer SCYM over recent alternatives [@ma_qdann_2024] given temporal overlap with CHC-CMIP6.

### Neighborhoods
We align these variables to a common grid in order to create the discrete instances needed for model training and evaluation. More specifically, we create "neighborhoods" [@manski_diversified_2024] of geographically proximate fields paired with climate data through 4 character geohashing^[This algorithm creates a hierarchical set of grid cells where each point is assigned a unique string through a hashing algorithm. For example, the first 4 characters identifies a grid cell which contains all points with the same first 4 characters of their geohash. We evaluate alternative neighborhood sizes (number of geohash characters) in our interactive tools.] [@niemeyer_geohashorg_2008]. This defines small populations in a grid of cells roughly 28 by 20 kilometers for use within statistical tests [@haugen_geohash_2020]. We simulate units within each of these cells as described below.

### Yield deltas
Having created these spatial groups, we model against SCYM-observed deviations from yield expectations ($\frac{y_{actual} - y_{expected}}{y_{expected}}$) which can be used to calculate loss probability ($l$) and severity ($s$). This step converts from a distribution of absolute yield outcomes to a distribution of changes or "yield deltas" relative to the average production histories. This reflects the mechanics of Yield Protection policies.

## Regression
With these data in mind, we build predictive models for use in simulations of future yield loss outcomes.

### Response and input vector
We predict yield delta distributions per year ahead of Monte Carlo simulations. Specifically, we predict either two parmameters (mean, std) for a normal distribution or four parameters [@scipy_beta_2024] for a beta distribution [@nelson_influence_1990] with distribution type chosen by skew and kurtosis [@kim_statistical_2013]. This use of summary statistics helps ensure appropriate dimensionality for the dataset size [@alwosheel_dataset_2018]. To predict these responses, we describe each of the 9 CHC-CMIP6 variables as min, max, mean, count, and standard deviation per month for the given year. These varibles constitute the model input vector along with year, the historic absolute yield mean ($y_{\mu-historic}$), and standard deviation ($y_{\sigma-historic}$) seen in the neighborhood which capture some measures around baseline variability. See supplemental and interactive tools for further exploration.

### Neural network
Our regressors ($f$) use neighborhood-level climate variables ($C$) and historic yield information to predict future yield changes ($y_{\Delta\%}$) per year. We preprocess these inputs using z score normalization [@kim_investigating_2024].

$f(C_z, y_{\mu-historic-z}, y_{\sigma-historic-z}) \hat= y_{\Delta\%}(x) = \frac{y_{actual} - y_{expected}}{y_{expected}} = \frac{y_{\Delta}}{y_{\mu-historic}}$

Note that we use machine learning per the advice of @leng_predicting_2020 and @klompenburg_crop_2020. In addition to possibly better out-of-sample estimation relative to other similar approaches [@mwiti_random_2023], we specifically use feed forward artificial neural networks [@baheti_essential_2021] as they "natively" support multi-variable output to predict distribution parameters together in the same network as opposed to some other machine learning options which must predict them separately [@brownlee_deep_2020]. Of course, many different kinds of neural network structures and configurations could meet these criteria. Therefore, we grid sweep hyper-parameters [@joseph_grid_2018]. 

| **Parameter**                | **Options**                  | **Description**                                                                                                                                                       | **Purpose**                                                                                                                          |
| ----------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Layers                       | 1 - 6                        | Number of feed forward layers to include where 2 layers include 32 and then 8 nodes while 3 layers include 64, 32, and 8. Layer sizes are {512, 256, 128, 64, 32, 8}. | More layers might allow networks to learn more sophisticated behaviors but also might overfit to input data.                         |
| Dropout                      | 0.00, 0.01, 0.05, 0.10, 0.50 | This dropout rate applies across all hidden layers.                                                                                                                   | Random disabling of neurons may address overfitting.                                                                                 |
| L2                           | 0.00, 0.05, 0.10, 0.15, 0.20   | This L2 regularization strength applies across all hidden layer neuron connections.                                                                                   | Penalizing networks with edges that are "very strong" may confront overfitting without changing the structure of the network itself. |
| Attr Drop                    | 9                              | Retraining where the sweep individually drops each of the input distributions or year or keeps all inputs.                                                      | Removing attributes helps determine if an input may be unhelpful.                                                                    |

Table: Parameters which we try in different permutations to find an optimal configuration. {#tbl:sweepparam}

In order to find a suitable combination of hyper-parameters, this process involves permuting different options from Table @tbl:sweepparam before we select a configuration^[All non-output neurons use Leaky ReLU activation per @maas_rectifier_2013 and we use AdamW optimizer [@kingma_adam_2014; @loshchilov_decoupled_2017].] from the hundreds of candidate models. Finally, with meta-parameters chosen, we can then retrain on all available data ahead of simulations.

## Evaluation
We choose our model using each candidate's ability to predict into future years, a task representative of the Monte Carlo simulations [@brownlee_what_2020]:

- Train on all data between 1999 to 2012 inclusive.
- Use 2014 and 2016 as validation set to compare candidates.
- Test in which 2013 and 2015 serve as a fully hidden set in order to estimate how the chosen model may perform in the future.

Having performed model selection, we further evaluate our chosen regressor through additional tests which more practically estimate performance in different ways one may consider using this method (see Table @tbl:posthoc).

| **Trial**             | **Purpose**                                               | **Train**                                                                                                   | **Test**                                         |
| ------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Random Assignment     | Evaluate ability to predict generally.                    | Random 75% of year / geohash combinations such that a geohash may be in training one year and test another. | The remaining 25% of year / region combinations. |
| Temporal Displacement | Evaluate ability to predict into future years.            | All data from 1999 to 2013 inclusive.                                                                       | All data 2014 to 2016 inclusive.                 |
| Spatial Displacement  | Evaluate ability to predict into unseen geographic areas. | All 4 character geohashes in a randomly chosen 75% of 3 character regions.                                  | Remaining 25% of regions.                        |
| Climatic Displacement | Evaluate ability to predict into out of sample growing conditions. | All years but 2012. | 2012 (unusually dry / hot) |

Table: Overview of trials after model selection. {#tbl:posthoc}

These post-hoc trials use only training and test sets as we fully retrain models using unchanging sweep-chosen hyper-parameters as described in Table @tbl:sweepparam. Note that some of these tests use "regions" which we define as all geohashes sharing the same first three characters, creating a grid of 109 x 156 km cells [@haugen_geohash_2020] each including all neighborhoods (4 character geohashes) found within.

## Simulation
After training machine learning models using historic data, predictions of future distributions feed into Monte Carlo simulations [@metropolis_beginning_1987; @kwiatkowski_monte_2022] as described in Figure @fig:pipeline. This operation happens for 17 individual years sampled separately from both the 2030 and 2050 CHC-CMIP6 series [@williams_high_2024].

![Model pipeline overview diagram. Code released as open source.](./img/pipeline.png "Model pipeline overview diagram. Code released as open source."){ width=80% #fig:pipeline }

With trials consisting of sampling at the neighborhood scale, this approach allows us to consider a distribution of future outcomes for each neighborhood. These results then enable us to make statistical statements about systems-wide institution-relevant events such as claims rate ($p_{l}$). We specifically simulate loss probability prior to unit optimizations as explored in discussion.

### Trials
Each Monte Carlo trial involves multiple sampling operations. First, we sample climate variables and model error residuals to propagate uncertainty [@yanai_estimating_2010]. Next, we draw yield multiple times to approximate the size of a risk unit with its portfolio effects. Note that the size but not the specific location of insured units is publicly disclosed. Therefore, we draw the geographic size of each insured unit randomly from historic data [@rma_statecountycrop_2024] as part of Monte Carlo. Trials are further described in our supplemental materials.

### Statistical tests
Altogether, this approach simulates insured units individually per year. Having found these outcomes as a distribution per neighborhood, we can then evaluate these results probabilistically. As further described in supplemental, we determine significance both in this paper and our interactive tools via Bonferroni-corrected [@bonferroni_il_1935] Mann Whitney U [@mann_test_1947] per neighborhood.

# Results
We project climate change to roughly double loss probabilities ($p_{l}$) at mid-century.

## Aggregation outcomes
Our resulting historic dataset spans 1999 to 2016 during which we observe a median of 83k SCYM yield estimations at roughly field-scale per neighborhood. These outcomes are represented within neighborhood-level distributions per year. Consistent with @nelson_influence_1990, neighborhood distributions of SCYM yields are often not normally distributed. However, we observe that nearly all yield _delta_ distributions exhibit approximate normality [@kim_statistical_2013]. Therefore, we report our model outputs under a normal deltas assumption but our supplemental materials provide further statistics and alternative beta distribution results.

## Neural network outcomes
With bias towards performance in mean prediction, we select {{numLayers}} hidden layers ({{layersDescription}}) using {{dropout}} dropout and {{l2}} L2 from our sweep with all data attributes included. We report mean absolute error (MAE) in yield delta points ($|\frac{y_{actual} - y_{expected}}{y_{expected}} - y_{\Delta\% - Predicted}|$). Our selected model sees {{retrainMeanMae}} MAE when predicting neighborhood mean change in yield ($y_{\Delta\%}$) and {{retrainStdMae}} when predicting neighborhood standard deviation when evaluting using the fully hidden test set. Though our supplemental offers additional performance metrics, Table @tbl:trainresults reports on performance after retraining with train and validation together.

| **Set**             | **MAE for Mean Prediction** | **MAE for Std Prediction** |
| -------------------- | ----------------------- | ---------------------- |
| Train                | {{trainMeanMae}}        | {{trainStdMae}}   |
| Validation           | {{validationMeanMae}}   | {{validationStdMae}}   |
| Test with retrain    | {{retrainMeanMae}}      | {{retrainStdMae}}      |

Table: Results of model training and selection. {#tbl:trainresults}

We also evaluate regression performance through varied definitions of test sets in Table @tbl:posthocresults, showing similar results.

| **Task**              | **Test Mean Pred MdAE** | **Test Std Pred MdAE** | **% of Units in Test Set** |
| --------------------- | ---------------------- | --------------------- | -------------------------- |
| Random   | {{randomMeanMae}}      | {{randomStdMae}}      | {{randomPercent}}          |
| Temporal | {{temporalMeanMae}}    | {{temporalStdMae}}    | {{temporalPercent}}        |
| Spatial  | {{spatialMeanMae}}     | {{spatialStdMae}}     | {{spatialPercent}}         |
| Climatic | {{climateMeanMae}}     | {{climateStdMae}}     | {{climatePercent}}         |

Table: Results of tests after model selection. {#tbl:posthocresults}

Our interactive tools allow for further examination of error and candidates.

## Simulation outcomes
After retraining on all available data using the selected configuration from our sweep, Monte Carlo simulates overall outcomes. These trials track changes to average yields over time at the neighborhood level and sample test set model residuals to account for error. Despite the conservative nature of the Bonferroni correction [@mcdonald_handbook_2014], {{percentSignificant}} of maize acreage in SSP245 falls within a neighborhood with significant changes to claim probability ($p < 0.05 / n$) at some point during the 2050 series simulations. That said, we observe that some of the remaining neighborhoods failing to meet that threshold have less land dedicated to maize within their area (smaller sample size in simulations).

![Overview of Monte Carlo simulation results comparing SSP245 versus counterfactual for (A) loss probability, (B) loss severity, and (C) change in average yields. Counterfactual is a future without continued warming.](./img_static/sim.png "Overview of Monte Carlo simulation results comparing SSP245 versus counterfactual for (A) loss probability, (B) loss severity, and (C) change in average yields. Counterfactual is a future without continued warming."){#fig:simresults}

Our results find claims rate to elevate sharply in the 2030 series and double in the 2050 timeframe when using SSP245 relative to the no further warming counterfactual which sees relatively steady metrics. Additionally, climate change reduces the expected average yield and, as 2050 witnesses further warming compared to 2030, later simulations report higher claims rates. Finally, due to a high degree of aggregation, system-wide metrics visualized within Figure @fig:simresults are stable: each metric sees a range of less than 1% difference across 100 re-executions. However, some sources of uncertainty like from input data (SCYM and CHC-CMIP6) cannot be quanitified given currently available information. Our supplemental materials provide further results.

# Discussion
In addition to highlighting future work opportunities and further exploring our results, we observe a number of policy-relevant dynamics within our simulations.

## Yield expectations
Figure @fig:hist highlights predicted structural changes to yield deltas. This comparison reveals possible challenges with using a simple average in crop insurance products: simulations anticipate that the claims rate increases under climate change (SSP245) outpace evolving yield means used to set expectations.

![Interactive tool screenshot showing 2050 outcomes distribution as changes from $y_{expected}$, showing deltas and claims rates with climate change on the top and without further climate change (counterfactual) on bottom.](./img/hist.png "Interactive tool screenshot showing 2050 outcomes distribution. This graphic depicts changes from $y_{expected}$, showing deltas and claims rates with further climate change on the top and without climate change (counterfactual) on bottom."){#fig:hist}

Indeed, {{ dualIncreasePercent2050 }} of neighborhoods seeing higher claims rates under SSP245 in the 2050 series also report overall multi-year average yields remaining unchanged or even increasing. In other words, yield volatility could allow a sharp elevation in loss probability without necessarily decreasing overall mean yields substantially enough to reduce claims rates through $y_{expected}$. Though our interactive tools further explore these dynamics between yield expectations and volatility, our work may highlight a need for future research into alternative policy formulations such as historic yield variance in addition to a simple average.

### Impact to insurers
Plans where loss is calculated against averages of historic yields may fail to capture an increase in risk due to changing shapes of yield delta distributions [@fcic_common_2020]. This could allow elevated loss and insurer strain to hide behind the smoothing effect of mean yields. In other words, risk may increase at the insured unit scale in a way that is "invisible" to some current policy instruments.

### Impact to growers
Some risk mitigating practices such as regenerative agriculture may not always improve mean yields or can even come at the cost of a slightly reduced average [@deines_recent_2023]. Even though these practices may help guard against an elevated probability of loss events reflected in the yield delta shape [@renwick_long-term_2021], our results may indicate a mechanism for how average-based expectations could possibly disincentivize growers from climate change preparation. That said, we acknowledge that crop insurance effects on grower behavior remains an area of active investigation [@connor_crop_2022; @wang_warming_2021; @chemeris_insurance_2022].

## Recent actual claims rates
We generally predict a 13% claims rate in 2030 and 2050 "counterfactual" simulations which anticipate outcomes absent further climate change (future conditions similar to recent past). For comparison, the annual median of the historic series encouragingly reports an actual 14% claims rate [@rma_statecountycrop_2024] amid growing conditions similar to counterfactual trials.

### Under-estimation
Despite this similarity between predictions and the comparable recent actuals, a number of difficult to model factors likely lead us to underestimate the actual claims rate in practice. First, field-level yield data and the actual geographically specific risk unit structure are not currently public. Therefore, while we sample units randomly based on expected size, growers likely optimize their own unit structure when purchasing policies to optimize financial upside. Similarly, we do not have the geographically specific data required to model trend adjustment and yield exclusion options^[Under certain conditions, trend adjustment increases $y_{expected}$ to be higher than the historic average [@plastina_trend-adjusted_2014] to anticipate expected yield improvements while exclusions remove poor years from $y_{expected}$ calculation [@schnitkey_yield_2015].]. These factors likely increase the actual claims rates by raising $y_{expected}$. See supplemental for more details.

### Variation
While these model limitations likely overall lead to a supression of loss rates in our simulations relative to actuals, note that these adjustments change over time and could cause further fluctations alongside growing condition variability. For example, 2014 saw a number of statutory changes [@ers_farm_2014]. We anticipate that the future will likely see substantial annual variation similar to the recent past even as our results still capture overall long term trends.

## Geographic bias
Neighborhoods with significant results ($p < 0.05 / n$) may be more common in some areas as shown in Figure @fig:geo. This spatial pattern may partially reflect that a number of neighborhoods have less land dedicated to maize so simulations have smaller sample sizes and fail to reach significance. However, this may also reflect geographical bias in altered growing conditions.

![Interactive geographic view. Color describes type of change. Larger dots are larger areas of maize growing activity. Band of increased risk concentrates in Iowa, Illinois, and Indiana. This could reveal a possible geographic bias within our results. Use of the interactive tool allows for consideration of different parameters including alternative statistical treatments.](./img/map.png "Interactive geographic view. Color describes type of change and larger dots are larger areas of maize growing activity. Band of increased risk concentrates in Iowa, Illinois, and Indiana."){#fig:geo}

In particular, we note some spatial patterns in precipitation and temperature changes that may intersect with these trends. Indeed, our model shows depressed yields in response to anticipated combined warmer and drier conditions similar to 2012 and its historically poor maize production [@ers_weather_2013]. Our predictions thus reflect empirical studies that document the negative impacts of heat stress and water deficits on maize yields [@sinsawat_effect_2004; @marouf_effects_2013]. In total, our outputs may reveal geographically and temporally specific outlines of anticipated impacts as described in our interactive tools.

## Other limitations and future work
We next highlight opportunities for future work.

### Future data
We acknowledge limitations of our findings due to constraints of the currently available public datasets. First, though our interactive tools consider different spatial aggregations such as 5 character (approx 4 x 5 km) geohashes, future work may consider modeling with actual reported field-level yield data and the actual risk unit structure if later made public as earlier mentioned. Additionally, we highlight that we focus on systematic changes in growing conditions impacting claims rates across a broad geographic scale. This excludes highly localized effects like certain inclement weather which may require more granular climate predictions. This possible future work may be relevant to programs with smaller geographic portfolios. Finally, as further described in supplemental, our model shows signs that it is data constrained. In particular, additional years of training data may improve performance. Our data pipeline should and can be re-run as future versions of CHC-CMIP6 and SCYM or similar are released.

### Other programs
Outside of Yield Protection, future study could extend to the highly related Revenue Protection form of insurance. Indeed, the yield stresses that we describe in this model may also impact this other plan. On that note, we include historic yield as inputs into our neural network, allowing those data to "embed" adaptability measures [@hsiang_estimating_2017] such as grower practices where, for example, some practices may reduce loss events or variability [@renwick_long-term_2021]. That said, we highlight that later studies looking at revenue may require additional economic information to serve a similar role.

### Future benchmarking
We offer a unique focus on broad geographic institutionally-relevant loss probability prediction at risk unit scale given remote sensed yield estimations. Lacking a compatible study for direct contrasting of performance measures, we invite further research on alternative regression and simulation approaches for similar modeling objectives. While not directly comparable, we note that @lobell_statistical_2010 as well as @leng_predicting_2020 possibly offer precedent.

## Visualizations and software
In order to explore these simulations, we offer interactive open source web-based visualizations built alongside our experiments. These both aid us in constructing our own conclusions and allow readers to consider possibilities and analysis beyond our own narrative. This software runs within a web browser and is made publicly available at https://ag-adaptation-study.org. It includes the ability to explore alternative statistical treatments and regressor configurations as well as generate additional geographic visualizations. Finally, in addition to visualizations, we also offer our work as an open source data science pipeline. This software may help aid future research into other crops such as soy, geographic areas such as other parts of the United States of America, other programs such as Revenue Protection, and extension of our results as underlying datasets are updated. Supplemental materials offer further detail.

# Conclusion
We present Monte Carlo on top of neural network-based regressors for prediction of institution-relevant crop yield changes. We specifically simulate climate-driven system-wide impacts to maize growing conditions at a policy-relevant scale of granularity. Our results find that maize Yield Protection claim rates may double for the U.S. Federal Crop Insurance Program (Multi-Peril Crop Insurance) within the U.S. Corn Belt relative to a no further warming counterfactual.

In addition to publishing our raw model outputs under a creative commons license, we explore the specific shape of these results from the perspective of insurance structures. First, we describe a possible agriculturally-relevant geographic bias in climate impacts. Second, we also highlight potential mathematical properties of interest including increasing volatility without fully offsetting average-based yield expectation measures. These particular kinds of changes may pose specific threats to the current structure of existing insurance structures.

Altogether, this study considers how this machine learning and interactive data science approach may understand existing food system policy structures in the context of climate projections. Towards that end, we release our software under permissive open source licenses and make interactive tools available publicly at https://ag-adaptation-study.org to further interrogate these results. These visualizations also allow readers to explore alternatives to key analysis parameters. This work may inform agriculture policy response to continued climate change.

# Acknowledgements
Study funded by the Eric and Wendy Schmidt Center for Data Science and Environment at the University of California, Berkeley. We have no conflicts of interest to disclose. Using yield estimation data from @lobell_scalable_2015 and @deines_million_2021 with our thanks to David Lobell for permission. We also wish to thank Carl Boettiger, Magali de Bruyn, Jiajie Kong, Kevin Koy, and Ciera Martinez for conversation regarding these results. Thanks to Color Brewer [@brewer_colorbrewer_2013] and Public Sans [@general_services_administration_public_2024].

**Data availability statement**: Our software and pipeline source code [@pottinger_data_2024] as well as our model training data and simulation outputs [@pottinger_data_2024-1] are available on Zenodo as open source / creative common licensed resources. Public hosted version at https://ag-adaptation-study.org.

# Works Cited
