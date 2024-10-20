---
bibliography: ./paper.bib
title: "Climate-Driven Doubling of U.S. Maize Loss Probability: Simulation through Neural Network Monte Carlo"
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
date: 2024-07-31
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
  - name: Nick Gondek
    affil-id: 1
  - name: Timothy Bowles \orcidlink{0000-0002-4840-3787}
    affil-id: 3
output:
  pdf_document:
    number_sections: yes
    template: default.tex
---

**Abstract:** Climate change not only threatens agricultural producers but also strains public agencies and financial institutions. These important food system actors include government entities tasked with insuring grower livelihoods and supporting agricultural response to continued global warming. We examine the impacts of climate-driven crop loss on these important organizations. Specifically, we build upon prior work by offering prediction of institutionally-relevant future yield loss by employing neural network Monte Carlo at policy-salient "risk unit" scale. We demonstrate this simulation of future risk within the U.S. Corn Belt in the context of the crucial U.S. Federal Crop Insurance Program (FCIP). Our results worryingly suggest more frequent and severe losses seen within our predictions would result in a financially onerous doubling in the annual probability of maize loss claims within FCIP's Yield Protection (YP) option at mid-century relative to current levels. Offering visualization-rich interactive tools and open source pipelines, our data science contribution fills an actionable gap in current understanding by bridging existing historic yield estimation and climate prediction. As these kinds of system-wide stresses may impact programs important to food system stability, this simulation of future crop loss may help inform policy for adaptation.

# Introduction
Specifically focusing on maize, we investigate how climate change may impact the insurance claims rate within the U.S. Federal Crop Insurance Program's Yield Protection scheme. We generate these projections through Monte Carlo simulations on top of neural network regressors for future agricultural yield loss predictions at an institutionally-relevant spatial scale. These estimations may help inform future climate adapation efforts.

## Background
Global warming threatens production of key staple crops, including maize [@rezaei_climate_2023]. Climate variability already drives a substantial proportion of year-to-year crop yield variation [@ray_climate_2015] and continued climate change may reduce planet-wide maize yields by up to 24% by the end of this century [@jagermeyr_climate_2021]. Beyond reduced mean output, growing frequency and severity of stressful weather [@dai_increasing_2013] to which maize is increasingly sensitive [@lobell_changes_2020] will also impact both farmers' revenue [@sajid_extreme_2023] and the institutions designed to protect those producers [@hanrahan_crop_2024].

Within this context, the United States of America is the world's largest maize producer and exporter [@ates_feed_2023]. Its government-backed Federal Crop Insurance Program (FCIP) covers a large share of this growing risk [@tsiboe_crop_2023]. The costs of crop insurance in the U.S. have already increased by 500% since the early 2000s with annual indemnities reaching $19B in 2022 [@schechinger_crop_2023]. Furthermore, retrospective analysis attributes 19% of "national-level crop insurance losses" between 1991 and 2017 to climate warming, an estimate rising to 47% during the drought-stricken 2012 growing season [@diffenbaugh_historical_2021]. Looking forward, @li_impact_2022 show progressively higher U.S. maize loss rates as warming elevates.

## Prior work
Modeling the possible changes in frequency and severity of crop loss events that trigger indemnity claims is an important step to prepare for the future impacts of global warming. Related studies have predicted changes in crop yields at broad scales such as county-level aggregation [@leng_predicting_2020] and have estimated climate change impacts to U.S. maize within whole-sector or whole-economy analysis [@hsiang_estimating_2017]. In addition to traditional statistical models [@lobell_statistical_2010], an increasing body of work favors machine learning approaches [@leng_predicting_2020].

Despite this important body of prior work, programs like insurance products frequently include elements operating at the producer level [@rma_crop_2008]. Prior studies often do not include more granular models of insurer-focused claims rate and loss severity at that policy-relevant spatial scale. Of particular interest, the "risk" or "insured" unit refers to a set of fields that are insured together within an individual policy [@fcic_common_2020]. This important scale essential to crop insurance structure is typically much smaller than a county [@rma_statecountycrop_2024]. While modeling at this scale may provide important institutional insight, many prior studies either do not offer this granularity [@leng_predicting_2020] or focus on estimating historic yields instead of predicting future insurer-relevant metrics [@lobell_scalable_2015; @ma_qdann_2024].

## Contribution
We address this need for institutionally-relevant granular future loss prediction through neural network Monte Carlo. We provide these projections at the risk unit scale, probabilistically forecasting institution-relevant outcome metrics under climate change. Within the important U.S. Corn Belt multi-state geographic region, we simulate the Multiple Peril Crop Insurance Program, "the oldest and most common form of federal crop insurance" [@chite_agricultural_2006]. We specifically model changes to risk under the Yield Protection (YP) plan. Furthermore, by contrasting results to a "counterfactual" which does not include further climate warming, we then quantitatively highlight the insurer-relevant effects of climate change in the 2030 and 2050 timeframes for which sufficient climate projections are available [@williams_high_2024].

# Methods
We first build predictive models of crop yield distributions using a neural network at a spatial scale relevant to insurers. We then estimate changes to yield losses under different climate conditions with Monte Carlo simulation in order to estimate loss probability and severity.

## Definitions
Before modeling these systems, we articulate domain-specific mathematical definitions. First, insurers pay out based on the magnitude of a yield loss across the aggregation of all of the fields in an insured unit. This loss ($l$) is defined as the difference between actual yield ($y_{actual}$) and a guarantee threshold set by a coverage level ($c$) typically described as a precentage of an expected yield ($y_{expected}$) [@rma_crop_2008].

$l = max(c * y_{expected} - y_{actual}, 0)$

Growers submit production histories for the covered crop ($y_{historic}$) and the average of the 10 most recent years of yield ($d=10$) are generally used to define expectations [@rma_crop_2008]. This is further explored in our interactive tools.

$y_{expected} = \frac{y_{historic}[-d:]}{d}$

Next, we can create a definition of loss risk ($p$) as the probability of experiencing a loss that may incur a claim.

$p = P(l > 0) = P(c * y_{expected} - y_{actual} > 0) = P(\frac{y_{actual} - y_{expected}}{y_{expected}} < c - 1) = P(y_{\Delta\%} < c - 1)$

Ginally, the severity ($s$) of a loss when it occurs defines the size of the claim.

$s = \frac{l}{y_{expected}} = \max(c - \frac{y_{actual}}{y_{expected}}, 0) = \max(-1 * y_{\Delta\%} - (1 - c), 0)$

Note that this paper presents results using the more common 75% coverage level ($c=0.75$) per Federal Crop Insurance Corporation guidelines [@fcic_crop_2023] though our interactive tools allow for further exploration.

## Data
As Yield Protection operates at the level of a risk unit (set of fields insured together), modeling these formulations requires highly local yield and climate information. Therefore, we use the Scalable Crop Yield Mapper (SCYM) which provides remotely sensed yield estimations from 1999 to 2022 at 30m resolution across the US Corn Belt [@lobell_scalable_2015]. This data product benefits from substantial validation efforts [@deines_million_2021]. Meanwhile, we use climate data from CHC-CMIP6 [@williams_high_2024] which, at daily 0.05 degree or approximately 5km scale, offers both historic data from 1983 to 2016 as well as future projections with multiple years per a 2030 and a 2050 series. In choosing from its two available shared socioeconomic pathways, we prefer the "intermediate" SSP245 within CHC-CMIP6 over SSP585 per the advice of @hausfather_emissions_2020. This offers the following climate variables for modeling: precipitation, temperature (min and max), relative humidity (average, peak), heat index, wet bulb temperature, vapor pressure deficit, and saturation vapor pressure. Note that we prefer SCYM over some more recent alternatives [@ma_qdann_2024] given its temporal overlap with CMIP6 data. A more recent or longer year range should and will be revisited after the release of CHC-CMIP7.

We align these variables to a common grid in order to create the discrete instances needed for model training and evaluation. More specifically, we create "neighborhoods" [@manski_diversified_2024] of geographically proximate fields paired with climate data through 4 character geohashing^[This algorithm creates a hierarchical set of grid cells where each point is assigned a string through a hashing algorithm. For example, the first 4 characters identifies a grid cell which contains all points with the same first 4 characters of their geohash. That cell is roughtly 28 x 20 km in the U.S. Corn Belt. We also evaluate alternative neighborhood sizes (number of geohash characters) in the interactive tools.] [@niemeyer_geohashorg_2008]. This defines small populations in a grid of cells roughly 28 by 20 kilometers for use within statistical tests [@haugen_geohash_2020]. We observe a median of 83k SCYM yield estimations at roughly field-scale per neighborhood. These outcomes are represented within neighborhood-level distributions.

Having created these spatial groups, we model against SCYM-observed deviations from yield expectations ($y_{expected} - y_{actual}$) which can be used to calculate loss probability ($l$) and severity ($s$). This converts from a distribution of absolute yield outcomes to a distribution of changes or "yield deltas" which we summarize as neighborhood-level means and standard deviations. Using these summary statistics as the response variables for regression helps ensure appropriate dimensionality for the dataset size given approximate normalilty in most neighborhoods [@kim_statistical_2013]. See interactive tools for further exploration. Finally, we similarly describe climate deltas as min, max, mean and standard deviation per month.

## Regression
With these data in mind, we next build predictive models for use in simulations of future yield loss outcomes. Our regressors ($f$) use neighborhood-level climate variables ($C$), year ($x$), and historic yield mean / std to predict future neighborhood-level mean and standard deviation of yield changes ($y_{\Delta\%}$) per year [@kim_investigating_2024]. This uses z score normalization ($z_{\mu-historic}, z_{\sigma-historic}$).

$y_{\Delta\%}(x) = \frac{y_{actual} - y_{expected}}{y_{expected}} = \frac{y_{\Delta}}{y_{\mu-historic}} = f(C, x, z_{\mu-historic}, z_{\sigma-historic})$

Note that we use machine learning per the advice of @leng_predicting_2020 and @klompenburg_crop_2020. We specifically use feed forward artificial neural networks [@baheti_essential_2021] as they "natively" support multi-variable output to predict mean and standard deviation together in the same network as opposed to some other machine learning options which must predict them separately [@brownlee_deep_2020]. Neural networks may also perform better in out-of-sample range estimation [@mwiti_random_2023]. Of course, many different kinds of neural network structures and configurations could meet these criteria. Therefore, we try various combinations of "hyper-parameters" in a grid search sweep [@joseph_grid_2018]. 

| **Parameter**                | **Options**                  | **Description**                                                                                                                                                       | **Purpose**                                                                                                                          |
| ----------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Layers                       | 1 - 6                        | Number of feed forward layers to include where 2 layers include 32 and then 8 nodes while 3 layers include 64, 32, and 8. Layer sizes are {512, 256, 128, 64, 32, 8}. | More layers might allow networks to learn more sophisticated behaviors but also might overfit to input data.                         |
| Dropout                      | 0.00, 0.01, 0.05, 0.10, 0.50 | This dropout rate applies across all hidden layers.                                                                                                                   | Random disabling of neurons may address overfitting.                                                                                 |
| L2                           | 0.00, 0.05, 0.10, 0.15, 0.20   | This L2 regularization strength applies across all hidden layer neuron connections.                                                                                   | Penalizing networks with edges that are "very strong" may confront overfitting without changing the structure of the network itself. |
| Attr Drop                    | 10                           | Retraining where the sweep individually drops each of the input distributions or year or keeps all inputs.                                                      | Removing attributes helps determine if an input may be unhelpful.                                                                    |

Table: Parameters which we try in different permutations to find an optimal configuration. {#tbl:sweepparam}

In order to find a suitable combintaion of hyper-parameters, this process involves permuting different option combinations from Table @tbl:sweepparam before we select a configuration^[All non-output neurons use Leaky ReLU activation per @maas_rectifier_2013 and we use AdamW optimizer [@kingma_adam_2014; @loshchilov_decoupled_2017].] from the 1,500 candidate models. Finally, with meta-parameters chosen, we can then retrain on all available data ahead of simulations.

## Simulation
After training machine learning models using historical data, predictions of future distributions feed into Monte Carlo simulations [@metropolis_beginning_1987; @kwiatkowski_monte_2022] as described in Figure @fig:pipeline. This happens for five individual years separately in both the 2030 and 2050 CHC-CMIP6 series [@williams_high_2024].

![Model pipeline overview diagram. Code released as open source.](./img/pipeline.png "Model pipeline overview diagram. Code released as open source."){ width=80% #fig:pipeline }

With trials consisting of sampling at the neighborhood scale, this approach allows us to consider a distribution of future outcomes for each neighborhood. These results then enable us to make statistical statements about systems-wide institution-relevant events such as claims rate.

### Trials
Each Monte Carlo trial involves multiple sampling operations. First, we sample climate variables and model error residuals to propagate uncertainty [@yanai_estimating_2010]. Next, we draw multiple times to approximate the size of a risk unit with its portfolio effects. Note that the size but not the specific location of insured units is publicly disclosed. Therefore, we first draw the geographic size of an insured unit randomly from historic data [@rma_statecountycrop_2024]. Afterwards, we can then draw yields from the neighborhood distribution with the number of samples dependent on that insured unit size. Note that supplemental provides further data on risk units.

### Statistical tests
Altogether, this approach simulates insured units individually per year. Having found these outcomes as a distribution per neighborhood, we can then evaluate these results probabilistically. As further described in supplemental, we determine significance both in this paper and our interactive tools via Bonferroni-corrected [@bonferroni_il_1935] Mann Whitney U [@mann_test_1947] per neighborhood per year.

## Evaluation
We choose our model using each candidate's capability to predict into future years, a task representative of the Monte Carlo simulations [@brownlee_what_2020]:

- Train on all data between 1999 to 2012 inclusive.
- Use 2014 and 2016 as validation set to compare the 1,500 candidates.
- Test in which 2013 and 2015 serve as a fully hidden set in order to estimate how the chosen model may perform in the future.

Having performed model selection, we further evaluate our chosen regressor through four additional tests which more practically estimate performance in different ways one may consider using this method (see Table @tbl:posthoc).

| **Trial**             | **Purpose**                                               | **Train**                                                                                                   | **Test**                                         |
| ------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Random Assignment     | Evaluate ability to predict generally.                    | Random 75% of year / geohash combinations such that a geohash may be in training one year and test another. | The remaining 25% of year / region combinations. |
| Temporal Displacement | Evaluate ability to predict into future years.            | All data from 1999 to 2013 inclusive.                                                                       | All data 2014 to 2016 inclusive.                 |
| Spatial Displacement  | Evaluate ability to predict into unseen geographic areas. | All 4 character geohashes in a randomly chosen 75% of 3 character regions.                                  | Remaining 25% of regions.                        |
| Climatic Displacement | Evaluate ability to predict into out of sample growing conditions. | All years but 2012. | 2012 (unusually dry / hot) |

Table: Overview of trials after model selection. {#tbl:posthoc}

These post-hoc trials use only training and test sets as we fully retrain models using unchanging sweep-chosen hyper-parameters as described in Table @tbl:sweepparam. Note that some of these tests use "regions" which we define as all geohashes sharing the same first three characters. This two tier definition creates a grid of 109 x 156 km cells [@haugen_geohash_2020] each including all neighborhoods (4 character geohashes) found within that area.

# Results
We project loss probabilities to more than double ({{experimentalProbability2050}} claims rate) under SSP245 at mid-century in comparison to the no additional warming counterfactual scenario ({{counterfactualProbability2050}} claims rate).

## Neural network outcomes
With bias towards performance in mean prediction, we select {{numLayers}} hidden layers ({{layersDescription}}) using {{dropout}} dropout and {{l2}} L2 from our sweep with all data attributes included. Table @tbl:sweep describes performance for the chosen configuration.

| **Set**             | **MAE for Mean Prediction** | **MAE for Std Prediction** |
| ------------------- | ----------------------- | ---------------------- |
| Train               | {{trainMeanMae}}        | {{trainStdMae}}        |
| Validation          | {{validationMeanMae}}   | {{validationStdMae}}   |
| Test before retrain | {{testMeanMae}}         | {{testStdMae}}         |
| Test after retrain  | {{retrainMeanMae}}      | {{retrainStdMae}}      |

Table: Results of chosen configuration during the "sweep" for model selection. {#tbl:sweep}

After retraining with train and validation together, we see {{retrainMeanMae}} MAE when predicting neighborhood mean and {{retrainStdMae}} when predicting neighborhood standard deviation when using the fully hidden test set.

Next, having chosen this set of hyper-parameters, we also evaluate regression performance through varied definitions of test sets.

| **Task**              | **Test Mean Pred MAE** | **Test Std Pred MAE** | **% of Units in Test Set** |
| --------------------- | ---------------------- | --------------------- | -------------------------- |
| Random   | {{randomMeanMae}}      | {{randomStdMae}}      | {{randomPercent}}          |
| Temporal | {{temporalMeanMae}}    | {{temporalStdMae}}    | {{temporalPercent}}        |
| Spatial  | {{spatialMeanMae}}     | {{spatialStdMae}}     | {{spatialPercent}}         |
| Climatic | {{climateMeanMae}}     | {{climateStdMae}}     | {{climatePercent}}         |

Table: Results of tests after model selection. {#tbl:posthocresults}

The interactive tools website allows for further examination of error.

## Simulation outcomes
After retraining on all available data using the selected configuration from our sweep, Monte Carlo simulates overall outcomes. Despite the conservative nature of the Bonferroni correction [@mcdonald_handbook_2014], {{percentSignificant}} of maize acreage in SSP245 falls within a neighborhood with significant changes to claim probability ($p < 0.05 / n$) at some point during the 2050 series simulations. That said, we observe that some of the remaining neighborhoods failing to meet that threshold have less land dedicated to maize within their area and, thus, a smaller sample size in our simulations.

| **Scenario**   | **Year** | **Unit mean yield change** | **Unit loss probability**         | **Avg covered loss severity**  |
| ---------------------------- | -------- | -------------------------- | --------------------------------- | ------------------------------ |
| Counterfactual | 2030     | {% if counterfactualMean2030|float > 0 %}+{% endif %}{{counterfactualMean2030}} | {{counterfactualProbability2030}} | {{counterfactualSeverity2030}} |
| SSP245         | 2030     | {% if experimentalMean2030|float > 0 %}+{% endif %}{{experimentalMean2030}}   | {{experimentalProbability2030}}   | {{experimentalSeverity2030}}   |
| Counterfactual | 2050     | {% if counterfactualMean2050|float > 0 %}+{% endif %}{{counterfactualMean2050}} | {{counterfactualProbability2050}} | {{counterfactualSeverity2050}} |
| SSP245         | 2050     | {% if experimentalMean2050|float > 0 %}+{% endif %}{{experimentalMean2050}}   | {{experimentalProbability2050}}   | {{experimentalSeverity2050}}   |
|                |          | $y_{\Delta \mu}$           | $p_{\mu}$                         | $s_{\mu}$                      |

Table: Overview of Monte Carlo simulation results. Counterfactual is a future without continued warming in contrast to SSP245. {#tbl:simresults}

As described in Table @tbl:simresults, the loss probability increases in both the 2030 and 2050 time frames considered for SSP245. Note that this happens in addition to wiping out the gains that our neural network would otherwise expect without climate change as would possibly be anticipated historic trends and expectations [@nielsen_historical_2023]. 

# Discussion
In addition to highlighting future work opportunities, we observe a number of policy-relevant dynamics within our simulations.

## Yield expectations
As shown in Figure @fig:hist, the SSP245 overall yield mean remains similar to the historic baseline in the 2050 series even as distribution tails differ more substantially. Granular simulation results reflect this system-wide finding: {{ dualIncreasePercent2050 }} of neighborhoods seeing higher claims rates under SSP245 in the 2050 series also counter-intuitively report overall multi-year average yields remaining unchanged or even increasing. This observation around stability and changing tails shows how yield volatility could allow a sharp elevation in loss probability without necessarily decreasing overall mean yields that would be reflected in $y_{expected}$. In the context of the U.S. Federal Crop Insurance Program specifically, our results may suggest that the current definition of yield expectations as an average of up to the last ten years may fail to capture an increase in risk or climate-relevant yield volatility [@fcic_common_2020].

![Interactive tool screenshot showing 2050 outcomes distribution. This graphic depicts changes from $y_{expected}$, showing deltas and claims rates with climate change on the top and without climate change on the bottom. In addition to showing increased claims rate, the circles along the horizontal axis also depict climate change reducing the expected increase in yields that would otherwise follow historic trends.](./img/hist.png "One of our interactive tools showing 2050 outcomes distribution relative to $y_{expected}$ highlighting loss with and without climate change."){#fig:hist}

These results highlight possible challenges if average-based production histories may reward increases in mean yield but not necessarily yield stability. Some practices such as regenerative agriculture may not always improve mean yields or can even come at the cost of a slightly reduced average [@deines_recent_2023] even though they guard against elevations in the probability of loss events [@renwick_long-term_2021]. This is further explored in our interactive tools. Regardless, our work may highlight a need for future research into alternative policy formulations that may incorporate, for example, historic yield variance in addition to a simple average.

## Geographic bias
Neighborhoods with significant results ($p < 0.05 / n$) may be more common in some areas as shown in Figure @fig:geo.

![Interactive geographic view. Color describes type of change. Larger dots are larger areas of maize growing activity. Band of increased risk concentrates in Iowa, Illinois, and Indiana. This could reveal a possible geographic bias within our results.](./img/map.png "Interactive geographic view. Color describes type of change and larger dots are larger areas of maize growing activity. Band of increased risk concentrates in Iowa, Illinois, and Indiana."){#fig:geo}

This spatial pattern may partially reflect that a number of neighborhoods have less land dedicated to maize so simulations have smaller sample sizes and fail to reach significance. However, this geographic effect may also reflect disproportionate stress or other changes relative to historic conditions. In particular, as further explorable in our interactive tools, we note some geographic bias in changes to precipitation, temperature, and VPD / SVP.

## Heat and drought stress
Our model shows depressed yields during combined warmer and drier conditions, combinations similar to 2012 and its historically poor maize production [@ers_weather_2013]. In this context, precipitation may serve as a protective factor: neighborhoods with drier July conditions see higher loss probability ($p < 0.05 / 2$) in both the 2030 and 2050 series via rank correlation [@spearman_proof_1904]. Our predictions thus reflect empirical studies that document the negative impacts of heat stress and water deficits on maize yields [@sinsawat_effect_2004; @marouf_effects_2013]. As further described in our interactive tools, these outputs may also reveal geographically and temporally specific outlines of these protective factors as possibly useful for insurer and grower adaptation. Even so, we caution that analysis finds significant but still weak rank correlations in both series, indicating that model expectations cannot be described by precipitation alone.

## Other future work
We also highlight limitations and additional future modeling opportunities beyond the scope of this study.

- We evaluate yield deltas and include historic yield as inputs into our neural network, allowing those data to "embed" adaptability measures [@hsiang_estimating_2017] such as soil properties and practices. However, those estimating absolute yield prediction may consider @rayburn_comparison_2022 as well as @woodard_efficiency_2017 to incorporate other variables like soil properties.
- Later research may also extend to genetic modification and climate-motivated practice changes which we assume to be latent in historic data.
- Though our interactive tools consider different spatial aggregations such as 5 character (4 x 5 km) geohashes, future work may consider modeling with actual field-level yield data and the actual risk unit structure. To that end, we observe that the actual unit yields / revenue and risk unit structure are not currently public.
- Due to the robustness of yield data, we examine Yield Protection but future agroeconomic study could extend this to the highly related Revenue Protection (RP) form of insurance. Indeed, the yield stresses seen in YP that we describe in this model may also impact RP.
- With additional data, future modeling could relax our normality assumption though, in addition to 79% of neighborhoods seeing approximately normal yield deltas, we observe that 88% of neighborhoods see approximate symmetry per @kim_statistical_2013 so that later work would likely not remove a systemic directional bias in results.
- This work focuses on systematic changes in growing conditions impacting claims rates across a broad geographic scale. It does not necessarily capture inclement weather events which may require an alternative form of modeling. However, as left for future work, this step could further impact claims rates at a localized level which may be relevant to programs with portfolios spanning a smaller geographic scale.

Note that, while we do anticipate changing historic yield averages in our simulations, we take a conservative approach and do not consider trend adjustment [@plastina_trend-adjusted_2014] and yield exclusion years [@schnitkey_yield_2015]. In raising $y_{expected}$, both would likely increase simulated loss rates.

## Interactive tools
In order to explore these simulations geographically and under different scenarios, interactive open source web-based visualizations built alongside our experiments both aid us in constructing our own conclusions and allow readers to consider possibilities and analysis beyond our own narrative. These are further described in our supplemental appendix with design considerations. Our software  runs within a web browser and is made publicly available at https://ag-adaptation-study.org for further exploration.

## Open source
We demonstrate this method within the specifics of the U.S. Corn Belt and the U.S. Federal Crop Insurance Program's Multi-Peril Crop Insurance. However, this work may also help aid future research into other crops such as soy, geographic areas such as other parts of the United States of America, and other programs such as Revenue Protection. Towards that end, we offer our work as an open source data science pipeline as further described at https://ag-adaptation-study.org.

# Conclusion
We present a method for prediction of institution-relevant crop yield changes using Monte Carlo on top of neural network-based regressors. We specifically simulate climate-driven system-wide impacts to maize growing conditions at a policy-relevant scale of granularity that may help institutional food system actors better understand the shape of these possible future climate threats. Within this study, we find that maize claim rates may double for the Yield Protection option within the U.S. Corn Belt relative to present values in a 2050 timescale.

In addition to publising our model outputs under a creative commons license, we explore the specific shape of these results. We describe a possible agriculturally-relevant geographic bias in climate impacts. We also highlight potential mathematical properties of interest that may manifest in possible upcoming changes in growing conditions predicted by prior work. This includes increasing volatility without decreasing average-based yield expectation measures which may have specific impacts to existing insurance structures. This kind of data-driven simulation-based perspective may help inform climate adaptation efforts.

Altogether, this study considers how this machine learning and interactive data science approach may understand existing food system policy structures in the context of climate projections. Towards that end, we release our underlying work under permissive open source licenses and make interactive tools available publicly at https://ag-adaptation-study.org. These visualizations also allow readers to explore alternatives to key analysis parameters presented in this paper.

# Data availability statement
Our software [@pottinger_data_2024], data, and pipeline [@pottinger_data_2024-1] are available on Zenodo as open source / creative common licensed resources as well as within a public git repository as further described at https://ag-adaptation-study.org.

# Acknowledgements
Study funded by the Eric and Wendy Schmidt Center for Data Science and Environment at the University of California, Berkeley. We have no conflicts of interest to disclose. Using yield estimation data from @lobell_scalable_2015 and @deines_million_2021 with our thanks to David Lobell for permission. We also wish to thank Magali de Bruyn, Jiajie Kong, Kevin Koy, and Ciera Martinez for conversation regarding these results. Thanks to Color Brewer [@brewer_colorbrewer_2013] and Public Sans [@general_services_administration_public_2024].

# Works Cited
