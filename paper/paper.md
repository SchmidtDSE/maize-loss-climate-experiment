---
bibliography: ./paper.bib
title: "Climate-Driven Doubling of Maize Loss Probability in USA Crop Insurance: Spatiotemporal Prediction and Possible Policy Responses"
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
date: 2024-07-08
affiliations:
  - id: 1
    name: Eric and Wendy Schmidt Center for Data Science and Environment, University of California Berkeley, Berkeley 94720, CA, USA
  - id: 2
    name: Department of Environmental Science, Policy & Management, University of California Berkeley, Berkeley 94720, CA, USA
  - id: 2
    name: Department of Agricultural Economics and Agribusiness, University of Arkansas, Fayetteville 72701, AR, USA
author:
  - name: A Samuel Pottinger \orcidlink{0000-0002-0458-4985}
    affil-id: 1, 2
    correspondence: yes
    email: sam.pottinger@berkeley.edu
  - name: Lawson Connor
    affil-id: 3
  - name: Brookie Guzder-Williams \orcidlink{0000-0001-6855-8260}
    affil-id: 1, 2
  - name: Maya Weltman-Fahs
    affil-id: 1, 2
  - name: Timothy Bowles \orcidlink{0000-0002-4840-3787}
    affil-id: 2
output:
  pdf_document:
    number_sections: yes
    template: default.tex
---

**Abstract:** Climate change not only threatens agricultural producers but also strains financial institutions. These important food system actors include government entities tasked with both insuring grower livelihoods and supporting response to evolving conditions. We use an artificial neural network to spatiotemporally predict future yields, finding alarming system-wide changes to institutional risk exposure within the US crop insurance program. Specifically, our machine learning method on maize in the US Corn Belt anticipates more frequent and more severe losses, predicting the annual probability of Actual Production History claims to more than double at mid-century relative to simulations without additional warming. Furthermore, in dialogue with prior regenerative agriculture studies, these data also highlight a possible mitigating opportunity in the form of including variance into APH coverage formulas. This structural shift may help regulators both support grower adaptation and address these imminent broad-scale financial challenges. Altogether, paired with open source interactive tools for deeper investigation, our risk profile simulations fill an actionable gap in current methodologies, bridging granular historic yield estimation and climate-informed prediction of future insurer-relevant loss.

\bigskip

# Introduction
Global warming threatens production of key staple crops, including maize [@rezaei_climate_2023]. Climate variability already drives a substantial proportion of year-to-year crop yield variation [@ray_climate_2015] and continued climate change may reduce planet-wide maize yields by up to 24% by the end of this century [@jagermeyr_climate_2021]. Beyond reduced mean output, growing frequency and severity of stressful weather [@dai_increasing_2013] to which maize is increasingly sensitive [@lobell_changes_2020] will also impact both farmers’ revenue [@sajid_extreme_2023] and the institutions designed to protect those producers [@hanrahan_crop_2024].

In the United States of America, the world’s largest maize producer and exporter [@ates_feed_2023], the Federal Crop Insurance Program (FCIP) covers a large share of this growing risk [@tsiboe_crop_2023]. The costs of crop insurance in the U.S. have already increased by 500% since the early 2000s with annual indemnities reaching $19B in 2022 [@schechinger_crop_2023]. Furthermore, retrospective analysis attributes 19% of “national-level crop insurance losses” between 1991 and 2017 to climate warming, an estimate rising to 47% during the drought-stricken 2012 growing season [@diffenbaugh_historical_2021]. Looking forward, @li_impact_2022 show progressively higher U.S. maize loss rates as warming elevates.

Modeling the possible changes in frequency and severity of crop loss events that trigger indemnity claims is an important step to prepare for the future impacts of global warming. Related studies have predicted changes in crop yields at county-level aggregation ([@leng_predicting_2020] and have estimated climate change impacts to U.S. maize within whole-sector or whole-economy analysis [@hsiang_estimating_2017]. Even so, as insurance products may include elements operating at the producer level [@rma_crop_2008], often missing are more granular models of insurer-focused metrics at the risk unit scale across a large region [@wang_warming_2021]. Such far-reaching but detailed data are prerequisite to designing proactive policy instruments benefiting both institutions and growers.

We address this need by predicting the probability and severity of maize loss within the U.S. Corn Belt at sub-county-level, probabilistically forecasting insurer-relevant outcome metrics under climate change. We find these projections using simulations of the Multiple Peril Crop Insurance Program, "the oldest and most common form of federal crop insurance" [@chite_agricultural_2006]. More precisely, we model changes to risk under the Yield Protection plan which covers farmers in the event of yield losses due to an insured cause. Furthermore, by contrasting those simulations to a counterfactual which does not include further warming, we then quantitatively highlight the insurer-relevant effects of climate change in the 2030 and 2050 timeframes. Finally, we use these data to suggest possible policy changes to help mitigate and enable adaptation to the specific climate-fueled risks we observe in our results.

\bigskip

# Methods
Seeking to provide this highly localized distributional information, we combine a neural network estimating changes to yield losses with Monte Carlo simulation under different climate conditions in order to estimate loss probability and severity.

## Formalization
Before modeling these systems, we articulate specific mathematical definitions of the attributes we seek to predict. First, from the insurer perspective, farm-level APH losses ($l$) represent yield below a guarantee threshold [@rma_crop_2008].

$l = max(c * y_{expected} - y_{actual}, 0)$

Though discussion explores alternatives, we generally use up to 10 years of historic yields data ($d=10$) with 75% coverage level ($c=0.75$) per FCIC guidelines [@fcic_crop_2023].

$y_{expected} = \frac{y_{historic}[-d:]}{d}$

Within this formulation we create working definitions of loss probability ($p$) and severity ($s$) in order to model changes to these insurer-relevant metrics.

$p = P(l > 0) = P(c * y_{expected} - y_{actual} > 0) = P(\frac{y_{actual} - y_{expected}}{y_{expected}} < c - 1) = P(y_{\Delta\%} < c - 1)$

$s = \frac{l}{y_{expected}} = \max(c - \frac{y_{actual}}{y_{expected}}, 0) = \max(-1 * y_{\Delta\%} - (1 - c), 0)$

Note that we define severity from the insurer perspective, reporting the percentage points gap between actual yield and the covered portion of expected yield.

## Data
As APH operates at unit-level, modeling these formulations requires highly local yield and climate information. Therefore, we use the Scalable Crop Yield Mapper (SCYM) which provides remote sensed yield estimations from 1999 to 2022 at 30m resolution across the US Corn Belt [@lobell_scalable_2015; @deines_million_2021]. Meanwhile, in order to predict these differential outcomes, we use climate data from CHC-CMIP6 [@williams_high_2024] which, at daily 0.05 degree scale, offers both historic data from 1983 to 2016 as well as future projections in a 2030 and 2050 series. In choosing from its two available shared socioeconomic pathways, we use the "intermediate" SSP245 within CHC-CMIP6 over SSP585 per @hausfather_emissions_2020. This offers the following climate variables for modeling: precipitation, temperature (minimum and maximum), relative humidity (n, x, average), heat index, wet bulb temperature, VPD, and SVP.

Though discussion explores alternatives, we align these variables to a common grid in order to create the discrete instances needed for model training and evaluation. To that end, we create "neighborhoods" [@manski_diversified_2024] of geographically proximate fields paired with climate data through 4 character geohashing [@niemeyer_geohashorg_2008], defining small populations in a grid of cells roughly 28 by 20 kilometers for use within statistical tests [@haugen_geohash_2020]. Having created these spatial groups, we model against observed distributions of yield deltas described as mean and standard deviation, helping ensure dimensionality is appropriate for the size of the input dataset. Finally, we similarly describe climate variable deltas as min, max, mean and standard deviation per month.

## Regression
With these data in mind, we next build predictive models for use in simulations of future insurance outcomes. Using machine learning per @leng_predicting_2020, we build regressors forecasting change in yields. However, we use a feed forward artificial neural network [@baheti_essential_2021] as it:

- Natively supports multi-variable output [@brownlee_deep_2020], helpful as we need to predict both future neighborhood mean and standard deviation together.
- Is well suited for out-of-sample range estimation [@mwiti_random_2023], important as warming may incur growing conditions outside the historical conditions seen in the input domain.

As described below, we fit ($f$) neighborhood-level climate variables ($C$) and year ($x$) against z-normalized yield deltas [@kim_investigating_2024].

$y_{\Delta\%}(x) = \frac{y_{actual} - y_{expected}}{y_{expected}} = \frac{y_{\Delta}}{y_{\mu-historic}} = f_{z}(C, x, z_{\mu-historic}, z_{\sigma-historic})$

Many different kinds of neural network structures could meet these criteria. Therefore, building individual networks using the Adam optimizer [@kingma_adam_2014], we also try various combinations of "hyper-parameters" in a grid search sweep [@joseph_grid_2018]. Most of these options address "overfitting" in which regressors learn non-generalizable trends from input data. Specifically, this performance-optimizing algorithm permutes different numbers of layers, dropout rates [@srivastava_dropout_2014], L2 regularization strengths [@tewari_regularization_2021], and removal of input attributes. In total, we select a preferred configuration from 2,400 candidate models before retraining on all available data ahead of simulations. In this design, all non-output neurons use Leaky ReLU activation per @maas_rectifier_2013. 

| **Parameter**                | **Options**                  | **Description**                                                                                                                                                       | **Purpose**                                                                                                                          |
| ----------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Layers                       | 1 - 6                        | Number of feed forward layers to include where 2 layers include 32 and then 8 nodes while 3 layers include 64, 32, and 8. Layer sizes are {512, 256, 128, 64, 32, 8}. | More layers might allow networks to learn more sophisticated behaviors but also might overfit to input data.                         |
| Dropout                      | 0.00, 0.01, 0.05, 0.10, 0.50 | This dropout rate applies across all hidden layers.                                                                                                                   | Random disabling of neurons may address overfitting.                                                                                 |
| L2                           | 0.000, 0.001, 0.010, 0.100   | This L2 regularization strength applies across all hidden layer neuron connections.                                                                                   | Penalizing networks with edges that are "very strong" may confront overfitting without changing the structure of the network itself. |
| Attr Drop                    | 10                           | Retraining where the sweep individually drops each of the dozen input distributions or year or keeps all inputs.                                                      | Removing attributes helps determine if an input may be unhelpful.                                                                    |
| Count                        | Yes / No                     | Indicate if the model can access the count of observations in a geohash.                                                                                              | Determine if having information availability is helpful.                                                                             |

## Simulation
After training machine learning models using historical data, predictions of future distributions feed into Monte Carlo simulations [@metropolis_beginning_1987; @kwiatkowski_monte_2022] in the 2030 and 2050 CHC-CMIP6 series [@williams_high_2024].

![Model pipeline overview diagram. Code released as open source.](./img/pipeline.png "Model pipeline overview diagram. Code released as open source."){ width=80% }

With trials consisting of sampling at the neighborhood scale, this approach allows us to consider many possible values to understand what the distribution of outcomes may look like in the future and make probability statements about insurance-relevant events.  In addition to sampling climate variables, we also "draw" different model error residuals to propagate uncertainty [@yanai_estimating_2010]. Furthermore, as the exact geospatial risk unit structure is not publicly known, these trials also sample multiple times from a neighborhood to approximate the size of an APH insured. In this operation, we also draw the unit size itself randomly per trial from historic data [@rma_statecountycrop_2024]. Altogether, this simulates a single unit (as further described in supplemental materials) per year for 5 years. Finally, as variance may differ between the two expected and counterfactual sets [@mcdonald_handbook_2014], we use Mann Whitney U [@mann_test_1947] with Bonferroni-correction [@bonferroni_il_1935] to determine significance of neighborhood-level changes per year. Here observe that, though offering predictions at 30 meter scale, SYCM uses Daymet variables at 1 km resolution [@thornton_daymet_2014] and, thus, we more conservatively assume this 1km granularity in determining sample sizes.

## Evaluation
We choose our model using each candidate’s capability to predict into future years, a "sweep temporal displacement" task representative of the Monte Carlo simulations with three steps [@brownlee_what_2020].

- Train on all data between 1999 to 2012 inclusive.
- Sweep using 2014 and 2016 as a validation set which lets us compare different candidate models.
- Test in which 2013 and 2015 serve as a fully hidden set in order to estimate how the chosen model may perform in the future.

This sweep structure fits with a relatively small dataset. Therefore, having performed model selection, we further evaluate our chosen regressor through three additional tests which more practically estimate performance in different ways one may consider using this method (random assignment, temporal displacement, and spatial displacement) while using a larger training set. 

| **Trial**             | **Purpose**                                               | **Train**                                                                                                   | **Test**                                         |
| ------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Random Assignment     | Evaluate ability to predict generally.                    | Random 75% of year / geohash combinations such that a geohash may be in training one year and test another. | The remaining 25% of year / region combinations. |
| Temporal Displacement | Evaluate ability to predict into future years.            | All data from 1999 to 2013 inclusive.                                                                       | All data 2014 to 2016 inclusive.                 |
| Spatial Displacement  | Evaluate ability to predict into unseen geographic areas. | All 4 character geohashes in a randomly chosen 75% of 3 character regions.                                  | Remaining 25% of regions.                        |

These post-hoc trials use only training and test sets as we fully retrain models using unchanging sweep-chosen hyper-parameters. Note that some of these tests use "regions" which we define as all geohashes sharing the same first three characters. This two tier definition creates a grid of 109 x 156 km cells [@haugen_geohash_2020] each including all neighborhoods (4 character geohashes) found within that area.

\bigskip

# Results
We project loss probabilities to roughly double ({{experimentalProbability2050}} claims rate) under SSP245 at mid-century in comparison to the no additional warming counterfactual scenario ({{counterfactualProbability2050}} claims rate) even as the average yield sees only minor changes from current values under the clime change simulation.

## Neural network outcomes
With slight bias towards the mean prediction task, we select {{numLayers}} layers ({{layersDescription}}) using {{dropout}} dropout and {{l2}} L2 from our sweep (count information {{countInfoStr}}). As further explored in supplemental materials, we use all input variables.

| **Set**    | **Mean Prediction MAE** | **Std Prediction MAE** |
| ---------- | ----------------------- | ---------------------- |
| Train      | {{trainMeanMae}}        | {{trainStdMae}}        |
| Validation | {{validationMeanMae}}   | {{validationStdMae}}   |
| Test       | {{testMeanMae}}         | {{testStdMae}}         |

The selected model sees some overfit but this trial fits with fewer data points than the simulations’ model. Therefore, having chosen this set of hyper-parameters, we further evaluate regression performance through varied definitions of test sets representing different tasks with a larger training set.

| **Task**              | **Test Mean Pred MAE** | **Test Std Pred MAE** | **% of Units in Test Set** |
| --------------------- | ---------------------- | --------------------- | -------------------------- |
| Temporal Displacement | {{temporalMeanMae}}    | {{temporalStdMae}}    | {{temporalPercent}}        |
| Spatial Displacement  | {{spatialMeanMae}}     | {{spatialStdMae}}     | {{spatialPercent}}         |
| Random                | {{randomMeanMae}}      | {{randomStdMae}}      | {{randomPercent}}          |

In these trials, the temporal displacement task best resembles expected error in simulations.


## Simulation outcomes
{{percentSignificant}} of neighborhoods in SSP245 across both the 2030 and 2050 see significant changes to claim probability in at least one year ($p<0.05/n$). We observe that the remaining neighborhoods failing to meet that threshold often have less land dedicated to corn within their area and, thus, a smaller sample size in our simulations.

| **Scenario**   | **Year** | **Unit mean yield change** | **Unit loss probability**         | **Avg covered loss severity**  |
| ---------------------------- | -------- | -------------------------- | --------------------------------- | ------------------------------ |
| Counterfactual | 2030     | {{counterfactualMean2030}} | {{counterfactualProbability2030}} | {{counterfactualSeverity2030}} |
| SSP245         | 2030     | {{experimentalMean2030}}   | {{experimentalProbability2030}}   | {{experimentalSeverity2030}}   |
| Counterfactual | 2050     | {{counterfactualMean2050}} | {{counterfactualProbability2050}} | {{counterfactualSeverity2050}} |
| SSP245         | 2050     | {{experimentalMean2050}}   | {{experimentalProbability2050}}   | {{experimentalSeverity2050}}   |
|                |          | $y_{\Delta \mu}$           | $p_{\mu}$                         | $s_{\mu}$                      |

In highlighting these climate threats, these simulations suggest that warming disrupts historic trends of increasing average yield [@nielsen_historical_2023]. However, while these SSP245 mean remains similar to the historic baseline, the distribution tails differ more substantially, showing more than double the counterfactual loss probability in the 2050 series compared to counterfactual. This represents a decrease from the 2030 series as the average within the APH partially catches up within simulations.

![One of our interactive tools showing 2050 outcomes distribution relative to $y_{expected}$ highlighting loss with and without climate change.](./img/hist.png "One of our interactive tools showing 2050 outcomes distribution relative to $y_{expected}$ highlighting loss with and without climate change."){ width=85% }

Note that, though we observe these changes across a wide geographic area, the data show sophisticated interactions between inputs where overlapping climate variable trends create a wide variety of specific outcomes across the study geography as further explorable in the interactive visualizations described later. Additionally, as described in supplemental materials, our open source pipeline allows for configuration of this simulation including switching to field instead of risk unit. As anticipated, we observe that simulation at field level increases the claims rate in all scenarios, confirming the expected portfolio effect.

\bigskip

# Discussion
In addition to highlighting future work opportunities, we observe a number of policy-relevant dynamics within our simulations.

## Stress
First, we note that our neural network depresses yields during combined warmer and drier conditions potentially similar to 2012 which saw poor US maize outcomes [@ers_weather_2013]. Our predictions may add additional evidence to prior empirical studies such as @sinsawat_effect_2004 and @marouf_effects_2013 which describe the negative impacts of heat stress and water deficits. Indeed, this prior work may explain why precipitation may serve as a protective factor: those with drier July conditions are more likely to see higher loss probability (p < 0.05 / 2) in both the 2030 and 2050 series via rank correlation [@spearman_proof_1904] though $\rho = -0.2$, suggesting many simultaneous factors at play..

![Screenshot of one of our interactive tools, showing precipitation and loss probability changes where precipitation may offer some protective benefit.](./img/scatter.png "Screenshot of one of our interactive tools, showing precipitation and loss probability changes where precipitation may offer some protective benefit."){ width=85% }

All that said, this concurrence between "top-down" remote sensing and "bottom-up" physical experimentation not only offers further confidence in these results but these specific results may reveal geographically and temporally specific outlines of these threats possibly useful for insurer and grower adaptation.

## Policy implications
Adaptation to these adverse conditions is imperative for both farmers and insurers [@oconnor_covering_2017; @mcbride_redefining_2020.] In order to confront this alarming increase in climate-driven risk, preparations may include altered planting dates [@mangani_projecting_2023],  physically moving operations [@butler_adaptation_2013], employing stress-resistant varieties [@tian_genome_2023], modifying pesticide usage [@keronen_management_2023], and adopting risk-mitigating regenerative farming systems [@renwick_long-term_2021]. Even so, significant structural and financial barriers inhibit adoption of such mitigations [@mcbride_redefining_2020] and, though the magnitude remains contested [@connor_crop_2022], financial safety net programs like crop insurance may actually disincentivize adoption of regenerative systems [@wang_warming_2021; @chemeris_insurance_2022] despite likely benefits for both farmers and insurers [@oconnor_covering_2017].

In this context, these simulations reveal how this particular predicted combination of stable yield averages ($y_{expected}$) paired with higher loss probabilities ($l$) not only poses particularly difficult challenges to the MPCI but offers unique adaptation opportunities. Structurally, average-based APH production histories [@fcic_common_2020] reward increases in mean yield but may be poorly situated to confront evolving stability. Indeed, despite possibly guarding against this elevation in loss events [@bowles_long-term_2020; @renwick_long-term_2021], this present day structure may discourage resilience-promoting regenerative agriculture methods which may not improve mean yields or even come at the cost of a slightly reduced average [@deines_recent_2023]. 

This in mind, this study recommends that coverage levels could be redefined from the current percentage based approach ($l_{\%}$) to variance ($l_{\sigma}$), thus rewarding improvements both in average yield and stability.

| **Current formulation**                           | **Possible proposal**                                                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| $l_{\%} = \max(c_{\%} * y_{\mu} - y_{acutal}, 0)$ | $l_{\sigma} = \max(\frac{c_{\sigma} * y_{\mu}}{y_{\sigma}} - y_{acutal}, 0)$ |

For example, using historic values as guide, {{equivalentStd}} standard deviations ($c_\sigma$) would achieve the current system-wide coverage levels ($c_\% = 0.75$) but realign APH incentives towards a balance between a long-standing aggregate output incentive and a new resilience reward that could recognize regenerative systems and other similar practices, valuing the stability offered by some producers for the broader food system [@renwick_long-term_2021].

![Histogram showing percent change from $y_{expected}$ for one standard deviation in each simulated unit.](./img/std.png "Histogram showing percent change from $y_{expected}$ for one standard deviation in each simulated unit."){ width=80% }

Even so, federal statute caps coverage levels as percentages of production history [@cfr_crop_nodate]. Therefore, though regulators may make improvements like through 508h without congressional action, our simulations possibly suggest that the ability to incorporate climate adaptation variability may remain limited without statutory change. 

Regardless, $l_\sigma$ enables rate setters to directly reward outcomes instead of individual practices and combining across management tools may address unintended consequences of elevating individual risk management options [@connor_crop_2022]. Though recognizing the limits of insurance alone and echoing prior calls for multi-modal support for regenerative practices [@mcbride_redefining_2020], this outcomes-based approach may enable insurance to more directly participate in climate adaptation.

## Alternative models
We also highlight additional future modeling opportunities beyond the scope of this study.

- We evaluate yield deltas and include historic yield as inputs into our neural network, allowing those data to “embed” adaptability measures [@hsiang_estimating_2017] such as soil properties and practices. However, those estimating absolute yield prediction may consider @rayburn_comparison_2022 as well as @woodard_efficiency_2017 to incorporate other variables like soil properties.
- Later research may also extend to genetic modification and climate-motivated practice changes which we assume to be latent in historic data. 
- Though our supplemental materials consider alternatives to SCYM and different spatial aggregations such as 5 character (4 x 5 km) geohashes, future work may consider modeling with actual field-level yield data and the actual risk unit structure. To that end, we observe that the actual unit yields / revenue and risk unit structure are not currently public.
- We examine the yield guarantee form of the APH but future agroeconomic study could extend this to the highly related revenue guarantee form.

Note that, while we do anticipate changing historic yield averages in our simulations, we take a conservative approach and do not consider trend adjusted APH [@plastina_trend-adjusted_2014] and yield exclusion years [@schnitkey_yield_2015]. In raising $y_{expected}$, both would likely increase simulated loss rates.

## Interactive tools
In order to explore these simulated distributions geographically and under different scenarios, interactive open source web-based visualizations built alongside our experiments both aid in constructing our own conclusions and allow readers to consider possibilities and analysis beyond our own narrative.

| **Simulator**   | **Question**                                                                    | **Loop**                                                                                                                                                                   | **JG**                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Hyper-Parameter | How do hyper-parameters impact regressor performance?                           | Iteratively change neural network hyper-parameters to see influence on validation set performance.                                                                         | Improving on previous hyper-parameter hypotheses. |
| Distributional  | How do overall simulation results change under different simulation parameters? | Iterative manipulation of parameters (geohash size, event threshold) to change loss probability and severity.                                                              | Deviating from the study’s main results.          |
| Neighborhood    | How do simulation results change across geography and climate conditions?       | Inner loop changing simulation parameters to see changes in neighborhood outcomes. Outer loop of observing changes across different views. | Identifying neighborhood clusters of concern.     |
| Claims          | How do different regulatory choices influence grower behavior?                  | Iteratively change production history to see which years result in claims under different regulatory schemes.                                                              | Redefining policy to improve yield stability.     |
In crafting these "explorable explanations" [@victor_explorable_2011], we draw analogies to micro-apps  [@bridgwater_what_2015] or mini-games [@dellafave_designing_2014] in which the user encounters a series of small experiences that, each with distinct interaction and objectives, can only provide minimal instruction [@brown_100_2024]. As these visualizations cannot take advantage of design techniques like Hayashida-style tutorials [@pottinger_pyafscgaporg_2023], they rely on simple "loops" [@brazie_designing_2024] for immediate "juxtaposition gratification" (JG) [@jm8_secret_2024], showing fast progression after minimal input.

![Example simulation in our interactive tool’s geographic view.](./img/map.png "Example simulation in our interactive tool’s geographic view."){ width=85% }

We follow the framework from @unwin_why_2020 and note that our custom tools first serve as internal "exploratory" graphics enabling the insights detailed in our results, outlining specific observations we attribute to the use of these tools.

| **Simulator**   | **Observation**                                                                                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Distributional  | Dichotomy of larger changes to insurer-relevant tails contrasting smaller changes to mean yield                                         |
| Claims          | Issues of using average in APH [@fcic_common_2020]                                                                                                         |
| Neighborhood    | Model output relationships with broader climate factors, highlighting the possible systemic protective value of increased precipitation |
| Hyper-parameter | Model resilience to removing individual inputs.                                                                                         |

Continuing to "presentation" within @unwin_why_2020, we next release these tools into a public open source website at [https://ag-adaptation-study.pub](https://ag-adaptation-study.pub) and, specifically, we highlight that our interactive visualizations allow for further exploration of our modeling such as different loss thresholds for other insurance products, finding relationships of outcomes to different climate variables, answering geographically specific questions beyond the scope of this study, and modification of machine learning parameters to understand performance. In preparation for this open science release, we also report briefly on experiences from a 9 person "real-world" workshop co-exploring these results similar to @pottinger_combining_2023 which offered feedback^[We collect information about the tool only, falling under ”quality assurance” activity. IRB questionnaire on file.] on and changed the designs of these tools.

- Facilitators and participants report simple single loop [@brazie_designing_2024] designs performing best and that the two loop neighborhood simulator requires too much time to digest, prompting us to frame that experience as supplemental in the public release of the software.
- Facilitators observe workshop discussion contrasting different configurations so the design for juxtaposition gratification may achieve this intended comparative form of analysis [@jm8_secret_2024]. Even so, feedback also indicates that this technique’s reliance on "visual working memory" [@ware_information_2021] strains over many interactions so we add the option to see all results at once for the meta-parameter simulation for public release.

While not controlled experimentation [@lewis_using_1982] or diary studies [@shneiderman_strategies_2006], these separately offer valuable qualitative insights informing real-world use of this explorable explanation [@victor_explorable_2011].

\bigskip

# Conclusion
Maize production not only suffers from warming’s effects [@jagermeyr_climate_2021] but also adds to future climate change [@kumar_assessment_2021]. Inside this cyclic relationship, agriculture could contribute to the solution for the same global crisis that ails it [@schon_cover_2024]. In dialogue with prior work [@wang_warming_2021; @chemeris_insurance_2022], we highlight how APH currently prioritizes increased mean yield over climate adaptations benefiting stability. Furthermore, we demonstrate how existing policy structures could fail to adjust to our predicted distributional changes and how inclusion of variance into this regulatory structure may positively influence adoption of mitigating practices such as regenerative systems. This responsive shift in coverage levels could reorient incentives: balancing overall yield with longitudinal stability and more comprehensively incorporating an understanding of risk. These changes may benefit both grower and insurer without requiring practice-specific regulation. Recognizing that these structural levers require modification by policy makers, we therefore encourage scientists to further study, regulators / lawmakers to further consider, and producers to further inform these revisions to APH. These essential multi-stakeholder efforts are crucial in preparing the US food system and its insurance program for a warmer future.

\bigskip

# Acknowledgements
Study funded by the Eric and Wendy Schmidt Center for Data Science and Environment at the University of California, Berkeley. We have no conflicts of interest to disclose. Using yield estimation data from @lobell_scalable_2015 and @deines_million_2021 with our thanks to David Lobell for permission. We also wish to thank Magali de Bruyn, Nick Gondek, Jiajie Kong, Kevin Koy, and Ciera Martinez for conversation regarding these results. Thanks to Color Brewer [@brewer_colorbrewer_2013].

\bigskip

# Works Cited