---
bibliography: ./paper.bib
title: "Climate-Driven Doubling of Maize Loss Probability in U.S. Crop Insurance: Spatiotemporal Prediction and Possible Policy Responses"
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
  - name: Timothy Bowles \orcidlink{0000-0002-4840-3787}
    affil-id: 3
output:
  pdf_document:
    number_sections: yes
    template: default.tex
---

**Abstract:** Climate change not only threatens agricultural producers but also strains financial institutions. These important food system actors include government entities tasked with both insuring grower livelihoods and supporting response to continued global warming. We use an artificial neural network to predict future maize yields in the U.S. Corn Belt, finding alarming changes to institutional risk exposure within the Federal Crop Insurance Program. Specifically, our machine learning method anticipates more frequent and more severe yield losses that would result in the annual probability of Yield Protection (YP) claims to more than double at mid-century relative to simulations without continued climate change. Furthermore, our dual finding of relatively unchanged average yields paired with decreasing yield stability reveals targeted opportunities to adjust coverage formulas to include variability. This important structural shift may help regulators support grower adaptation to continued climate change by recognizing the value of risk-reducing strategies such as regenerative agriculture. Altogether, paired with open source interactive tools for deeper investigation, our risk profile simulations fill an actionable gap in current understanding, bridging granular historic yield estimation and climate-informed prediction of future insurer-relevant loss.

\bigskip

# Introduction
Global warming threatens production of key staple crops, including maize [@rezaei_climate_2023]. Climate variability already drives a substantial proportion of year-to-year crop yield variation [@ray_climate_2015] and continued climate change may reduce planet-wide maize yields by up to 24% by the end of this century [@jagermeyr_climate_2021]. Beyond reduced mean output, growing frequency and severity of stressful weather [@dai_increasing_2013] to which maize is increasingly sensitive [@lobell_changes_2020] will also impact both farmers’ revenue [@sajid_extreme_2023] and the institutions designed to protect those producers [@hanrahan_crop_2024].

In the United States of America, the world’s largest maize producer and exporter [@ates_feed_2023], the Federal Crop Insurance Program (FCIP) covers a large share of this growing risk [@tsiboe_crop_2023]. The costs of crop insurance in the U.S. have already increased by 500% since the early 2000s with annual indemnities reaching $19B in 2022 [@schechinger_crop_2023]. Furthermore, retrospective analysis attributes 19% of "national-level crop insurance losses" between 1991 and 2017 to climate warming, an estimate rising to 47% during the drought-stricken 2012 growing season [@diffenbaugh_historical_2021]. Looking forward, @li_impact_2022 show progressively higher U.S. maize loss rates as warming elevates.

Modeling the possible changes in frequency and severity of crop loss events that trigger indemnity claims is an important step to prepare for the future impacts of global warming. Related studies have predicted changes in crop yields at county-level aggregation [@leng_predicting_2020] and have estimated climate change impacts to U.S. maize within whole-sector or whole-economy analysis [@hsiang_estimating_2017]. Even so, as insurance products may include elements operating at the producer level [@rma_crop_2008], often missing are more granular models of insurer-focused claims rate and loss severity at the level of the risk unit^[The "insured unit" or "risk unit" refers to set of insured fields or an insured area within an individual policy.] across a large region. Such far-reaching but detailed data are prerequisite to designing proactive policy instruments benefiting both institutions and growers.

We address this need by predicting the probability and severity of maize loss within the U.S. Corn Belt at the level of insured units, probabilistically forecasting insurer-relevant outcome metrics under climate change. We find these projections using simulations of the Multiple Peril Crop Insurance Program, "the oldest and most common form of federal crop insurance" [@chite_agricultural_2006]. More precisely, we model changes to risk under the Yield Protection (YP) plan, which covers farmers in the event of yield losses due to an insured cause. Furthermore, by contrasting those simulations to a "counterfactual" which does not include further climate warming, we then quantitatively highlight the insurer-relevant effects of climate change in the 2030 and 2050 timeframes. Finally, we use these data to suggest possible policy changes to help mitigate and enable adaptation to the specific climate-fueled risks we observe in our results.

\bigskip

# Methods
We first build predictive models of crop yield distributions using a neural network at a spatial scale relevant to insurers. We then estimate changes to yield losses under different climate conditions with Monte Carlo simulation in order to estimate loss probability and severity.

## Formalization
Before modeling these systems, we articulate specific mathematical definitions of the attributes we seek to predict. First, from the insurer perspective, unit-level losses ($l$) under the YP program represents yield below a guarantee threshold [@rma_crop_2008] with terms defined below.

$l = max(c * y_{expected} - y_{actual}, 0)$

We use up to 10 years of historic yields data ($d=10$) with 75% coverage level ($c=0.75$) per Federal Crop Insurance Corporation guidelines [@fcic_crop_2023], with our interactive tools allowing for consideration of different coverage levels.

$y_{expected} = \frac{y_{historic}[-d:]}{d}$

Within this formulation we create working definitions of loss probability ($p$) and severity ($s$) in order to model changes to these insurer-relevant metrics.

$p = P(l > 0) = P(c * y_{expected} - y_{actual} > 0) = P(\frac{y_{actual} - y_{expected}}{y_{expected}} < c - 1) = P(y_{\Delta\%} < c - 1)$

$s = \frac{l}{y_{expected}} = \max(c - \frac{y_{actual}}{y_{expected}}, 0) = \max(-1 * y_{\Delta\%} - (1 - c), 0)$

Note that we define severity from the insurer perspective, reporting the percentage points gap between actual yield and the covered portion of expected yield.

## Data
As YP operates at unit-level, modeling these formulations requires highly local yield and climate information. Therefore, we use the Scalable Crop Yield Mapper (SCYM) which provides remotely sensed yield estimations from 1999 to 2022 at 30m resolution across the US Corn Belt [@lobell_scalable_2015; @deines_million_2021]. Meanwhile, we use climate data from CHC-CMIP6 [@williams_high_2024] which, at daily 0.05 degree scale, offers both historic data from 1983 to 2016 as well as future projections in a 2030 and 2050 series. In choosing from its two available shared socioeconomic pathways, we prefer the "intermediate" SSP245 within CHC-CMIP6 over SSP585 per @hausfather_emissions_2020. This offers the following climate variables for modeling: precipitation, temperature (minimum and maximum), relative humidity (average, peak), heat index, wet bulb temperature, vapor pressure deficit, and saturation vapor pressure.

We align these variables to a common grid in order to create the discrete instances needed for model training and evaluation. More specifically, we create "neighborhoods" [@manski_diversified_2024] of geographically proximate fields paired with climate data through 4 character^[We also evaluate alternative neighborhood sizes in the interactive tools.] geohashing [@niemeyer_geohashorg_2008], defining small populations in a grid of cells roughly 28 by 20 kilometers for use within statistical tests [@haugen_geohash_2020]. Having created these spatial groups, we model against observed deviations from yield expectations which YP defines through historic yield ($y_{expected}$). This creates a distribution of changes or "yield deltas" which we summarize as neighborhood-level means and standard deviations. This helps ensure appropriate dimensionality for the dataset size given approximate normalilty in 79% of geohashes per @kim_statistical_2013. Finally, we similarly describe climate deltas as min, max, mean and standard deviation per month.

## Regression
With these data in mind, we next build predictive models for use in simulations of future insurance outcomes. Using machine learning per @leng_predicting_2020, we build regressors ($f$) by fitting neighborhood-level climate variables ($C$) and year ($x$) against neighborhood-level mean and standard deviation of yield changes [@kim_investigating_2024].

$y_{\Delta\%}(x) = \frac{y_{actual} - y_{expected}}{y_{expected}} = \frac{y_{\Delta}}{y_{\mu-historic}} = f(C, x, z_{\mu-historic}, z_{\sigma-historic})$

For $f$, we use feed forward artificial neural networks [@baheti_essential_2021] as they support:

- Multi-variable output [@brownlee_deep_2020], helpful as we need to predict both mean and standard deviation.
- Out-of-sample range estimation [@mwiti_random_2023], important as warming may incur conditions outside the historical record.

Many different kinds of neural network structures and configurations could meet these criteria. Therefore, we try various combinations of "hyper-parameters" in a grid search sweep [@joseph_grid_2018]. 

| **Parameter**                | **Options**                  | **Description**                                                                                                                                                       | **Purpose**                                                                                                                          |
| ----------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Layers                       | 1 - 6                        | Number of feed forward layers to include where 2 layers include 32 and then 8 nodes while 3 layers include 64, 32, and 8. Layer sizes are {512, 256, 128, 64, 32, 8}. | More layers might allow networks to learn more sophisticated behaviors but also might overfit to input data.                         |
| Dropout                      | 0.00, 0.01, 0.05, 0.10, 0.50 | This dropout rate applies across all hidden layers.                                                                                                                   | Random disabling of neurons may address overfitting.                                                                                 |
| L2                           | 0.00, 0.05, 0.10, 0.15, 0.20   | This L2 regularization strength applies across all hidden layer neuron connections.                                                                                   | Penalizing networks with edges that are "very strong" may confront overfitting without changing the structure of the network itself. |
| Attr Drop                    | 10                           | Retraining where the sweep individually drops each of the input distributions or year or keeps all inputs.                                                      | Removing attributes helps determine if an input may be unhelpful.                                                                    |

Table: Parameters which we try in different permutations to find an optimal configuration. {#tbl:sweepparam}

We permute different option combinations from Table @tbl:sweepparam before we select a preferred configuration^[All non-output neurons use Leaky ReLU activation per @maas_rectifier_2013 and we use AdamW optimizer [@kingma_adam_2014; @loshchilov_decoupled_2017].] from 1,500 candidate models. Finally, with meta-parameters chosen, we can then retrain on all available data ahead of simulations.

## Simulation
After training machine learning models using historical data, predictions of future distributions feed into Monte Carlo simulations [@metropolis_beginning_1987; @kwiatkowski_monte_2022] in the 2030 and 2050 CHC-CMIP6 series [@williams_high_2024] as described in Figure @fig:pipeline. With trials consisting of sampling at the neighborhood scale, this approach allows us to consider many possible values to understand what the distribution of outcomes may look like in the future. These results then enable us to make probability statements about insurance-relevant events such as claims rate. In addition to sampling climate variables and model error residuals to propagate uncertainty [@yanai_estimating_2010], we also draw multiple times to approximate the size of an insured unit^[In this operation, we also draw the unit size itself randomly per trial from historic data [@rma_statecountycrop_2024].] as the exact geospatial risk unit structure is not publicly known. Altogether, this simulates each unit individually per year. Finally, we determine significance via Bonferroni-corrected [@bonferroni_il_1935] Mann Whitney U [@mann_test_1947] per neighborhood per year as variance may differ between the two expected and counterfactual sets [@mcdonald_handbook_2014]. 

![Model pipeline overview diagram. Code released as open source.](./img/pipeline.png "Model pipeline overview diagram. Code released as open source."){ width=80% #fig:pipeline }

Note that, though offering predictions at 30 meter scale, SYCM uses Daymet variables at 1 km resolution [@thornton_daymet_2014] and, thus, we more conservatively assume this 1km granularity in determining sample sizes.

## Evaluation
We choose our model using each candidate’s capability to predict into future years, a "sweep temporal displacement" task representative of the Monte Carlo simulations [@brownlee_what_2020]:

- Train on all data between 1999 to 2012 inclusive.
- Use 2014 and 2016 as validation set to compare the 1,500 candidates.
- Test in which 2013 and 2015 serve as a fully hidden set in order to estimate how the chosen model may perform in the future.

Having performed model selection, we further evaluate our chosen regressor through three additional tests which more practically estimate performance in different ways one may consider using this method (see Table @tbl:posthoc) while using a larger training set.

| **Trial**             | **Purpose**                                               | **Train**                                                                                                   | **Test**                                         |
| ------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Random Assignment     | Evaluate ability to predict generally.                    | Random 75% of year / geohash combinations such that a geohash may be in training one year and test another. | The remaining 25% of year / region combinations. |
| Temporal Displacement | Evaluate ability to predict into future years.            | All data from 1999 to 2013 inclusive.                                                                       | All data 2014 to 2016 inclusive.                 |
| Spatial Displacement  | Evaluate ability to predict into unseen geographic areas. | All 4 character geohashes in a randomly chosen 75% of 3 character regions.                                  | Remaining 25% of regions.                        |

Table: Overview of trials after model selection. {#tbl:posthoc}

These post-hoc trials use only training and test sets as we fully retrain models using unchanging sweep-chosen hyper-parameters. Note that some of these tests use "regions" which we define as all geohashes sharing the same first three characters. This two tier definition creates a grid of 109 x 156 km cells [@haugen_geohash_2020] each including all neighborhoods (4 character geohashes) found within that area.

\bigskip

# Results
We project loss probabilities to more than double ({{experimentalProbability2050}} claims rate) under SSP245 at mid-century in comparison to the no additional warming counterfactual scenario ({{counterfactualProbability2050}} claims rate).

## Neural network outcomes
With bias towards performance in mean prediction, we select {{numLayers}} layers ({{layersDescription}}) using {{dropout}} dropout and {{l2}} L2 from our sweep with all data attributes included. Table @tbl:sweep describes performance for the chosen configuration.

| **Set**             | **MAE for Mean Prediction** | **MAE for Std Prediction** |
| ------------------- | ----------------------- | ---------------------- |
| Train               | {{trainMeanMae}}        | {{trainStdMae}}        |
| Validation          | {{validationMeanMae}}   | {{validationStdMae}}   |
| Test before retrain | {{testMeanMae}}         | {{testStdMae}}         |
| Test after retrain  | {{retrainMeanMae}}      | {{retrainStdMae}}      |

Table: Results of chosen configuration during the "sweep" for model selection. {#tbl:sweep}

After retraining with train and validation together, we see {{retrainMeanMae}} MAE when predicting neighborhood mean and {{retrainStdMae}} when predicting neighborhood standard deviation when using the fully hidden test set.

Next, having chosen this set of hyper-parameters, we also evaluate regression performance through varied definitions of test sets representing different tasks.

| **Task**              | **Test Mean Pred MAE** | **Test Std Pred MAE** | **% of Units in Test Set** |
| --------------------- | ---------------------- | --------------------- | -------------------------- |
| Temporal | {{temporalMeanMae}}    | {{temporalStdMae}}    | {{temporalPercent}}        |
| Spatial  | {{spatialMeanMae}}     | {{spatialStdMae}}     | {{spatialPercent}}         |
| Random                | {{randomMeanMae}}      | {{randomStdMae}}      | {{randomPercent}}          |

Table: Results of tests after model selection. {#tbl:posthocresults}

From the trials outlined in Table @tbl:posthocresults, the temporal task best resembles expected error in simulations as they predict into the future. The interactive tools website allows for further examination of error.

## Simulation outcomes
After retraining on all available data using the selected configuration from our sweep, Monte Carlo simulates overall outcomes. Despite the conservative nature of the Bonferroni correction [@mcdonald_handbook_2014] and the 1km sample assumption, {{percentSignificant}} of maize acreage in SSP245 falls within a neighborhood with significant changes to claim probability ($p < 0.05 / n$) at some point during the 2050 series simulations. That said, we observe that some of the remaining neighborhoods failing to meet that threshold have less land dedicated to maize within their area and, thus, a smaller sample size in our simulations.

| **Scenario**   | **Year** | **Unit mean yield change** | **Unit loss probability**         | **Avg covered loss severity**  |
| ---------------------------- | -------- | -------------------------- | --------------------------------- | ------------------------------ |
| Counterfactual | 2030     | {% if counterfactualMean2030|float > 0 %}+{% endif %}{{counterfactualMean2030}} | {{counterfactualProbability2030}} | {{counterfactualSeverity2030}} |
| SSP245         | 2030     | {% if experimentalMean2030|float > 0 %}+{% endif %}{{experimentalMean2030}}   | {{experimentalProbability2030}}   | {{experimentalSeverity2030}}   |
| Counterfactual | 2050     | {% if counterfactualMean2050|float > 0 %}+{% endif %}{{counterfactualMean2050}} | {{counterfactualProbability2050}} | {{counterfactualSeverity2050}} |
| SSP245         | 2050     | {% if experimentalMean2050|float > 0 %}+{% endif %}{{experimentalMean2050}}   | {{experimentalProbability2050}}   | {{experimentalSeverity2050}}   |
|                |          | $y_{\Delta \mu}$           | $p_{\mu}$                         | $s_{\mu}$                      |

Table: Overview of Monte Carlo simulation results. Counterfactual is a future without continued warming in contrast to SSP245. {#tbl:simresults}

Regardless, with Table @tbl:simresults highlighting these climate threats, these simulations suggest that warming disrupts historic trends of increasing average yield [@nielsen_historical_2023]. Furthermore, in addition to wiping out the gains that our neural network would otherwise expect without climate change, the loss probability increases in both of the time frames considered for SSP245. Indeed, as shown in Figure @fig:hist, the SSP245 overall yield mean remains similar to the historic baseline in the 2050 series even as distribution tails differ more substantially.

![Interactive tool screenshot showing 2050 outcomes distribution (changes from $y_{expected}$), highlighting loss with and without climate change. In addition to showing increased claims rate, this also depicts climate change reducing the expected increase in yields that would otherwise follow historic trends.](./img/hist.png "One of our interactive tools showing 2050 outcomes distribution relative to $y_{expected}$ highlighting loss with and without climate change."){#fig:hist }

Granular simulation results reflect this system-wide finding: {{ dualIncreasePercent2050 }} of neighborhoods seeing instances of higher claims rates under SSP245 in the 2050 series simultaneously report overall multi-year average yields remaining unchanged or even increasing. This observation around stability and changing tails shows how yield volatility could allow a sharp elevation in loss probability without necessarily decreasing overall mean yields.

\bigskip

# Discussion
In addition to highlighting future work opportunities, we observe a number of policy-relevant dynamics within our simulations.

## Adaptation and policy structure
Adaptation to these adverse conditions is imperative for both farmers and insurers [@oconnor_covering_2017; @mcbride_redefining_2020.]. In order to confront this alarming increase in climate-driven risk, preparations may include:

 - Altering planting dates [@mangani_projecting_2023].
 - Physically moving operations [@butler_adaptation_2013].
 - Employing stress-resistant varieties [@tian_genome_2023].
 - Modifying pesticide usage [@keronen_management_2023].
 - Adopting risk-mitigating regenerative farming systems [@renwick_long-term_2021].

Most notably, regenerative practices can reduce risks through diverse crop rotations [@bowles_long-term_2020] and improvements to soil health [@renwick_long-term_2021]. These important steps may provide output stability in addition to other environmental benefits [@hunt_fossil_2020], valuable resilience given that our results see higher claims not through overall reduced averages but higher volatility. Still, significant structural and financial barriers inhibit adoption of such systems [@mcbride_redefining_2020]. In particular, though the magnitude remains the subject of empirical investigation [@connor_crop_2022], financial safety net programs like crop insurance may reduce adoption [@wang_warming_2021; @chemeris_insurance_2022] despite likely benefits for both farmers and insurers [@oconnor_covering_2017].

One specific feature of U.S. crop insurance policy may be partly responsible for this disincentive: average-based production histories [@fcic_common_2020] structurally reward increases in mean yield but not necessarily yield stability. Indeed, regenerative agricultural practices may not always improve mean yields or can even come at the cost of a slightly reduced average [@deines_recent_2023] even though they guard against elevations in the probability of loss events [@renwick_long-term_2021].

That in mind, if coverage levels are redefined from the current percentage based approach ($l_{\%}$) to variance ($l_{\sigma}$) as shown in Table @tbl:covformula, then improvements both in average yield and stability could be rewarded. For example, using historic values as guide, {{equivalentStd}} standard deviations ($c_\sigma$) would achieve the current system-wide coverage levels ($c_\% = 0.75$) but realign incentives towards a balance between a long-standing aggregate output incentive and a new resilience reward that could recognize regenerative systems and other strategies that reduce variability, valuing the stability offered by some producers for the broader food system [@renwick_long-term_2021].

| **Current formulation**                           | **Possible proposal**                                                        |
| -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| $l_{\%} = \max(c_{\%} * y_{\mu} - y_{acutal}, 0)$ | $l_{\sigma} = \max(\frac{c_{\sigma} * y_{\mu}}{y_{\sigma}} - y_{acutal}, 0)$ |

Table: Change in coverage formulas. {#tbl:covformula}

Table @fig:stdev further describes this translation between standard deviation and historic coverage levels. Even so, federal statute may cap coverage levels as percentages of production history [@cfr_crop_nodate]. Therefore, though regulators may make improvements like through 508h without congressional action, our simulations possibly suggest that the ability to incorporate climate adaptation variability may remain limited without statutory change.

![Histogram showing percent change from $y_{expected}$ for one standard deviation in each simulated unit.](./img/std.png "Histogram showing percent change from $y_{expected}$ for one standard deviation in each simulated unit."){ width=95% #fig:stdev }

Regardless, $l_\sigma$ enables rate setters to directly reward outcomes instead of individual practices and combining across management tools may address unintended consequences of elevating individual risk management options [@connor_crop_2022]. Though recognizing the limits of insurance alone and echoing prior calls for multi-modal support for regenerative systems [@mcbride_redefining_2020], this outcomes-based approach may enable insurance to more directly support climate adaptation without picking specific systems or practices to incentivize.

## Geographic bias
Neighborhoods with significant results ($p < 0.05 / n$) may be more common in some areas as shown in Figure @fig:geo. This spatial pattern may partially reflect that a number of neighborhoods have less land dedicated to maize so simulations have smaller sample sizes and fail to reach significance. In other cases, this geographic effect may also reflect disproportionate stress or other changes relative to historic conditions. In particular, as further explorable in our interactive tools, we note some geographic bias in changes to precipitation, temperature, and VPD / SVP.

![Interactive geographic view. Color describes type of change and larger dots are larger areas of maize growing activity. Band of increased risk concentrates in Iowa, Illinois, and Indiana.](./img/map.png "Interactive geographic view. Color describes type of change and larger dots are larger areas of maize growing activity. Band of increased risk concentrates in Iowa, Illinois, and Indiana."){#fig:geo }

## Trend-adjustment
@nielsen_historical_2023 suggests that historic trends would anticipate continued increases in maize outputs but our simulations predict climate change to wipe out the {{ counterfactualMean2050 }} yield increase that our neural network would otherwise expect within the counterfactual simulation without further warming. This flattening of yield increases may impact not just aggregate output but also how growers choose options such as trend adjustment  [@plastina_trend-adjusted_2014].

## Heat and drought stress
Our model shows depressed yields during combined warmer and drier conditions, combinations similar to 2012 and its historically poor maize production [@ers_weather_2013]. In this context, precipitation may serve as a protective factor: neighborhoods with drier July conditions see higher loss probability ($p < 0.05 / 2$) in both the 2030 and 2050 series via rank correlation [@spearman_proof_1904]. Our predictions thus reflect empirical studies that document the negative impacts of heat stress and water deficits on maize yields [@sinsawat_effect_2004; @marouf_effects_2013].

![Screenshot of an interactive tool showing precipitation and loss probability changes. The horizontal axis is change in precipitation and the vertical axis is the change in claims rate (probability of covered loss).](./img/scatter.png "Screenshot of an interactive tool showing precipitation and loss probability changes. The horizontal axis is change in precipitation and the vertical axis is the change in claims rate (probability of covered loss)."){#fig:chirps}

These outputs may also reveal geographically and temporally specific outlines of these protective factors, possibly useful for insurer and grower adaptation. Even so, as pictured in Figure @fig:chirps, we caution that analysis finds significant but weak rank correlations in both series, indicating that model expectations cannot be described by precipitation alone.

## Risk unit size
Prior work expects larger insured units to reduce risk [@knight_developing_2010] and we similarly observe a claims rate decrease as the acreage included in an insured unit grows. As this structure may change in the future, we run post-hoc experiments altering unit size to determine robustness of our findings:

 - Simulate outcomes as if units contain approximately single fields, decreasing aggregation.
 - Simulate outcomes with removal of smaller Optional Units [@zulauf_importance_2023], increasing aggregation. 

In both cases, a gap persists in claims rates between the counterfactual and SSP245, suggesting concerns remain relevant even as unit sizes evolve.

## Other models and programs
We also highlight additional future modeling opportunities beyond the scope of this study.

- We evaluate yield deltas and include historic yield as inputs into our neural network, allowing those data to "embed" adaptability measures [@hsiang_estimating_2017] such as soil properties and practices. However, those estimating absolute yield prediction may consider @rayburn_comparison_2022 as well as @woodard_efficiency_2017 to incorporate other variables like soil properties.
- Later research may also extend to genetic modification and climate-motivated practice changes which we assume to be latent in historic data.
- Though our interactive tools consider alternatives to SCYM and different spatial aggregations such as 5 character (4 x 5 km) geohashes, future work may consider modeling with actual field-level yield data and the actual risk unit structure. To that end, we observe that the actual unit yields / revenue and risk unit structure are not currently public.
- Due to the robustness of yield data, we examine Yield Protection but future agroeconomic study could extend this to the highly related Revenue Protection (RP) form of insurance. Indeed, the yield stresses seen in YP that we describe in this model may also impact RP.
- With additional data, future modeling could relax our normality assumption though, in addition to 79% of neighborhoods seeing approximately normal yield deltas, we observe that 88% of neighborhoods see approximate symmetry per @kim_statistical_2013 so that later work would likely not remove a systemic directional bias in results.

Note that, while we do anticipate changing historic yield averages in our simulations, we take a conservative approach and do not consider trend adjustment [@plastina_trend-adjusted_2014] and yield exclusion years [@schnitkey_yield_2015]. In raising $y_{expected}$, both would likely increase simulated loss rates.

## Interactive tools
In order to explore these simulated distributions geographically and under different scenarios, interactive open source web-based visualizations built alongside our experiments both aid in constructing our own conclusions and allow readers to consider possibilities and analysis beyond our own narrative.

| **Simulator**   | **Question**                                                                    | **Loop**                                                                                                                                                                   | **JG**                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Rates | What factors influence the price and subsidy of a policy? | Iteratively change variables to increase subsidy.  | Improving on previous hypotheses. |
| Hyper-Parameter | How do hyper-parameters impact regressor performance?                           | Iteratively change neural network hyper-parameters to see influence on validation set performance.                                                                         | Improving on previous hyper-parameter hypotheses. |
| Distributional  | How do overall simulation results change under different simulation parameters? | Iterative manipulation of parameters (geohash size, event threshold, year) to change loss probability and severity.                                                              | Deviating from the study’s main results.          |
| Neighborhood    | How do simulation results change across geography and climate conditions?       | Inner loop changing simulation parameters to see changes in neighborhood outcomes. Outer loop of observing changes across different views. | Identifying neighborhood clusters of concern.     |
| Claims          | How do different regulatory choices influence grower behavior?                  | Iteratively change production history to see which years result in claims under different regulatory schemes.                                                              | Redefining policy to improve yield stability.     |

Table: Overview of explorable explanations. {#tbl:apps}

In crafting the "explorable explanations" [@victor_explorable_2011] in Table @tbl:apps, we draw analogies to micro-apps  [@bridgwater_what_2015] or mini-games [@dellafave_designing_2014] in which the user encounters a series of small experiences that, each with distinct interaction and objectives, can only provide minimal instruction [@brown_100_2024]. As these visualizations cannot take advantage of design techniques like Hayashida-style tutorials [@pottinger_pyafscgaporg_2023], they rely on simple "loops" [@brazie_designing_2024] for immediate "juxtaposition gratification" (JG) [@jm8_secret_2024], showing fast progression after minimal input.

Following @unwin_why_2020, our custom tools first serve as internal "exploratory" graphics enabling the insights detailed in our results with Table @tbl:insights outlining specific observations we attribute to our use of these tools.

| **Simulator**   | **Observation**                                                                                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Distributional  | Dichotomy of larger changes to insurer-relevant tails contrasting smaller changes to mean yield.                                         |
| Claims          | Issues of using average for $y_{expected}$ [@fcic_common_2020].                                                                                                         |
| Neighborhood    | Eastward bias of impact. Model output relationships with broader climate factors, highlighting the possible systemic protective value of increased precipitation. |
| Hyper-parameter | Model resilience to removing individual inputs.                                                                                         |

Table: Observations we made from our own tools in the "exploratory" graphic context of @unwin_why_2020. {#tbl:insights}

Then, continuing to "presentation" [@unwin_why_2020], we next release these tools into a open source website at [https://ag-adaptation-study.pub](https://ag-adaptation-study.pub).

![Example interactive showing how a high stability unit could see a claim for a bad year under $l_{\sigma}$ but not $l_{\%}$.](./img/yield_sim.png "Example interactive showing how a high stability unit could see a claim for a bad year under $l_{\sigma}$ but not $l_{\%}$."){ width=90% #fig:stdev }

These public interactive visualizations allow for further exploration of our modeling such as different loss thresholds for other insurance products, finding relationships of outcomes to different climate variables, answering geographically specific questions beyond the scope of this study, and modification of machine learning parameters to understand performance. This may include use as workshop activity and we also report^[We collect information about the tool only and not generalizable knowledge about users or these patterns, falling under "quality assurance" activity. IRB questionnaire on file.] briefly on design changes made to our interactive tools in response to its participation in a 9 person "real-world" workshop session co-exploring these results:

 - Like @pottinger_combining_2023, facilitatation alternates between presentation and interaction but, in response to a previously "lecture-heavy" initial background, we add the rates simulator to let facilitators break up that section.
 - Facilitators suggest that single loop [@brazie_designing_2024] designs perform best within the limited time of the workshop and we now let facilitators hold the longer two loop neighborhood simulator till the end by default.
 - As expected by the JG design [@jm8_secret_2024], discussion contrasts different results sets and configurations of models but meta-parameter visualization relies heavily on memory so we now offer a "sweep" button for facilitators to show all results at once.

Later work may include controlled experimentation [@lewis_using_1982] or diary studies [@shneiderman_strategies_2006] for more generalizable design knowledge.


\bigskip

# Conclusion
Maize production not only suffers from climate warming's effects [@jagermeyr_climate_2021] but also adds to future climate change [@kumar_assessment_2021]. Inside this cyclic relationship, agriculture could crucially contribute to the necessary solution for the same global crisis that ails it [@schon_cover_2024]. In dialogue with prior work [@wang_warming_2021; @chemeris_insurance_2022], we highlight how existing policy currently prioritizes increased mean yield over climate adaptations benefiting stability. Furthermore, we demonstrate how current structures could fail to adjust to our predicted distributional changes and how inclusion of variance into those regulatory structures may positively influence adoption of mitigating practices such as regenerative systems. This responsive shift in coverage levels could reorient incentives: balancing overall yield with longitudinal stability and more comprehensively incorporating an understanding of risk. These resilience-promoting changes may benefit both grower and insurer without requiring practice-specific regulation. Recognizing that these structural levers require modification by policy makers, we therefore encourage scientists to further study, regulators / lawmakers to further consider, and producers to further inform these revisions. These essential multi-stakeholder efforts are crucial in preparing the US food system and its insurance program for a warmer future.

\bigskip

# Acknowledgements
Study funded by the Eric and Wendy Schmidt Center for Data Science and Environment at the University of California, Berkeley. We have no conflicts of interest to disclose. Using yield estimation data from @lobell_scalable_2015 and @deines_million_2021 with our thanks to David Lobell for permission. We also wish to thank Magali de Bruyn, Nick Gondek, Jiajie Kong, Kevin Koy, and Ciera Martinez for conversation regarding these results. Thanks to Color Brewer [@brewer_colorbrewer_2013] and Public Sans [@general_services_administration_public_2024].

\bigskip

# Works Cited
