---
bibliography: ./paper.bib
title: "Supplementary Materials for Climate-Driven Doubling of U.S. Maize Loss Probability: Interactive Simulation through Neural Network Monte Carlo"
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

**Overview**: These supplementary materials complement "Climate-Driven Doubling of U.S. Maize Loss Probability: Interactive Simulation through Neural Network Monte Carlo" to further describe the work including statistical tests employed, simulation specfics, and the interactive tools available at https://ag-adaptation-study.pub.

# Methods and data
These materials start with further explanation of the methods and data employed.

## Statistical tests
To determine significance of changes to loss probability at neighborhood-level, we use Mann Whitney U [@mann_test_1947] as variance is observed to differ between the two expected and counterfactual sets [@mcdonald_handbook_2014]. Given that our neural network attempts to predict the distribution of yield deltas, we note that the granularity of the response variable specifically may influence statistical power and we observe that SYCM [@lobell_scalable_2015] uses Daymet variables at 1 km resolution [@thornton_daymet_2014]. Therefore, due to potential correlation within those 1km cells, we assume 1km resolution for the purposes of statistical tests to avoid artificially increasing the number of "true" SCYM yield estimations per neighborhood. Finally, we recognize that we are engaging in one statistical test per neighborhood per series (2030, 2050). We control for this through Bonferroni-correction [@bonferroni_il_1935].

## Insured risk unit data
To further describe our treatment of insured risk units, consider that the USDA provides anonymized information about risk structure [@rma_statecountycrop_2024]. We provide a histogram of this distribution in Figure @fig:riskunit.

![Examination of risk unit size in years 2013, 2018, and 2023. First, this figure shows how risk unit size changed between each year examined (A) to highlight that the structures do evolve substantially between years. However, these results also indicate that the overall distribution of risk unit sizes is relatively stable (B) when considered system-wide. Some extreme outliers not shown to preseve detail.](./img_static/risk_unit_shape.png "Examination of risk unit size in years 2013, 2018, and 2023. First, this figure shows how risk unit size changed between each year examined (A) to highlight that the structures do evolve substantially between years. However, these results also indicate that the overall distribution of risk unit sizes is relatively stable (B) when considered system-wide. Some extreme outliers not shown to preseve detail."){ width=95% #fig:riskunit}

Though these data lack precise geographic specificity, the USDA indicates the county in which these units are located. Even so, we notice year to year instability at the county level. This may reflect growers reconfiguring their risk structure to optimize rates as yield profiles change over time. Altogether, this may cause the geographic location of larger units to shift between years.

All this in mind, sampling the risk unit size at the county level likely represents over-confidence or overfitting to previous configurations. Instead, we observe that the system-wide risk unit size distribution remains relatively stable. This may suggest that, even as more local changes to risk unit structure may be more substantial between years, overall expectations for the size of risk units are less fluid. Therefore, we use that larger system-wide distribution to sample risk unit sizes within our Monte Carlo simulation instead of the county-level distributions. This also has the effect of propogating risk unit size uncertainty into results through the mechanics of Monte Carlo.

## Yield distributions
Our treatment of yield data considers two practical constraints:

 - Due to the size of the input dataset and engineering limitations, we cannot take all SCYM data per neighborhood into Monte Carlo.
 - We must avoid dramatic expansions to the output vector size as this could cause the input dataset requirements to exceed feasibility [@alwosheel_dataset_2018].

These concerns in mind, we sample annual SCYM yields to generate yield delta distributions which allows us to wait until the later parts of our pipeline to make shape assumptions. This ensures "just in time" that our neural network can predict a smaller number of distribution shape parameters while maintaing underlying shape information for as long as possible.

### Yield delta distributions
In generating the historic yield delta distributions ahead of training neural networks, we sample 1000 yield values per neighborhood per year to represent a growing season^[The resulting historic yield delta distributions are further sampled based on simulated risk unit size, either from historic actuals or neural network predicted distributions. Note that we also sample to represent historic averages as aggregation to $y_{expected}$ can be subject to "small samples" stochastic effects per risk unit.]. Altogether, this design avoids needing to make distributional assumptions about yield ahead of neural network operation while maintaining the original distributional shape.

### Pipline flexibility
Our neural network requires a distributional shape assumption to maintain a smaller output vector size. We decide the shape to predict based on observed skew and kurtosis of yield deltas. To that end, our open source pipeline can be run with beta or normal distribution assumptions. The former has precedent in the literature [@nelson_influence_1990].

### Practical yield delta shape
Despite pipeline flexibility, we observe that nearly all^[Around 97% - 99% of neighborhoods and maize growing acreage depending on yield sampling are approximately normal per @kim_statistical_2013.] yield delta distributions exhibit approximate normality in practice [@kim_statistical_2013]. Separately, using beta distributions in our neural networks results in similar median absolute errors but elevated mean absolute errors as shown in Table @tbl:betadist which highlights prediction of distribution location specifically. 

| **Shape**            | **Mean Absolute Error** | **Median Absolute Error** |
| -------------------- | ----------------------- | ------------------------- |
| Normal               | {{retrainMeanMae}}      | {{retrainMeanMdae}}       |
| Beta                 | 16.9%                   | 7.1%                      |

Table: Test set performance after retraining for predicting distribution location (mean or center) for both a normal distribution and beta distribution assumption. {#tbl:betadist}

Further investigation finds that that a minority population of neighborhoods causes this swing where small changes in beta distribution parameters can infrequently cause large error. Therefore, as prediction of that population shows stronger performance under a normality assumption for yield deltas, we prefer this approach in our main text.

## Neural network configuration
We offer additional information about the specific neural network configuration chosen.

### Input vector
We allow the model to use the count of growing condition estimations as a possible measure of uncertainty. This generally leads to better performance. We also allow inclusion of the year. However, as can be executed in our open source pipeline, we find that including absolute year generally increases overfitting. Therefore, we use a relative measure (years since the start of the series within the simulations).

### Included years and areas
To further document how we structure our consideration of timeseries variables, we emphasize that we sample for 17 individual years in the 2030 CHC-CMIP6 series and 17 individual years in 2050 CHC-CMIP6 series. Importantly, projections in these series are not necessarily intended as specific predictions in specific years. We do not provide a year by year timeseries for this reason. Instead, our analysis produces distributions of anticipated outcomes at the 2030 and 2050 timeframes. Note that our choice to create these two series follows a similar structure to CHC-CMIP6. Finally, note that many growers engage in even simple crop rotations so the effective average crop yield for a field used to define yield expectations may span 10 crop years but possibly more than 10 consecutive calendar years.

### Instance weight
We document that we build our model with instance weighting. Specifically, we use the number (not value) of SCYM pixels in a neighborhood to weight each neighborhood. In other words, the weight is higher in neighborhoods with more maize growing acreage.

### Error and residuals
Table @tbl:retrain provides mean absolute error for the selected model from the sweep. A drop in error observed from validation to test performance may be explained by the increased training set size. This may indicate that the model is specifically data constrained by the number of years available for training. Our open source data pipeline can and will be used to rerun analysis as input datasets are updated to include additional years in the future.

| **Set**             | **MAE for Mean Prediction** | **MAE for Std Prediction** |
| -------------------- | ----------------------- | ---------------------- |
| Train                | {{trainMeanMae}}        | {{trainStdMae}}   |
| Validation           | {{validationMeanMae}}   | {{validationStdMae}}   |
| Test with retrain    | {{retrainMeanMae}}      | {{retrainStdMae}}      |
| Test without retrain | {{testMeanMae}}         | {{testStdMae}}         |

Table: Residuals for the main training task with and without retraining. {#tbl:retrain}

The test set residuals are sampled during Monte Carlo to propogate uncertainty. That said, we observe that a relatively small sub-population of large percentage changes may skew results, causing the mean and median error to diverge as shown in post-hoc tasks in Table @tbl:posthocresults.

| **Task**              | **Test Mean Pred MAE** | **Test Std Pred MAE** | **Test Mean Pred MdAE** | **Test Std Pred MdAE** |
| --------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| Random   | {{randomMeanMae}}      | {{randomStdMae}}      | {{randomMeanMdae}}          | {{randomStdMdae}}          |
| Temporal | {{temporalMeanMae}}    | {{temporalStdMae}}    | {{temporalMeanMdae}}        | {{temporalStdMdae}}        |
| Spatial  | {{spatialMeanMae}}     | {{spatialStdMae}}     | {{spatialMeanMdae}}         | {{spatialStdMdae}}         |
| Climatic | {{climateMeanMae}}     | {{climateStdMae}}     | {{climateMeanMdae}}         | {{climateStdMdae}}         |

Table: Results of tests after model selection. {#tbl:posthocresults}

Even so, the overall error remains acceptable. In general, increased model size is showing diminishing returns and we do not currently consider additional layers (4 vs 5 neural network layers changes mean prediction MAE by less than one point).

## Grower behaviors
We further document some grower behaviors which may be difficult to capture within our curent modeling structure.

### Historic yield averages
Our simulations expect yield expectations to change over time. In practice, we sample ten years of historic yields per neighborhood per year per trial and we offset the yield deltas produced by the neural network accordingly. This allows for some accounting of uncertainty in yield baselines. In practice, this means that predictions for 2030 claims rate samples the 2010 (historic) series and 2050 samples the 2030 series. To prevent discontiniuity in the data, the 2010 deltas are retroactively predicted. Model error residuals are sampled in each case.

### Yield history adjustments
We once more highlight that, in practice, the values used to set yield expectations depend on trend adjustment [@plastina_trend-adjusted_2014] and yield exclusions [@schnitkey_yield_2015] which, due to insufficient data, are left for future work. Again, by increasing $y_{expected}$, these may lead to an artifical supression of our predicted claims rates.

### Crop rotations
A large share of growers will engage in at least simple crop rotations [@manski_diversified_2024] which is important for our simulations because it may change the locations in which maize is grown. We use SCYM to implicitly handle this complexity. These reported sample sizes impact the sampling behavior during Monte Carlo and, while this approach does not require explicit consideration of crop rotations, the set of geohashes present in results may vary from one year to the next in part due to this behavior.

All this in mind, historic locations of growth and crop rotation behavior from the past are sampled in the future simulations. In addition to this spatial complexity, we highlight that crop rotations mean that the last 10 years of yield data for a crop may not correspond to the last 10 calendar years but, due to the "year series" approach in this model, this probably has limited effect on our multi-year claims rates estimations.

# Detailed simulation results
For reference, we provide furhter detailed simulated results in Table @tbl:simresults.

| **Scenario**   | **Series** | **Unit mean yield change** | **Unit loss probability**         | **Avg covered loss severity**  |
| ---------------------------- | -------- | -------------------------- | --------------------------------- | ------------------------------ |
| Historic | 2010     | {% if referenceMean2010|float > 0 %}+{% endif %}{{referenceMean2010}} | {{referenceProbability2010}} | {{referenceSeverity2010}} |
| Counterfactual | 2030     | {% if counterfactualMean2030|float > 0 %}+{% endif %}{{counterfactualMean2030}} | {{counterfactualProbability2030}} | {{counterfactualSeverity2030}} |
| SSP245         | 2030     | {% if experimentalMean2030|float > 0 %}+{% endif %}{{experimentalMean2030}}   | {{experimentalProbability2030}}   | {{experimentalSeverity2030}}   |
| Counterfactual | 2050     | {% if counterfactualMean2050|float > 0 %}+{% endif %}{{counterfactualMean2050}} | {{counterfactualProbability2050}} | {{counterfactualSeverity2050}} |
| SSP245         | 2050     | {% if experimentalMean2050|float > 0 %}+{% endif %}{{experimentalMean2050}}   | {{experimentalProbability2050}}   | {{experimentalSeverity2050}}   |
|                |          | $y_{\Delta \mu}$           | $p_{l-\mu}$                         | $s_{\mu}$                      |

Table: Details of Monte Carlo simulation results. Counterfactual is a future without continued warming in contrast to SSP245. {#tbl:simresults}

These results are also made available in Zenodo [@pottinger_data_2024].

## Series labels
Note that the "2010 series" label is used internally in our model for consistency with 2030 and 2050 from CHC-CMIP6 though that "2010" language does not explicitly appear in their data model.

## Confidence
We re-execute simulations 100 times to understand variability for system-wide metrics in Table @tbl:simresults. The range of all standard deviations of each metric's distribution is under 0.1% and the range under 1%. These tight intervals likely reflect the high degree of aggregation represented in our system-wide metrics. However, lacking confidence measures from SCYM and CHC-CMIP6, this post-hoc experiment cannot account for input data uncertainty which is likely more substantial.

# Interactive tools
Finally, we further describe our interactive tools. In crafting these "explorable explanations" [@victor_explorable_2011] listed in Table @tbl:apps, we draw analogies to micro-apps  [@bridgwater_what_2015] or mini-games [@dellafave_designing_2014] in which the user encounters a series of small experiences that, each with distinct interaction and objectives, can only provide minimal instruction [@brown_100_2024]. As these very brief visualization experiences cannot take advantage of design techniques like Hayashida-style tutorials [@pottinger_pyafscgaporg_2023], they rely on simple "loops" [@brazie_designing_2024] for immediate "juxtaposition gratification" (JG) [@jm8_secret_2024], showing fast progression after minimal input.


| **Simulator**   | **Question**                                                                    | **Loop**                                                                                                                                                                   | **JG**                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Rates | What factors influence the price and subsidy of a policy? | Iteratively change variables to increase subsidy.  | Improving on previous hypotheses. |
| Hyper-Parameter | How do hyper-parameters impact regressor performance?                           | Iteratively change neural network hyper-parameters to see influence on validation set performance.                                                                         | Improving on previous hyper-parameter hypotheses. |
| Distributional  | How do overall simulation results change under different simulation parameters? | Iterative manipulation of parameters (geohash size, event threshold, year) to change loss probability and severity.                                                              | Deviating from the study’s main results.          |
| Neighborhood    | How do simulation results change across geography and climate conditions?       | Inner loop changing simulation parameters to see changes in neighborhood outcomes. Outer loop of observing changes across different views. | Identifying neighborhood clusters of concern.     |
| Claims          | How do different regulatory choices influence grower behavior?                  | Iteratively change production history to see which years result in claims under different regulatory schemes.                                                              | Redefining policy to improve yield stability.     |

Table: Overview of explorable explanations. {#tbl:apps}

Following @unwin_why_2020, our custom tools first serve as internal exploratory graphics enabling the insights detailed in our results before acting as a medium for sharing our work with others.

## Internal use
First, these tools were built during our own internal exploration of data with Table @tbl:insights outlining specific observations we attribute to our use of these tools.

| **Simulator**   | **Observation**                                                                                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Distributional  | Dichotomy of changes to yield and changes to loss risk.                                         |
| Claims          | Issues of using average for $y_{expected}$ [@fcic_common_2020].                                                                                                         |
| Neighborhood    | Geographic bias of impact. Model output relationships with broader climate factors, highlighting the possible systemic protective value of increased precipitation. |
| Hyper-parameter | Model resilience to removing individual inputs.                                                                                         |

Table: Observations we made from our own tools in the "exploratory" graphic context of @unwin_why_2020. {#tbl:insights}

Altogether, these tools serve to support our exploration of our modeling such as different loss thresholds for other insurance products, finding relationships of outcomes to different climate variables, answering geographically specific questions beyond the scope of this study, and modification of machine learning parameters to understand performance.

## Workshops
In addition to supporting our finding of our own conclusions, we release this software publicly at https://ag-adaptation-study.pub/. Possible use of these tools include workshop activity and we also reflect^[These were implemented in response to our work's participation in a 9 person "real-world" workshop session encompassing scientists and engineers which was intended to improve these tools specifically through active co-exploration limited to these study results. We collect information about the tool only and not generalizable knowledge about users or these patterns, falling under "quality assurance" activity. IRB questionnaire on file.] briefly on design changes made to our interactive tools for that purpose:

 - Facilitators elected to alternate between presentation and interaction similar to @pottinger_combining_2023. However, we added the rates simulator to further improve presentation of the rate setting process due to the complexities of crop insurance, dyanmics previously explained in static diagrams.
 - Facilitators suggest that our single loop [@brazie_designing_2024] designs perform better than our our lone two-loop design within the limited time of the workshop. Therefore, we now let facilitators hold the longer two loop neighborhood simulator till the end by default.
 - While by the JG design [@jm8_secret_2024] expects discussion to contrast different results sets and configurations of models, the meta-parameter visualization specifically relies heavily on memory so we now offer a "sweep" button for facilitators to show all results at once.

This was *not* a public workshop or a formalized academic conference presentation. That said, while we are thankful for this pre-publication opportunity which only focused on quality improvements specific to the service offered by https://ag-adaptation-study.pub/, later work may more broadly explore this design space through controlled experimentation [@lewis_using_1982] or diary studies [@shneiderman_strategies_2006].

# Works cited
