---
bibliography: ./paper.bib
title: "Supplementary Materials for Climate-Driven Doubling of U.S. Maize Loss Probability: Interactive Simulation through Neural Network Monte Carlo"
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
date: 2024-10-31
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

**Overview**: These supplementary materials complement "Climate-Driven Doubling of U.S. Maize Loss Probability: Interactive Simulation through Neural Network Monte Carlo" to further describe the work including statistical tests employed, the simulation of insured units, and the interactive tools available at https://ag-adaptation-study.pub.

# Methods and data
These materials start with further explanation of the methods and data empoloyed.

## Input vector
For our presented results, we allow the model to use the count of growing condition estimations as a possible measure of uncertainty. However, we exclude the year being predicted which, in addition to seeing potential signs of overfit, may also assume more specificity in individual year conditions than potentially appropriate given the 2030 and 2050 series structure of CHC-CMIP6 [@williams_high_2024]. Even so, these options may be configured in our open source data pipeline for model retraining.

## Statistical tests
To determine significance of changes to loss probability at neighborhood-level, we use Mann Whitney U [@mann_test_1947] as variance is observed to differ between the two expected and counterfactual sets [@mcdonald_handbook_2014]. As our neural network attempts to predict the distribution of yield values, we note that the granularity of the response variable (SCYM yield) specifically may influence statistical power and we observe that SYCM [@lobell_scalable_2015] uses Daymet variables at 1 km resolution [@thornton_daymet_2014]. Therefore, due to potential correlation within those 1km cells, we assume 1km resolution for the purposes of statistical tests to avoid artificially increasing the number of "true" SCYM yield estimations per neighborhood. Finally, we recognize that we are engaging in one statistical test per neighborhood per series (2030, 2050). We control for this through Bonferroni-correction [@bonferroni_il_1935].

## Insured risk unit data
To further describe our treatment of insured risk units, consider that the USDA provides anonymized information about risk structure [@rma_statecountycrop_2024]. We provide a histogram of this distribution in Figure @fig:riskunit. Though these data lack precise geographic specificity, the USDA indicates the county in which these units are located. Even so, we notice year to year instability at the county level. This may reflect growers reconfiguring their risk structure to optimize rates as yield profiles change over time. Altogether, this may cause the geographic location of larger units to shift between years.

![Examination of risk unit size in years 2013, 2018, and 2023. First, this figure shows how risk unit size changed between each year examined (A) to highlight that the structures do evolve substantially between years. However, these results also indicate that the overall distribution of risk unit sizes is relatively stable (B) when considered system-wide. Some extreme outliers not shown to preseve detail.](./img_static/risk_unit_shape.png "Examination of risk unit size in years 2013, 2018, and 2023. First, this figure shows how risk unit size changed between each year examined (A) to highlight that the structures do evolve substantially between years. However, these results also indicate that the overall distribution of risk unit sizes is relatively stable (B) when considered system-wide. Some extreme outliers not shown to preseve detail."){ width=95% #fig:riskunit}

All this in mind, sampling the risk unit size at the county level likely represents over-confidence or overfitting to previous configurations. Even so, we observe that the system-wide risk unit size distribution remains relatively stable. This may suggest that, even as more local changes to risk unit structure may be more substantial between years, overall expectations for the size of risk units are less fluid. Therefore, we use that larger system-wide distribution to sample risk unit sizes within our Monte Carlo simulation instead of the county-level distributions. This also has the effect of propogating risk unit size uncertainty into results through the mechanics of Monte Carlo.

## Additional notes on included years and areas
To further document how we structure our consideration of timeseries variables, we emphasize that we sample for nine individual years in the 2030 CHC-CMIP6 series and nine individual years in 2050 CHC-CMIP6 series. Importantly, projections in these series are not necessarily intended as specific predictions in specific years. We do not provide a year by year timeseries for this reason. Instead, our analysis produces distributions of anticipated outcomes at the 2030 and 2050 timeframes. Note that our choice to create these two series follows a similar structure to CHC-CMIP6.

## Crop rotations
We treat practices as latent within our observed yield distributions. That in mind, a large share of growers will engage in at least simple crop rotations [@manski_diversified_2024] which is important for our simulations because it may change the locations in which maize is grown. We use SCYM to implicitly handle this complexity. These reported sample sizes impact the sampling behavior during Monte Carlo and, while this approach does not require explicit consideration of crop rotations, the set of geohashes present in results may vary from one year to the next in part due to this behavior. All that said, historic locations of growth and crop rotation behavior from the past are sampled in the future simulations.

## Normality assumption
In consideration of our normality assumption, we document that, in addition to 79% of neighborhoods exhibiting approximately normal yield deltas, 88% see approximate symmetry [@kim_statistical_2013]. Even so, future modeling could relax our normality assumption with additional data, potentially by avoiding the use of summary variables.

## Instance weight
We document that we build our model with instance weighting. Specifically, we use the number (not value) of SCYM pixels in a neighborhood to weight each neighborhood. In other words, the weight is higher in neighborhoods with more maize growing acreage.

## Limitations of sample size
The drop in error observed from validation to test performance may be explained by the increased training set size. Indeed, evaluating test performance without retraining with train and validation together leads to an elevated test set error as indicated in Table @tbl:retrain.

| **Set**             | **MAE for Mean Prediction** | **MAE for Std Prediction** |
| -------------------- | ----------------------- | ---------------------- |
| Validation           | {{validationMeanMae}}   | {{validationStdMae}}   |
| Test with retrain    | {{retrainMeanMae}}      | {{retrainStdMae}}      |
| Test without retrain | {{testMeanMae}}         | {{testStdMae}}         |

Table: Follow up experiment in which the test is evaluated without retraining. {#tbl:retrain}

This may indicate that the model is specifically data constrained by the number of years available for training. Our open source data pipeline can and will be used to rerun analysis as input datasets are updated to include additional years in the future.

# Interactive tools
Next, we further describe our interactive tools. In crafting these "explorable explanations" [@victor_explorable_2011] in Table @tbl:apps, we draw analogies to micro-apps  [@bridgwater_what_2015] or mini-games [@dellafave_designing_2014] in which the user encounters a series of small experiences that, each with distinct interaction and objectives, can only provide minimal instruction [@brown_100_2024]. As these very brief visualization experiences cannot take advantage of design techniques like Hayashida-style tutorials [@pottinger_pyafscgaporg_2023], they rely on simple "loops" [@brazie_designing_2024] for immediate "juxtaposition gratification" (JG) [@jm8_secret_2024], showing fast progression after minimal input.


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
| Distributional  | Dichotomy of larger changes to insurer-relevant tails contrasting smaller changes to mean yield.                                         |
| Claims          | Issues of using average for $y_{expected}$ [@fcic_common_2020].                                                                                                         |
| Neighborhood    | Geographic bias of impact. Model output relationships with broader climate factors, highlighting the possible systemic protective value of increased precipitation. |
| Hyper-parameter | Model resilience to removing individual inputs.                                                                                         |

Table: Observations we made from our own tools in the "exploratory" graphic context of @unwin_why_2020. {#tbl:insights}

Altogether, these tools serve to support our exploration of our modeling such as different loss thresholds for other insurance products, finding relationships of outcomes to different climate variables, answering geographically specific questions beyond the scope of this study, and modification of machine learning parameters to understand performance.

## Workshops
In addition to supporting our finding of our own conclusions, we release this software publicly at https://ag-adaptation-study.pub/. Possible use of these tools include workshop activity and we also report^[We collect information about the tool only and not generalizable knowledge about users or these patterns, falling under "quality assurance" activity. IRB questionnaire on file.] briefly on design changes made to our interactive tools for that purpose. These were implemented in response to our work's participation in a 9 person "real-world" private workshop session encompassing scientists and engineers which was intended to improve these tools specifically through active co-exploration limited to these study results. Changes include:

 - Facilitators elected to alternate between presentation and interaction similar to @pottinger_combining_2023 but we added the rates simulator to further improve presentation of the rate setting process.
 - Facilitators suggest that single loop [@brazie_designing_2024] designs perform best within the limited time of the workshop and we now let facilitators hold the longer two loop neighborhood simulator till the end by default.
 - As expected by the JG design [@jm8_secret_2024], discussion contrasts different results sets and configurations of models but meta-parameter visualization relies heavily on memory so we now offer a "sweep" button for facilitators to show all results at once.

This was *not* a public workshop or a formalized academic conference presentation. That said, while we are thankful for this pre-publication opportunity which only focused on quality improvements specific to the service offered by https://ag-adaptation-study.pub/, later work may further more broadly explore this design space through controlled experimentation [@lewis_using_1982] or diary studies [@shneiderman_strategies_2006].

# Works cited
