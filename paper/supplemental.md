---
bibliography: ./paper.bib
title: "Supplementary Materials for Climate-Driven Doubling of U.S. Maize Loss Probability: Simulation through Neural Network Monte Carlo"
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
date: 2024-10-23
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

# Overview
These supplementary materials complement "Climate-Driven Doubling of U.S. Maize Loss Probability: Simulation through Neural Network Monte Carlo" to further describe the statistical tests employed, the simulation of insured units, and further details on the interactive tools deployed at https://ag-adaptation-study.pub.

# Statistical tests
We specifically use Mann Whitney U [@mann_test_1947] as variance is observed to differ between the two expected and counterfactual sets [@mcdonald_handbook_2014]. Furthermore, as the neural network attempts to predict the distribution of yield values, we note that the granularity of the response variable (SCYM yield) specifically may influence statistical power. Though prior validation sutides offer confidence [@deines_million_2021], we observe that SYCM [@lobell_scalable_2015] uses Daymet variables at 1 km resolution [@thornton_daymet_2014]. Therefore, we assume 1km resolution for the purposes of statistical tests as autocorrelation in our response variable could artificially increase the number of "true" SCYM yield estimations per neighborhood.

# Risk unit size
The USDA provides anonymized information about insured units [@rma_statecountycrop_2024]. Though this information lacks geographic specificity, the USDA indicates the county in which these units are located. We provide a histogram of this distribution in Figure @fig:riskunit.

![Examination of risk unit size in years 2013, 2018, and 2023. Visualizes the average size of the risk unit within a county. First, this figure shows how risk unit size changed between each year examined (A) to highlight that the structures do evolve substantially between years. However, these results also indicate that the overall distribution of risk unit sizes is relatively stable (B) when considered system-wide. Some extreme outliers not shown to preseve detail.](./img_static/risk_unit_shape.png "Examination of risk unit size in years 2013, 2018, and 2023. Visualizes the average size of the risk unit within a county. First, this figure shows how risk unit size changed between each year examined (A) to highlight that the structures do evolve substantially between years. However, these results also indicate that the overall distribution of risk unit sizes is relatively stable (B) when considered system-wide. Some extreme outliers not shown to preseve detail."){ width=95% #fig:riskunit}

Analysis highlights year to year instability at the county level which may reflect growers reconfiguring their risk structure to opitimize rates as yield profiles change over time, causing the geographic location of larger units to shift over time. 

All this in mind, sampling the risk unit size at the county level likely represents over-confidence or overfitting to previous configurations. Even so, we observe that the system-wide risk unit size distribution remains relatively stable. This may suggest that, even as more local changes to risk unit structure may be more substantial between years, overall expectations for the size of risk units are less fluid. All this in mind, we use that larger system-wide distribution to sample risk unit sizes within our Monte Carlo simulation instead of the county-level distributions. This also has the effect of propogating risk unit size uncertainty into results through the mechanics of the Monte Carlo.

# Tool design


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

![Example interactive showing how a high stability unit could see a claim for a bad year under $l_{\sigma}$ but not $l_{\%}$.](./img/yield_sim.png "Example interactive showing how a high stability unit could see a claim for a bad year under $l_{\sigma}$ but not $l_{\%}$."){ width=90% #fig:stdev}

These public interactive visualizations like Figure @fig:stdev allow for further exploration of our modeling such as different loss thresholds for other insurance products, finding relationships of outcomes to different climate variables, answering geographically specific questions beyond the scope of this study, and modification of machine learning parameters to understand performance. This may include use as workshop activity and we also report^[We collect information about the tool only and not generalizable knowledge about users or these patterns, falling under "quality assurance" activity. IRB questionnaire on file.] briefly on design changes made to our interactive tools in response to its participation in a 9 person "real-world" workshop session co-exploring these results:

 - Facilitators elected to alternate between presentation and interaction similar to @pottinger_combining_2023 but we added the rates simulator to further improve presentation of the rate setting process.
 - Facilitators suggest that single loop [@brazie_designing_2024] designs perform best within the limited time of the workshop and we now let facilitators hold the longer two loop neighborhood simulator till the end by default.
 - As expected by the JG design [@jm8_secret_2024], discussion contrasts different results sets and configurations of models but meta-parameter visualization relies heavily on memory so we now offer a "sweep" button for facilitators to show all results at once.

Later work may further explore this design space through controlled experimentation [@lewis_using_1982] or diary studies [@shneiderman_strategies_2006].

# Works cited
