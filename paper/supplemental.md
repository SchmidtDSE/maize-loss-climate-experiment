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

First, we specifically use the Mann Whitney U [@mann_test_1947] as variance is observed to differ between the two expected and counterfactual sets [@mcdonald_handbook_2014]. Furthermore, as the neural network attempts to predict the distribution of yield values, we note that the granularity of the response variable (SCYM yield) specifically may influence statistical power. Though prior validation sutides offer confidence [@deines_million_2021], we observe that SYCM [@lobell_scalable_2015] uses Daymet variables at 1 km resolution [@thornton_daymet_2014]. Therefore, we assume 1km resolution for the purposes of statistical tests as autocorrelation in our response variable could artificially increase the number of "true" SCYM yield estimations per neighborhood.

The USDA provides anonymized information about insured units [@rma_statecountycrop_2024] though this information lacks geographic specificity and note that some units may span more than one county. We provide a histogram of this distribution in Figure @fig:insuredunit.

Lacking more geographically detailed information, this distribution is used across the entire U.S. Corn Belt. Even so, inspection of records with county information reveals that this distribution is relatively similar across the corn belt. In comparing the county with the largest insured unit to the one with the smallest, we fail to find a statistically significant difference ($p \geq 0.05$).

