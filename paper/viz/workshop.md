# Workshop outline
Slides which overview Climate-driven doubling of U.S. maize loss probability: Interactive simulation with neural network Monte Carlo. Published in JDSSV as Pottinger et al 2025 with DOI: [10.52933/jdssv.v5i3.134](https://doi.org/10.52933/jdssv.v5i3.134).

## Background
 - The **costs of crop insurance in the U.S. have already increased by 500%** since the early 2000s with annual indemnities reaching $19B in 2022 ([Schechinger 2023](https://www.ewg.org/research/crop-insurance-costs-soar-over-time-reaching-record-high-2022)).
 - Retrospective analysis attributes 19% of “national-level crop insurance losses” between 1991 and 2017 to climate warming, an estimate rising to **47% during the drought-stricken 2012 growing season** ([Diffenbaugh et al. 2021](https://iopscience.iop.org/article/10.1088/1748-9326/ac1223))

## Study
 - **Research question**: We want to explore what the future of the Federal Crop Insurance Program might look like under climate change to evaluate how the structures of FCIP might interact with possible changes to yield patterns.
 - **Data**: We used SCYM ([Lobell et al 2015](https://www.sciencedirect.com/science/article/abs/pii/S0034425715001637)) for historic yield estimation and CHC-CMIP6 ([Funk et al 2024](https://www.nature.com/articles/s41597-024-03074-w)) under SSP245 at near term (approximately 2030s) and medium term (approximately 2050s).
 - **Tools**: We built interactive visualizations as "explorable explanations" to help support dialogue and user agency to experiment with these ideas.

## Method
 - Neural network uses past yield outcomes under past climate conditions as well as how growers may have responded to those conditions to **predict future yield outcomes as distributions of yield deltas** not point values, allowing us to propagate uncertainty and make probabilistic claims.
 - Monte Carlo **simulates outcomes both in terms of yield but also in terms of probability of claim** under the yield protection form of Multi-Peril Crop Insurance though similar stressors may be present for revenue protection.
 - We can run these at **risk unit-level resolution** sampled from historic values to build geospatial result datasets that incorporate a sense of historic variability under different climate conditions.

## Summary
 - **Simulations see a riskier future**: Yield losses which may trigger indemnity claims will not only be more severe around 2050 but also be twice as frequent. 
 - **We anticipate FCIP pressures**: the traditional "75% of average APH yield" will likely become riskier to insure than in the past.

## Details
 - The shape of the "yield deltas" distribution is changing to be **less symmetric** such that bad years are becoming more common.
 - The **probability of a loss is increasing** even if that loss is not realized each year and there are not consistent decreases in yield.
 - These seem to correspond somewhat to **drought / heat stress** but also different risk units see different variability.
 - **12.7% of neighborhoods** may see both **increased average yields AND increased claims rates** simultaneously, highlighting how yield volatility can elevate risk even when overall productivity improves

## Policy
 - Some risk units have more variability than others under growing condition stress. This may have to do with intrinsic properties but it **may also have to do with differences in practices which provide resilience**.
 - The current structure of MPCI is based on average with yield exclusions which **does not necessarily financially incentivize stability** as it encourages growers to drive up yields even at the expense of variability.
 - Switching to a standard deviation based coverage level can achieve claims rates similar to the current average-based coverage level today but **keeps those claims rates from dramatically increasing in the future** as growing conditions evolve.
 - This is **practice-agnostic**, allowing growers flexibility to achieve stability that can complement practice-specific efforts but possible offer mechanisms to mitigate issues with practice-specific approaches explored by prior work ([Connor et al 2021](https://onlinelibrary.wiley.com/doi/10.1002/aepp.13206)).

## Future
 - **Revenue Protection** is quite difficult to model but would be useful for painting a broader picture.
 - We explored practice-agnostic tweaks to FCIP formulations but we could further explore these results in **practice-specific** contexts.
 - Undertaking work to better understand policy opportunities including through **508h**.
