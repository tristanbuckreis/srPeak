# srPeak

A python implementation of an automated algorithm to identify resonant peak features in data-derived site response.

# Automated Peak Identification Algorithm

In general, “peaks” are defined as features possessing the following key attributes:
1. Relatively localized (i.e., the width should not span too large of a period/frequency range),
2. Have sufficiently large mean amplitude relative to adjacent periods/frequencies, and
3. Have sufficient confidence that the feature is meaningful (i.e., uncertainty in amplitudes or frequencies should not be too large).
   
<p align="justify">This peak detection algorithm is adapted from that recommended by Wang et al. (2021) for HVSR to application to site response, which considers the three attributes listed above. For the purposes of this algorithm, site response refers to the unmodeled site response after <i>V<sub>S30</sub></i>-scaling effects have been accounted for.</p>

<p align="justify">The principal challenge in assessing the first two attributes is defining the amplitudes and locations (periods/frequencies) of site response peaks; what is required is a parameterization of site response amplitudes adjacent to peak features and within peak features. This algorithm implements a regression tree (Breiman et al. 1984), which is a predictive modeling approach in machine learning, to effectively smooth and simplify the empirical site response as a piecewise function of non-overlapping linear segments (i.e., steps). A complexity parameter (<i>c</sub>p</sub></i>) is used to specify the penalty in tree regression. Large values of <i>c</sub>p</sub></i> produce relatively crude fits with wide steps, whereas smaller values produce better fits with narrow steps. Figure 1 illustrates the influence of the <i>c</sub>p</sub></i> parameter on the tree regression for an individual site. If <i>c</sub>p</sub></i> is too large, the fit is poor, whereas if <i>c</sub>p</sub></i> is too small, there is the potential that the tree regression captures too many small peaks, which is not amendable to defining a stable peak-adjacent plateau.Selection of the preferred value of <i>c</sub>p</sub></i> is subjective, however <i>c</sub>p</sub></i> = 0.0003 is found to provide a reasonable balance between accuracy and reliability.</p>


![Figure 7](https://github.com/tristanbuckreis/srPeak/assets/71461454/d8dd17de-dc54-4032-b59c-143eb5eb2c66)


# References:

Buckreis, T.E. (2022) Customization of path and site response components of global ground motion models for application in Sacramento-San Joaquin Delta region of California. Doctoral dissertation, University of California, Los Angeles, Los Angeles, CA.
