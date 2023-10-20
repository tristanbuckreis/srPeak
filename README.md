# srPeak

<p align="justify">A python implementation of an automated algorithm to identify resonant peak features in data-derived site response. Further details are provided in Buckreis (2022).</p>

In general, “peaks” are defined as features possessing the following key attributes:
1. Relatively localized (i.e., the width should not span too large of a period/frequency range),
2. Have sufficiently large mean amplitude relative to adjacent periods/frequencies, and
3. Have sufficient confidence that the feature is meaningful (i.e., uncertainty in amplitudes or frequencies should not be too large).

# Automated Peak Identification Algorithm
   
<p align="justify">This peak detection algorithm is adapted from that recommended by Wang et al. (2023) for HVSR to application to site response, which considers the three attributes listed above. For the purposes of this algorithm, site response refers to the unmodeled site response after <i>V<sub>S30</sub></i>-scaling effects have been accounted for (denoted 𝜂<sub>𝑆,𝑗</sub><sup>𝑣</sup>).</p>

<p align="justify">The principal challenge in assessing the first two attributes is defining the amplitudes and locations (periods/frequencies) of site response peaks; what is required is a parameterization of site response amplitudes adjacent to peak features and within peak features. This algorithm implements a regression tree (Breiman et al. 1984), which is a predictive modeling approach in machine learning, to effectively smooth and simplify the empirical site response as a piecewise function of non-overlapping linear segments (i.e., steps). A complexity parameter (<i>c<sub>p</sub></i>) is used to specify the penalty in tree regression. Large values of <i>c<sub>p</sub></i> produce relatively crude fits with wide steps, whereas smaller values produce better fits with narrow steps. Figure 1 illustrates the influence of the <i>c<sub>p</sub></i> parameter on the tree regression for an individual site. If <i>c<sub>p</sub></i> is too large, the fit is poor, whereas if <i>c<sub>p</sub></i> is too small, there is the potential that the tree regression captures too many small peaks, which is not amendable to defining a stable peak-adjacent plateau. Selection of the preferred value of <i>c<sub>p</sub></i> is subjective, however <i>c<sub>p</sub></i> = 0.0003 is found to provide a reasonable balance between accuracy and reliability.</p>

<p align="justify">The peak detection algorithm operates on the stepped results of tree regression and is shown as a flow chart in Figure 2. The algorithm operates as follows:</p>
1. <p align="justify">Identify potential peak steps: Step <i>m</i> is a peak if its amplitude (<i>amp<sub>p</sub></i>) is larger than those of steps <i>m</i> – 1 and <i>m</i> + 1.</p>
2. <p align="justify">For each potential peak step in Task (1), identify the left-peak-adjacent step-plateau (<i>step<sub>l</sub></i>). This identification has the following sub-steps:</p>
   <p align="justify">i. Let <i>l</i> denote the number of steps to the left of the peak step (i.e., <i>l</i> = 0 is the peak step <i>m</i>; <i>l</i> = 1 is the step immediately to the left of step <i>m</i>; <i>l</i> = 2 is the second step to the left of step <i>m</i>; and so on).</p>
   <p align="justify">ii. Starting at <i>l</i> = 1, obtain the left step amplitude (<i>amp<sub>m-l</sub></i>) and calculate the width (natural log difference of maximum and minimum period of step <i>m</i> – <i>l</i>; <i>wid<sub>m-l</sub></i>).</p>
   <p align="justify">iii. If <i>amp<sub>m-l</sub></i> > <i>amp<sub>m-l+1</sub></i>, then step <i>m</i> – <i>l</i> + 1 is the left-peak-adjacent step plateau, and proceed to Task (3). Otherwise, continue to Sub-task (iv).</p>
   <p align="justify">iv. If <i>wid<sub>m-l</sub></i> > <i>step<sub>thres</sub></i>, the step is sufficiently wide to be considered a plateau, and step <i>m</i> – <i>l</i> is the left-peak-adjacent step plateau, and proceed to Task (3). Otherwise, continue to Sub-task (v).</p>
   <p align="justify">v. If <i>m</i> – <i>l</i> = 1 (i.e., the first step), then go back through all left-steps and select the step with the largest width, and proceed to Task (3).</p>
3. <p align="justify">For each potential peak step in Task (1), identify the right-peak-adjacent step-plateau (<i>step<sub>r</sub></i>):</p>
   <p align="justify">i. Let <i>r</i> denote the number of steps to the right of the peak step (i.e., <i>r</i> = 0 is the peak step <i>m</i>; <i>r</i> = 1 is the step immediately to the right of step <i>m</i>; <i>r</i> = 2 is the second step to the right of step <i>m</i>; and so on).</p>
   <p align="justify">ii. Starting at <i>r</i> = 1, obtain the right step amplitude (<i>amp<sub>m+r</sub></i>) and calculate the width (log difference of maximum and minimum period of step <i>m</i> + <i>r</i>; <i>wid<sub>m+r</sub></i>).</p>
   <p align="justify">iii. If <i>amp<sub>m+r</sub></i> > <i>amp<sub>m+r-1</sub></i>, then step <i>m</i> + <i>r</i> + 1 is the right-peak-adjacent step plateau, and proceed to Task (4). Otherwise, continue to Sub-task (iv).</p>
   <p align="justify">iv. If <i>wid<sub>m+r</sub></i> > <i>step<sub>thres</sub></i>, the step is sufficiently wide enough to be considered a plateau, and step <i>m</i> + <i>r</i> is the right-peak-adjacent step plateau, and proceed to Task (4). Otherwise, continue to Sub-task (v).</p>
   <p align="justify">v. If <i>m</i> + <i>r</i> = <i>n</i>, where <i>n</i> represents the number of steps in the piecewise function, then go back through all right-steps and select the step with the largest width, and proceed to Task (4).</p>
4. <p align="justify">For each potential peak step in Task (1), compute the peak width (<i>wid<sub>p</sub></i>):</p>
   <p align="justify">i. Identify the maximum period of the left-peak-adjacent step plateau (<i>T<sub>l</sub></i>).</p>
   <p align="justify">ii. Identify the minimum period of the right-peak-adjacent step plateau (<i>T<sub>r</sub></i>).</p>
   <p align="justify">iii. Compute <i>wid<sub>p</sub></i> = ln(<i>T<sub>r</sub></i>) – ln(<i>T<sub>l</sub></i>).</p>
5. <p align="justify">For each potential peak step in Task (1), compute <i>k</i> = (<i>amp<sub>p</sub></i> – 𝜂̅<sub>𝑆,𝑗</sub><sup>𝑣</sup>)/ 𝑆𝐸̅̅<sub>𝑗</sub> for the left- and right-peak-adjacent step plateaus, where 𝜂̅<sub>𝑆,𝑗</sub><sup>𝑣</sup> and 𝑆𝐸̅̅<sub>𝑗</sub> represent the average amplitude and standard error over the step width, and <i>k</i> represents a non-zero multiplier.</p>
6. <p align="justify">Identify clear peak features from among potential peaks: Clear peaks are those which satisfy the following criteria:</p>
   <p align="justify">(a) The difference between <i>amp<sub>p</sub></i> and the maximum of <i>amp<sub>l</sub></i> and <i>amp<sub>r</sub></i> should exceed a threshold: <i>amp<sub>p</sub></i> − max(<i>amp<sub>l</sub></i>, <i>amp<sub>r</sub></i>) ≥ <i>amp<sub>thres</sub></i>.</p>
   <p align="justify">(b) The peak should not be too wide: <i>wid<sub>p</sub></i> ≤ <i>wid<sub>thres</sub></i>.</p>
   <p align="justify">(c) There should be sufficient confidence that the mean peak amplitude (<i>amp<sub>p</sub></i>) is greater than the right- and left-peak-adjacent step plateau amplitudes: min(<i>k<sub>l</sub></i>, <i>k<sub>r</sub></i>) ≥ <i>k<sub>thres</sub></i>.</p>
<br>

![Figure 7](https://github.com/tristanbuckreis/srPeak/assets/71461454/d8dd17de-dc54-4032-b59c-143eb5eb2c66)
<p align="justify"><b>Figure 2.</b>Flowchart illustrating site response peak detection algorithm utilizing tree regression stepped results; amp parameters relate to step amplitudes; wid and step parameters relate to step widths; <i>T</i> represents the periods within the step; subscripts <i>l</i>, <i>p</i>, and <i>r</i> indicate the left, peak, and right steps, respectively; <i>n</i> is the total number of steps; 𝜂̅<sub>𝑆,𝑗</sub><sup>𝑣</sup> and <𝑆𝐸̅̅<sub>𝑗</sub> are the average 𝜂<sub>𝑆,𝑗</sub><sup>𝑣</sup> amplitude and standard error within the step, respectively; and <i>step<sub>thres</sub></i>, <i>k<sub>thres</sub></i>, <i>amp<sub>thres</sub></i>, and <i>wid<sub>thres</sub></i> are adjustable algorithm parameters.</p>

# Functions:

## Identify_Site_Response_Peaks
```python
Identify_Site_Response_Peaks(period, site_response, standard_error, 
                             cp_alpha = 0.0003, step_thres = 0.65, amp_thres = 0.27, 
                             wid_thres = 2.3, k_thres = 0.9, plot = True)
```

Function to identify peak features in residual site response, using automated algorithm presented in Buckreis et al. (202x)

Input Arguments:
  - ```period``` = array of periods
  - ```site_response``` = array of residual site response, at periods specified in ```period```
  - ```standard_error``` = array of standard errors for residual site response, at periods specified in ```period```
  - ```cp_alpha``` = complexity parameter (default = 0.0003)
  - ```step_thres``` = threshold for width of "stable peak tails" (default = 0.65)
  - ```amp_thres``` = threshold for relative peak amplitude = peak - tail (default = 0.27)
  - ```wid_thres``` = threshold for width of peak (default = 2.3)
  - ```k_thres``` = scaling constant for evaluating uncertainty of peak and tail amplitudes (default = 0.9)
  - ```plot``` = option to plot the results or not (default = True)
  
Output Arguments:
  - if ```plot == 0```: ```[peak_indicator]```<br>
      ```peak_indicator``` = ```Flase``` if there is no peak; ```True``` if there is a peak
  - if ```plot == 1```: ```[peak_indicator, fig]```<br>
      ```peak_indicator``` = ```False``` if there is no peak; ```True``` if there is a peak<br>
      ```fig``` = matplotlib Figure object


## tree_to_nodes
```python
tree_to_nodes(period, site_response, cp_alpha)
```

Sub-function to fit tree-regression and extract nodes from results for peak identification code.

Input Arguments:
  - ```period``` = array of periods
  - ```site_response``` = array of residual site response, at periods specified in ```period```
  - ```cp_alpha``` = complexity parameter
  
Output Arguments:
  - ```nodes``` = array of period-values corresponding to tree-regression nodes
  - ```values``` = array of site response-values corresponding to tree regression nodes


# Examples:

# References:

Breiman L., Friedman J.H., Olshen R.A., and Stone C.J. (1984) <i>Classification and Regression Trees</i>. Wadsworth.

Buckreis, T.E. (2022) Customization of path and site response components of global ground motion models for application in Sacramento-San Joaquin Delta region of California. Doctoral dissertation, University of California, Los Angeles, Los Angeles, CA.

Wang P., Zimmaro P., Ahdi S.K., Yong A., and Stewart J.P. (2023) Identification protocols for horizontal-to-vertical spectral ratio peaks. <i>Bulletin of the Seismological Society of America</i> 113(2): 782 – 803.
