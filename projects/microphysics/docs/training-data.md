# Training Data

The training data is generated from several simulations with the FV3GFS atmospheric model.

{numref}`log-qc-before` shows the histogram of cloud water before and after the Zhao-carr Microphysics.
Trace amounts of cloud water with concentrations less than $10^{-22}$ kg/kg occur frequently in the input and output.
The zhao-carr scheme seems to substantially reduce the clouds for the larger band.

```{glue:figure} qc-before-after
:name: log-qc-before
Cloud water histograms before and after the Zhao-Carr microphysics. The total water change is {glue:text}`qc-before-after-total-percent:.2f` %.
```