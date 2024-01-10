# Bayesian at heart

This code allows to estimate HR dynamics following a Bayesian paradigm (i.e. via sampling a posterior distribution), as outlined in the paper:

_Rosas, Candia-Rivera, Luppi, Guo, & Mediano, (2023). Bayesian at heart: Towards autonomic outflow estimation via generative state-space modelling of heart rate dynamics. arXiv preprint arXiv:2303.04863._

Additionally, this code also allows to calculate Bayesian estiamtes of HR entropy as presented in the paper:

_Rosas*, Mediano*, et al. (2023). The entropic heart: Tracking the psychedelic state via heart rate dynamics. bioRxiv, 2023-11._

Please cite these papers - and let us know! - if you use this software. Please contact Fernando Rosas for bug reports, pull requests, and featrure requests.


## Download, installation, and code examples

The software requires no installation, just normal repository cloning - which should take just a few seconds.

The estimation of HR dynamics is done by the file _generate_hr.py_. This code works in Python 3, but calls the function _gmc_inference.jl_ which runs on Julia. To let the Python code to call Julia you need to install the package [PyJulia](https://pyjulia.readthedocs.io/en/latest/index.html). A brief example of how this code is used is provided in the file, which goes as follows:

```python
    # Define list of filenames 
    filenames = ['data.csv']

    # Calculate Bayesian and frequentist estimation of HR
    bayes_hr = bayesian_hr(filenames, IT=3)
    freq_hr  = frequentist_hr(filenames)
```

This code should generate a Pandas dataframe, where each of the `IT` column correspond to one sampled trajectory from the posterior distribution (see below). For comparison, the code also calculates the standard frequentist estimation of HR dynamics.

The calculation of HR entropy is done by the file _calculate_HRentropy.py_. This code works in Python 3, but calls Java code. For this, it needs the Python package [JPype](https://pypi.org/project/JPype1/). A small example of how the code works is provided in the file:

```python
    # Load Bayesian estimations
    data = pd.read_csv('./bayes_hr.csv', index_col=0)

    # Calculate HR entropy
    data_diff = data.diff().dropna()
    lz = ctw_entropy( data_diff)
    mean_lz = lz.mean()
```

This code generates a Pandas series `lz` which contains the calculated entropy of each trajectory, which corresponds to the posterior distribution of this property (see below). Above, we are calculating the mean value of this posterior.

Runtime of the generation of sampled trajectories of this example should take less than a minute; however, estimating hundreds of samples from e.g. 5mins ECG data can take a couple of hours on a typical laptop. The calculation of the HR entropy depends on the number of sampled trajectories, but should take not more than tens of minutes on a regular laptop even if there are hundreds of samples.

The code has none non-standard hardware requirements. The code has been tested in Python 3.9 and Julia 1.9.4.


## Background: A Bayesian approach for the estimation of HR dynamics

The conventional method to calculate heart rate involves inferring how many beats one would expect per minute on average given the observation of $N_\text{b}$ beats over a period of time of $T$ seconds, which leads to the estimate $\text{HR}=60 N_\text{b} / T$. 
If one is interested in a dynamical description of how the heart rate fluctuates over time, one can follow the same rationale and reduce the time period to the limit where $N_\text{b}\to1$ and $T$ becomes equal to the inter-beat interval $I_\text{b}$,  leading to the following estimate of the ``instantaneous'' heart rate:

$\text{HR}(t) = \frac{60}{I_\text{b}(t)}$.

From a statistical perspective, this expression can be understood as the outcome of an elementary frequentist method of inference that delivers a point estimate for the average number of beats per minute - in fact, it is the number of beats one would see if all beats were separated by the same inter-beat interval $I_\text{b}$. As such, it has the strengths and weaknesses of frequentist approaches: it is conceptually simple and computationally lightweight, although it cannot estimate its own uncertainty or incorporate prior knowledge on plausible heart rate values. Furthermore, as $\text{HR}(t)$ ignores previous inter-beat interval values, errors in the estimation of $I_\text{b}(t)$ inevitably lead to overestimations of heart rate fluctuations.

In contrast, our proposed approach conceives the heart rate as a hidden process that drives the actual observed heart beats, the statistical properties of which can be estimated via generative modelling.
Our model involves two time series corresponding to the values of a dynamical process sampled with sampling frequency $f_\text{s} = 1/\Delta t$: $x_t$, which counts the number of heart beats that take place
during a temporal bin of length $\Delta t$, and $z_t$, which is the heart rate that drives the corresponding heart beats. 
Our framework comprises a generative statistical model, the key component of which is a probability distribution $p$ that describes the likelihood of observing a given sequence of heart beats $x_1,\dots,x_N$ together with a heart rate time series $z_1,\dots,z_N$ .

![alt text](https://github.com/ferosas/BayesianAtHeart/blob/main/SSDiagram.pdf?raw=true)

Through this model, heart rate dynamics are now described not by a point estimate (i.e. as a single, most likely trajectory) but as obeying the following conditional distribution:

$\text{HR}_\text{bayes}: \quad z_1,\dots,z_N \sim  p(z_1,\dots,z_N|x_1,\dots,x_N).$

This posterior distribution describes the most likely heart rate trajectories $z_1,\dots,z_N$ given the observed data $x_1,\dots,x_N$. 
Crucially, by following methods proposed in this paper:

_Gugushvili, Van Der Meulen, Schauer, & Spreij, P. (2018). Fast and scalable non-parametric Bayesian inference for Poisson point processes. arXiv preprint arXiv:1804.03616._

we avoid the computationally intensive task of computing the explicit posterior distribution, and instead use a Gibbs sampler to efficiently obtain sample trajectories. 
This allows not only to find the most likely trajectory, but also to estimate uncertainty (e.g. via the posterior variance). 
These sampled trajectories also allow us to build accurate estimators of non-linear properties of heart rate dynamics - as illustrated bellow with the calculation of HR entropy. 
The sampling procedure employs two hyperparamenters $\theta$ and $\tau$, which are related with the connectivity strength between successive samples. In our experiments, sampled trajectories were seen to be fairly insensitive to the choice of $\tau$, while their smoothness strongly depended on $\theta$ - low values of $\theta$ induce a strong constraint between successive samples, making more unlikely abrupt changes of values.

## Background: Bayesian estimators of HR entropy

The sampled trajectories of heart rate dynamics to build Bayesian estimators of properties of heart rate dynamics. To explain this part of the method, we introduce the shorthand notation $z = (z_1,\dots,z_T)$ and $x = (x_1,\dots,x_T)$ for sampled trajectories of heart rate and heart beats respectively, and let $F(z)$ be a scalar function of this trajectory - i.e. any scalar property of the heart rate trajectory, such as its mean value or entropy. Then, the generative model above can be used to derive the posterior distribution of the property $F$, which corresponds to $p(F(z) |x)$. Sampled trajectories can be used to estimate various properties of this posterior - e.g. its mean:

$\hat{F} = \sum_{z} F(z) p(z|x), $ 

where the value of the property $F$ for each possible trajectory is weighted by the likelihood of such trajectory given the observed data. We use this approach to estimate the entropy of HR dynamics as explained below.

Brain entropy in neuroimaging data is usually calculated via Lempel-Ziv complexity (referred to as LZ), which estimates how diverse the patterns exhibited by a given signal. The method was introduced by engineers Abraham Lempel and Jacob Ziv to study the statistics of binary sequences (later becoming the basis of the well-known _zip_ compression algorithm), and has been used to study patterns in brain activity for more than 20 years. Unfortunately, there are two issues that make it challenging to apply LZ complexity to heart rate data: the need of a binarisation step combined with the non-stationarity of the time series (observed here in all drugs except LSD), and the relatively short length of the time series (LZ is usually estimated on brain data over windows of thousands of samples, which is challenging with a sampling frequency of 1 Hz). We addressed these challenges with two innovations:

- First, entropy estimation in neuroimaging requires the binarisation of the data, which is often done by thresholding on the signal's mean value. While this particular choice usually doesn't have a big impact on the entropy estimates, it becomes problematic with highly non-stationary data, as it could lead to an underestimation of the entropy due to long periods of the signal being either above or bellow its mean value. To avoid this problem, instead of thresholding based on the mean value, we threshold according to the sign of the derivative - hence a '1' implies the signal is increasing and a '0' that it is decreasing.

- To address the issue of the low sampling frequency and resulting short length of the time series data, we don't use the classic LZ algorithm but instead estimate entropy using the _Context Tree Weighted_ (CTW) algorithm. This algorithm has shown to converge quicker than other entropy rate estimators including LZ.
