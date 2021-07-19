---
title: 'pyCSEP: A Python Package For Earthquake Forecast Developers' 
tags:
  - python
  - seismology
  - forecasting
  - statistical seismology
  - seismic hazard 
authors:
  - name: William H. Savran^[Corresponding author reachable at wsavran \[at\] usc \[dot\] edu]
    orcid: 0000-0001-5404-0778
    affiliation: 1
  - name: Maximilian J. Werner
    affiliation: 2
  - name: Danijel Schorlemmer
    affiliation: 3
  - name: Philip J. Maechling
    affiliation: 1
affiliations:
  - name: Southern California Earthquake Center, University of Southern California
    index: 1
  - name: University of Bristol
    index: 2
  - name: GFZ Potsdam
    index: 3
date: 01 February 2021 
bibliography: paper.bib 

---

# Summary

For government officials and the public to act on real-time forecasts of earthquakes, the seismological community needs to
develop confidence in the underlying scientific hypotheses of the forecast generating models by assessing their
predictive skill. For this purpose, the Collaboratory for the Study of Earthquake Predictability (CSEP) provides
cyberinfrastructure and computational tools to evaluate earthquake forecasts. Here, we introduce pyCSEP, a Python package to
help earthquake forecast developers embed model evaluation into the model development process. The package contains the
following modules: (1) earthquake catalog access and processing, (2) data models for earthquake forecasts, (3) statistical
tests for evaluating earthquake forecasts, and (4) visualization routines. `pyCSEP` can evaluate earthquake forecasts expressed as
expected rates in space-magnitude bins, and simulation-based forecasts that produce thousands of synthetic seismicity catalogs.
Most importantly, `pyCSEP` contains community-endorsed implementations of statistical tests to evaluate earthquake forecasts,
and provides well defined file formats and standards to facilitate model comparisons. The toolkit will facilitate integrating
new forecasting models into testing centers, which evaluate forecast models and prediction algorithms in an automated,
prospective and independent manner, forming a critical step towards reliable operational earthquake forecasting. 

# Background

Successfully predicting the time, location, and size of future earthquakes would have immense societal value, and this quest
underlies much of the research in seismology and earthquake geology. To date, however there have been no reliable earthquake
predictions methods. An earthquake prediction makes a deterministic statement about whether or not an earthquake will occur in
a particular geographic region, time window, and magnitude range. On the other hand, an earthquake forecast provides the
probability that such an earthquake will occur [@Jordan2011a]. Most of the current research effort focuses on developing
probabilistic earthquake forecasting models that encode empirical or physics-based hypotheses about the occurrence of
seismicity.  To what degree earthquakes can be predicted remains an open and important question. 

As @Schorlemmer2018a states "the fundamental idea of CSEP is simple in principle but complex in practice: earthquake forecasts
should be tested against future observations to assess their performance, thereby ensuring an unbiased test of the forecasting
power of a model." Practically, this requires a prospective evaluation of the earthquake forecasts. Prospective evaluation
requires that model developers fully specify their models (with zero-degrees of freedom) before the experiment begins
[@Schorlemmer2018a]. Specific parameters for an experiment are determined through community consensus, such as the geographic
testing region and magnitude range, authoritative data sets used to evaluate the forecasts, the evaluation metrics, and the precise
specification of a forecast. These parameters are defined in full before the start of the experiment. This standardization
ensures that any potential conscious or unconscious biases are reduced, because the evaluation data are collected after each
model has been provided for evaluation.

# Statement of need

Over the last decade, CSEP has led numerous prospective earthquake forecasting experiments [see, e.g., @Michael2018a]. These
experiments are formally conducted within testing centers [@Schorlemmer2007b] that contain the software required to
autonomously run and evaluate earthquake forecasts in a fully prospective mode. The software design emphasized a carefully
controlled computing and software environment which ensured integrity of testing results [@Zechar2009a]. However, the
monolithic software design made it difficult for researchers to utilize various routines in the testing centers in their own
work without replicating the entire testing center configuration on their own system. In addition, software was developed by a
single developer, leading to personnel risk and a lack of opportunities for others to contribute directly.   

`pyCSEP` was designed to provide vetted methods to evaluate earthquake forecasts in a Python package that researchers can
include directly in their research. The statistical tests and tools to evaluate earthquake forecasts are required by all
model developers, and greatly benefit from open-source development practices by providing standardized, well-tested, and
community-reviewed software tools. At the time of publication, `pyCSEP` has been used for two published articles [@Bayona2020a;
@Savran2020a], and is being used by several research groups participating in the [Real-time earthquake risk reduction for a
resillient europe (RISE)](http://www.rise-eu.org/home) project and others.

# pyCSEP Overview

`pyCSEP` provides an open-source implementation of peer-reviewed statistical tests developed for evaluating probabalistic
earthquake forecasts [@Schorlemmer2007a; @Werner2011a; @Rhoades2011a; @Zechar2013a; @Savran2020a]. In addition, `pyCSEP`
provides routines for working with earthquake catalogs and visualizations. The core design of `pyCSEP` includes classes that
represent earthquake forecasts, catalogs, and various spatial regions. Higher level functions are implemented using these
classes to provide routines for common tasks in evaluating earthquake forecasts. 

Earthquake forecasts can either be specified as expected earthquake rates over discrete space-magnitude-time regions
[@Schorlemmer2007a] or as families of synthetic earthquake catalogs with each catalog representing a realization from the
underlying stochastic model [e.g., @Savran2020a].  Earthquake catalogs are row-based data sets that contain features of an
earthquake. At a minimum, an earthquake must be defined by its geographical location (latitude, longitude), origin time, and
magnitude. In addition, `pyCSEP` provides classes for working directly with forecasts from the Uniform California Earthquake
Rupture Forecast with Epidemic-type Aftershock Sequences Version 3 [@Field2017a].  `pyCSEP` also provides classes for
interacting with earthquake catalogs and performing operations on them, such as filtering and binning events on the
space-magnitude grids needed for evaluation. `pyCSEP` also includes numerous plotting utilities that interface directly with
`matplotlib` and `Cartopy`.  Space-magnitude regions facilite gridding operations that are necessary for evaluating earthquake
forecasts.  These objects model regular latitude, longitude cells where earthquakes can be aggregated for evaluation and
visualization purposes. `pyCSEP` provides pre-defined spatial regions that have been used in previous experiments
[@Field2007a; @Taroni2018a]. 

`pyCSEP` interfaces directly with popular numerical and plotting libraries such as `Numpy`, `matplotlib`, and `pandas`.
Users already familiar with these librarys can adapt `pyCSEP` directly into their code. `pyCSEP` provides file-formats
for forecasts and earthquake catalogs, and can allow users to specify custom filetypes.

# Acknowledgements

This research was supported by the Southern California Earthquake Center (Contribution No. 11030). SCEC is funded by NSF
Cooperative Agreement EAR-1600087 & USGS Cooperative Agreement G17AC00047. Maximilian J. Werner and Danijel Schorlemmer received funding from the European Union’s Horizon 2020 research and innovation program (Number 821115, RISE: Real-Time Earthquake Risk Reduction for a Resilient Europe) 

# References

