# ANT-DBS-Target_Identification
Created under the framework of the Thesis "Novel Contributions to Personalized Brain Stimulation Biomarkers for Better Management of Neurological Disorders" - Doctoral Program in Biomedical Engineering (FEUP), Supervisor: João Paulo Cunha (INESC TEC, Porto, Portugal).

## Scope
Positioned in the superior region of the thalamus, the Anterior Nucleus of the Thalamus (ANT) is distinguished from the rest of the tha- lamus by the anterior medullary lamina and comprises three subnuclei known as the anteroventral (AV), anterodorsal (AD), and anteromedial (AM) nuclei. Stimulation within the ANT region is generally considered more effective than outside this area.


Identification of the DBS target relies on magnetic resonance imaging (MRI), employing either indirect or direct methodologies. The indirect method defines the target in a brain atlas using common landmarks, such as the anterior and posterior commissures, while the direct method utilizes 3T MRI techniques enabling direct visualization of white-matter structures related to the ANT, such as the external medullary lamina (EML), internal medullary lamina (IML), and mammillothalamic tract (MMT). However, both methods have drawbacks: the in- direct method is limited by anatomical variations of the target structure in the stereotactic space between individuals, whereas the direct method requires advanced imaging techniques, potentially limiting access in some DBS centers.

Here, we presented a pipeline using multichannel LFPs collected months after the electrodes’ implantation with the Medtronic Neurostimulator, the Percept PC, already out of the influence of the microlesion effect, in the guidance of the DBS target identification.

## main_code.m
This pipeline allows to predict which DBS-electrode contact presents highest signature of ANT-target, by using machine learning models and ANT signals. This pipeline comprises the follosing functions:

PACKAGES
* Machine learning toolbox for classification and regression (@Evan Bollig, 2011)
* EEG processing and visualization toolbox (https://github.com/elodiemlopes89/Electrophysiological-Data-Preprocessing-Visualization)
* Brain Connectivity Toolbox (https://sites.google.com/site/bctnet/)
* EEG Feature Extraction Toolbox (@Jingwei Too, 2020)

### PrepareData.m
Extract all LFP data recorded in the BrainSense Survery mode and organize them into "Pass1" and "Pass2" montages.

### FeatureExtraction.m
Extract 19 features from LFP data:

STATISTICAL FEATURES:
* Variance
* Skewness
* Minimum
* Maximum
* Median

SPECTRAL FEATURES
* Band Power Delta
* Band Power Theta
* Band Power Alpha
* Band Power Beta
* Band Power Gamma
* Frequency of peak 2
* Magnitude of peak 2

MORPHOLOGICAL FEATURES
* Abosulte peak
* Mean peak

MULTIVARIATIVE FEATURES

Node strength of adjacency matrix computed using:
* Correlation
* Coherence
* Phase lag index
* Phase locking value

### Classification.m
Two-class classification model: Data is splited into 80% for training and 20% for test. Training set will be first divided into 80% for select the best model and 20% for test. 

Models include svm, knn and nn. After the selection of the best model, it will be select the best parameter of the best model. Then, the best model and best parameter will be training and tested for the first division of dataset.

RankFeatures function allow to assess which features are more relevant two discriminate between two class labels.

This function uses Principal Compoennt Analysis pipeline, described in (http://www.holehouse.org/mlclass/14_Dimensionality_Reduction.html).


### Prediction.m
Prediction of the % of ANT for each electrode channel by using the classifier obtained in the last step.

