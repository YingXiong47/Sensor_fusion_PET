# 1D CNN Plastic Spectral Classification Dataset

## Overview

This repository uses the **NIR/SWIR Spectral Library of Plastic-Substrate Mixtures** as input data for a **1D convolutional neural network (1D CNN)** model.

The original dataset contains Near-Infrared (NIR) and Short-Wave Infrared (SWIR) reflectance spectra for mixtures of plastics and environmental substrates. In this project, I am not claiming ownership of the original dataset. I am using the spectral data only as training/testing input for a 1D CNN plastic classification experiment.

## Original Dataset Credit

Original dataset:

**Holt, Z.K.; Khan, S.D.; Rodrigues, D.F.**  
*Spectral Library of Plastics mixed with Environmental Substrates*  
DOI: **10.5281/zenodo.14233290**

The dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, which allows sharing and adaptation as long as proper credit is given.

## My Use of the Data

For this project, the dataset is used to train and evaluate a **1D CNN model** on spectral reflectance data.

My modifications may include:

- Selecting specific wavelength bands
- Formatting the spectra for machine learning
- Creating train/validation/test splits
- Relabeling samples for classification tasks
- Normalizing or preprocessing spectra for model input

Any processed files in this repository are derived from the original dataset and are used for machine learning experimentation.

## Citation

Please cite the original dataset as:

> Holt, Z.K.; Khan, S.D.; Rodrigues, D.F.  
> *Spectral Library of Plastics mixed with Environmental Substrates*.  
> DOI: 10.5281/zenodo.14233290

## License

The original dataset is licensed under **CC BY 4.0**.  
This repository gives credit to the original authors and indicates that the data may have been modified for use in a 1D CNN classification model.
