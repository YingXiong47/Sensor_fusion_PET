# NIR/SWIR Spectral Library of Plastic-Substrate Mixtures

## Overview
This dataset contains Near-Infrared (NIR) and Short-Wave Infrared (SWIR) reflectance spectra for various mixtures of plastics and substrates. The data were collected to support the detection and mapping of plastic pollution using hyperspectral imaging. It includes preprocessed mean spectra for reflectance, absorption, first derivatives, and second derivatives, as well as raw ASD spectra and a python script for processing them.

The dataset comprises 228 unique substrate-plastic-concentration combinations, averaged from a minimum of 30 replicate NIR spectra per sample. Preprocessed spectra are included for ease of use, and raw spectra are provided along with the python script for processing.

For more details about the spectral collection, preprocessing methods, and analysis, refer to the accompanying paper (currently under review):  
[Holt, Z.K.; Khan, S.D.; Rodrigues, D.F.], *Hyperspectral Remote Sensing of Environmental Plastic Pollution at a Landfill Site*. [Environmental Science & Technology].

---

## Files in This Dataset
1. **__Mean_Spectra_Reflectance.csv**:
   - Contains mean reflectance spectra for all substrate-plastic-concentration combinations.
   - Columns:
     - `wavelength`: Wavelengths of the spectra (850–2400 nm).
     - Subsequent columns represent reflectance values for each substrate-plastic-concentration combination (e.g., `Bu5_PE_0X15`).

2. **__Mean_Spectra_Absorption.csv**:
   - Contains absorption spectra derived from the reflectance data using the Beer-Lambert Law.
   - Columns are structured the same as in the reflectance file.

3. **__Mean_Spectra_Absorption_der1.csv**:
   - Contains the first derivative of the absorption spectra.
   - Columns are structured the same as in the reflectance file.

4. **__Mean_Spectra_Absorption_der2.csv**:
   - Contains the second derivative of the absorption spectra.
   - Columns are structured the same as in the reflectance file.

5. **spectral_library_plastics.hdr**:
   - Header file for the ENVI spectral library, containing metadata for compatibility with hyperspectral imaging software.

6. **NIR_spectra_plastic_mixtures_ASD.zip**:
   - Contains the raw NIR spectra (8240 spectra) collected using the ASD FSP 350-2500P spectroradiometer.

7. **NIR_PlasticMixes_BandEQs.py**:
   - A python script used to read and process the raw spectra, generating the preprocessed CSV files provided in this dataset

---

## Data Description
### Spectral Data
- **Wavelength Range**: 850–2400 nm.
- **Substrates**:
  - Soils: `Bu5`, `TG`, `W6`.
  - Concrete: `C`.
  - Vegetation (oak leaves): `V`.
  - Water: `DIW`.
- **Plastics**:
  - Polyethylene (PE)
  - Polyethylene terephthalate (PET)
  - Polylactic acid (PLA)
  - Polypropylene (PP)
  - Polyvinyl chloride (PVC)
  - Styrene-butadiene rubber (SBR)
- **Concentrations**:
  - `0%`, `0.15%`, `0.5%`, `1.5%`, `5%`, `15%`, `50%`, `100%`.
  - Represented in the format `X` for decimals (e.g., `0X15` for 0.15% w/w).

### Preprocessing Steps
- **Raw Data Collection**:
  - NIR reflectance spectra were collected using an ASD FSP 350-2500P spectroradiometer.
  - Solid substrate mixtures were measured in aluminum weigh dishes, while DIW mixtures were measured in a glass beaker with a dark reference board to absorb transmitted light.
  
- **Preprocessing**:
  - Wavelengths below 850 nm were discarded to eliminate color-related features.
  - Splicing correction was applied to ensure spectral continuity.
  - Continuum removal was used to flatten baselines and enhance spectral features.
  - Reflectance data were converted to absorption using the Beer-Lambert Law.
  - Standard Normal Variate (SNV) normalization was applied to replicates collectively.
  - Savitzky-Golay filtering was used to smooth spectra and compute derivatives.
  - Corrected absorption spectra were backtransformed to reflectance for hyperspectral image analysis in ENVI.

---

## How to Use This Dataset
- **Reflectance Data**: Use as-is for spectral analysis, comparison, or hyperspectral image processing.
- **Absorption and Derivatives**: Derivative spectra are included to facilitate feature extraction for machine learning and statistical analysis.
- **ENVI Files**: Compatible with ENVI and other hyperspectral imaging software for advanced analysis.
- **Raw spectra with python script**: Use the raw NIR spectra with provided python script for custom preprocessing or analyses

---

## Citation
Please cite this dataset as:  
[Holt, Z.K.; Khan, S.D.; Rodrigues, D.F.], *Spectral Library of Plastics mixed with Environmental Substrates*, DOI: [10.5281/zenodo.14233290].

For detailed methodology, refer to the accompanying paper (currently under review):  
[Holt, Z.K.; Khan, S.D.; Rodrigues, D.F.], *Hyperspectral Remote Sensing of Environmental Plastic Pollution at a Landfill Site*. [Environmental Science & Technology].


---

## License
This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).

Under this license, you are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material for any purpose, even commercially.

Attribution:
You must give appropriate credit, provide a link to the license, and indicate if changes were made. For details, see https://creativecommons.org/licenses/by/4.0/.

---

## Contact Information
For questions or further information, please contact:  
- [Dr. Debora Rodrigues]  
- [Clemson University]  
- [dfrodri@clemson.edu]  
