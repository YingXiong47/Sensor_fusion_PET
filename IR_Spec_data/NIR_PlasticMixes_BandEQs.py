# Developed by Zach Holt, 2024

# The purpose of this script is to process spectral profiles of substrate-plastic-concentration mixtures

import os
import glob
import pandas as pd
import numpy as np
import scipy as sp
import specdal
import matplotlib.pyplot as plt; plt.style.use('bmh') # Set default plot style for all figures
import seaborn as sns
from pyspectra.transformers.spectral_correction import snv
from scipy.signal import savgol_filter
import itertools
from sklearn.decomposition import PCA



d = r'C:\Users\zachk\Desktop\RS\ZH_Specrad\RS_plastics' # Define root directory
os.chdir(d) # Set working directory



# Spectral pre-processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import & format spectral data
c = specdal.Collection(name = 'specrad') # collection of spectra
for f in os.listdir(d): # for each file in directory
    if f.endswith('.asd'): # if .asd extension
        spec = specdal.Spectrum(filepath = os.path.join(d, f)) # grab spectra
        c.append(spec) # add to collection
print('Raw spectra imported')
        
# Transfer specdal object to dataframe
df = c.data
samples = list(df.columns) # get list of sample names
df = df.reset_index() # get wavelength as a column, index from 0
df_raw = df.copy() # Store it for plotting later

# Define levels of variables
soil = ['', 'Bu5', 'W6', 'TG', 'C', 'V', 'DIW']
plastic = ['', 'PE', 'PET', 'PLA', 'PP', 'PVC', 'SBR']
pconc = ['0', '0X15', '0X5', '1X5', '5', '15', '50', '100'] # 'X' represents the decimal, but didn't use '.' in file naming
labels = [f'{p}%'.replace('X','.') for p in pconc] # String labels of concentrations with decimal and percentage

# Make color palette
colmap = plt.get_cmap('jet_r')
colores = [colmap(x) for x in np.linspace(0, .95, len(pconc))]
colores = colores[::-1]
colores = list(tuple([.7*x for x in colores[y][:-1]]) for y in range(len(colores)))
plt.figure(figsize=(10, 1)); [plt.bar(i, 1, color=color) for i, color in enumerate(colores)]; plt.title('Color palette used for plotting spectra', fontsize = 18) # Show color map

# Filter by wavelength
df = df[ df['wavelength'].between(850,2400)].reset_index(drop = True) # remove visible portion, keep index from 0

# Manual splice correction at known jumps
for j in range(1,len(df.columns)):
    b1 = int(1001 - min(df['wavelength'])) # index for 1st jump, shifted based on beginning wavelength
    b2 = int(1831 - min(df['wavelength']))
    d1 = df.iloc[b1, j] - df.iloc[b1-1, j] # First jump at 1001 nm
    d2 = df.iloc[b2, j] - df.iloc[b2-1, j] # 2nd jump at 1831 nm
    df.iloc[b1 : len(df['wavelength'] - 1), j] = df.iloc[b1 : len(df['wavelength'] - 1), j] - d1 # Adjust all values to right of jump by value
    df.iloc[b2 : len(df['wavelength'] - 1), j] = df.iloc[b2 : len(df['wavelength'] - 1), j] - d2
df_corr = df.copy() # Store it for plotting later

# Continuum removal
def continuum_removal(points):
    x, y = points.T
    augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
    hull = sp.spatial.ConvexHull(augmented)
    continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
    continuum_function = sp.interpolate.interp1d(*continuum_points.T, bounds_error=False, fill_value="extrapolate")
    yprime = y / continuum_function(x)
    return np.c_[x, yprime]

for j in range(1, len(df.columns)):  # For each spectrum
    for p in plastic:
        if p in df.columns[j]:
            points = np.c_[df['wavelength'].values, df.iloc[:, j].values]
            continuum_removed_points = continuum_removal(points)
            df.iloc[:, j] = continuum_removed_points[:, 1]
df_contrmv = df.copy() # Store it for plotting later

# Reciprocal-log transformation: Beer-Lambert's Law
if df.values.min() <= 0: # First ensure everything is positive
    df.loc[:, ~df.columns.str.startswith('wave')] += -df.values.min() + 0.01
df.iloc[:,1:] = (-np.log10(df.iloc[:,1:]))
df_abs = df.copy() # Store it for plotting later

# Standard Normal Variate Transformation removes additive and multiplicative baseline variation without altering the shape of the spectra (Comprehensive Chemometrics, 2009)
# SNV should be applied to only the same sample
for i in range(0, len(soil)): # For each soil
    for j in range(0, len(plastic)): # For each plastic
        for k in range(0, len(pconc)): # For each concentration
            n = soil[i] + '_' + plastic[j] + '_' + pconc[k] # 'soil_plastic_pconc'
            SNV = snv()
            df.loc[:, df.columns.str.contains(n)] = SNV.fit_transform(df.loc[:, df.columns.str.contains(n)])
df_snv = df.copy() # Store it for plotting later
   
# Function to rescale between 0-1
def rescale(y):
    y = (y - min(y)) / (max(y) - min(y))
    return y

# Savitzky-Golay Fltering & derivatives
w = 51
der, der2 = df.copy(), df.copy() # Make a copy of df, will replace spectra with their derivative
for j in range(1, len(df.columns)): # For each column after wavelength (each spectra)
    df.iloc[:,j] = rescale(savgol_filter(df.iloc[:,j], w, polyorder = 4, deriv = 0)) # smoothing zero derivative, and rescale
    der.iloc[:,j] = savgol_filter(df.iloc[:,j], w, polyorder = 4, deriv = 1) # smoothing 1st derivative
    der2.iloc[:,j] = savgol_filter(df.iloc[:,j], w, polyorder = 4, deriv = 2) # smoothing 2nd derivative
df_sg = df.copy() # Store it for plotting later

# Plots for each step of pre-processing
def substepsubplots(plastic, soil, pconc, df, supt, nrow, ncol): 
        fig, axs = plt.subplots(nrows = nrow, ncols = ncol, figsize = (16, 9))
        for i, soil_type in enumerate(soil): # For each soil (column index)
            for j, concentration in enumerate(pconc): # For each concentration (row index)
                n = f'{soil_type}_{plastic}_{concentration}' # name of sample
                if concentration == '100': # special cases
                    n = f'_{plastic}_100'
                elif concentration == '0':
                    n = f'{soil_type}__0'
                yyy = df.filter(like=n) # Pull all spectra replicates for a sample
                r = len(yyy.columns)
                [axs[j,i].plot(df['wavelength'], yyy.iloc[:,R], color = colores[R % 6], linewidth = 2, alpha = 0.5) for R in range(r)] # Add plots for replicate spectra
                axs[j,i].text(min(df['wavelength']), yyy.max(axis = None), n, verticalalignment = 'top')
                plt.suptitle(f'{supt} - {plastic}', fontsize = 28)
             
prepareplots = False # Toggle True/False if you want to produce subplots
if prepareplots:
    nrow, ncol = len(pconc), len(soil)-1 # rows for each concentration, columns for each soil, create individual plots for each plastic
    for p in plastic[1:]: # For each non-blank element in plastic list
        
        substepsubplots(p, soil[1:], pconc, df_raw, 'Raw Reflectance Spectra', nrow, ncol) # Plot raw spectra
        substepsubplots(p, soil[1:], pconc, df_corr, 'Reflectance Spectra w/ Basic Corrections', nrow, ncol) # Plot with jumps corrected
        substepsubplots(p, soil[1:], pconc, df_contrmv, 'with corr. & Continuum Removal', nrow, ncol)
        substepsubplots(p, soil[1:], pconc, df_abs, 'Absorbance Spectra w/ Corr.', nrow, ncol) # Plot converted to absorbance
        substepsubplots(p, soil[1:], pconc, df_snv, 'Abs. Spec. w/ SNV Transformation', nrow, ncol) # Plot after SNV transformation
        substepsubplots(p, soil[1:], pconc, df_sg, 'Abs. Spec. w/ S-G Filtering', nrow, ncol)
        substepsubplots(p, soil[1:], pconc, der, '1st Derivative Spectra', nrow, ncol)
        substepsubplots(p, soil[1:], pconc, der2, '2nd Derivative Spectra', nrow, ncol)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Spectral preprocessing complete') # Preprocessing complete



# Prepare processed spectra for output
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mean replicates
ns = list(itertools.product(soil[1:], plastic[1:], pconc[1:-1])) + list(itertools.product(soil[1:], [''], ['0'])) + list(itertools.product([''], plastic[1:], ['100'])) # All unique combinations of the variables, including controls
ns_str = ['wavelength'] + ['_'.join(item) if isinstance(item, tuple) else item for item in ns] # List of column names to use in averaged dataframe
ns_str_match = ['wavelength'] + ['_'.join(item) + '_' if isinstance(item, tuple) else item for item in ns] # List of column names to use in averaged dataframe

mean_spec = pd.DataFrame(columns = ns_str) # make dataframe for mean spectra
mean_der1 = pd.DataFrame(columns = ns_str)
mean_der2 = pd.DataFrame(columns = ns_str)
for i, i_match in zip(ns_str, ns_str_match): # For loop to take mean of replicates for each sample
    if i == 'wavelength':
        mean_spec[i] = df[i]  # Directly copy the wavelength column
        mean_der1[i] = df[i]  # Directly copy the wavelength column
        mean_der2[i] = df[i]  # Directly copy the wavelength column
    else:
        isocols = [col for col in df.columns if col.startswith(i_match)]
        mean_spec[i] = df[isocols].mean(axis=1)  # Take mean spectra
        mean_der1[i] = der[isocols].mean(axis=1)  # Take mean 1st derivative
        mean_der2[i] = der2[isocols].mean(axis=1)  # Take mean 2nd derivative

# Back-transformation of absorbance to reflectance for ENVI spectral library
mean_ref = mean_spec.copy()
mean_ref.iloc[:,1:] = 10**(-mean_ref.iloc[:,1:] )

for k in mean_ref.columns[1:]: # For each non-wavelength column (spectra)
    mean_ref.loc[:,k] = rescale(mean_ref.loc[:,k]) # Rescale spectra

exportcsv = True
if exportcsv:    
    mean_spec.to_csv('__Mean_Spectra_Absorption.csv', index = False) # Save spectra to .csv
    mean_der1.to_csv('__Mean_Spectra_Absorption_der1.csv', index = False)
    mean_der2.to_csv('__Mean_Spectra_Absorption_der2.csv', index = False)
    mean_ref.to_csv('__Mean_Spectra_Reflectance.csv', index = False)
    
# Export .sli file for ENVI (reflectance spectra)
export_sli = True
if export_sli:
    # Export as spectral library file for ENVI
    wvs = mean_ref['wavelength'].values
    specs = mean_ref.drop(columns='wavelength').values.T
    
    # Binary data file
    binary_f = r'C:\Users\zachk\Desktop\RS\spectral_library_plastics.sli'
    with open(binary_f, 'wb') as f:
        specs.astype(np.float32).tofile(f)
        
    # Header file
    header_f = r'C:\Users\zachk\Desktop\RS\spectral_library_plastics.hdr'
    with open(header_f, 'w') as f:
        f.write('ENVI\n')
        f.write('description = {Spectral Library}\n')
        f.write(f'samples = {wvs.size}\n')
        f.write(f'lines   = {specs.shape[0]}\n')
        f.write('bands   = 1\n')
        f.write('header offset = 0\n')
        f.write('file type = ENVI Spectral Library\n')
        f.write('data type = 4\n')  # float32
        f.write('interleave = bsq\n')
        f.write('byte order = 0\n')
        f.write('wavelength units = Nanometers\n')
        f.write('reflectance scale factor = 1.0\n')
        f.write('z plot titles = {Wavelength, Value}\n')
        f.write(f'z plot range = {{0.00, 1.00}}\n')
        f.write(f'band names = {{Sample Spectra}}\n')
        f.write('spectra names = {\n')
        f.write(', '.join(mean_ref.columns[1:]) + '}\n')
        f.write('wavelength = {\n')
        f.write(', '.join(f'{w:.5f}' for w in wvs) + '}\n')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Mean spectra exported') # Spectra exported



# Compute spectral angle between all spectra
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compute_spec_ang = False
if compute_spec_ang: # This may take a while
    def spec_ang(a, b):
        dot_prod = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        angle = np.arccos(dot_prod / (norma * normb))
        return angle
    
    def compute_spec_ang(df, s, p, pconc):
        specang = []
        A = f'{s}__0_' # pure soil/substrate
        B = '{s}_{p}_{pconc}_' # sample spectra
        As = df.loc[:, df.columns.str.contains(A)] # Grab columns for pure soil/substrate
        Bs = df.loc[:, df.columns.str.contains(B)] # Grab columns for sample spectra
        for a in As.columns: # For each reference spectrum
            for b in Bs.columns: # For each sample spectrum
                specang.append(spec_ang(As[a], Bs[b])) # Spectral angle between sample and reference
        return specang
            
    distinct_samples = mean_spec.columns[~mean_spec.columns.str.contains('wave')].tolist()
    n_distinct = len(distinct_samples)
    spectralangle = np.empty((n_distinct,n_distinct), dtype = object)
    sample_info = [samp.split('_') for samp in distinct_samples]
    
    for i, sample_i in enumerate(distinct_samples):
        for j, sample_j in enumerate(distinct_samples):
            if i <= j:  # To avoid redundant calculations
                # Isolate columns for replicates of sample_i and sample_j
                A = df.loc[:, df.columns.str.contains(f'{sample_i}_')]
                B = df.loc[:, df.columns.str.contains(f'{sample_j}_')]
                
                angles = []
                # Compute spectral angles between all replicates
                for col_A in A.columns:
                    for col_B in B.columns:
                        if col_A != col_B:
                            angle = spec_ang(A[col_A], B[col_B])
                            angles.append(angle)
                
                # Calculate mean and standard deviation
                mean_angle = np.mean(angles)
                std_angle = np.std(angles)
                
                # Store in the NumPy array as a tuple (mean, std)
                spectralangle[i, j] = (mean_angle, std_angle)
                if i != j:
                    spectralangle[j, i] = (mean_angle, std_angle)  # Ensure symmetry
                    
    spectralangle = pd.DataFrame(spectralangle, index = distinct_samples, columns = distinct_samples)
    spectralangle.to_csv('Spectral_Angle_raw.csv')
    mean_specang = spectralangle.applymap(lambda x: x[0])
    mean_specang.to_csv('Specral_Angle_Comparisons_Mean.csv')
    stdev_specang = spectralangle.applymap(lambda x: x[1])
    stdev_specang.to_csv('Spectral_Angle_Comparisons_stdev.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Spectral angles computed between all pairs of spectra in library') # Spectral angles computed



# Generation of band equations from PCA method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Identify wavelengths of interest by correlation with plastic concentration
good_band_ranges = [(850, 1340), (1449, 1789), (1975, 2415)] # List of good bands from AVIRIS image opened in ENVI (change for other sensors)

# Function to check if a wavelength is within good bands
def is_within_goodbands(wavelength, good_band_ranges):
    for start, end in good_band_ranges:
        if start <= wavelength <= end:
            return True
    return False

# Function to perform PCA on spectra, identify key wavelengths, and generate the plastic index
def key_wvs_band_eqs(p):
    # Identify columns containing the plastic name and relevant concentrations
    pcols = [col for col in mean_ref.columns if f'{p}_' in col and any(x in col for x in (soil[1:] + [f'_{p}_100']))]
    pdata = mean_ref[['wavelength'] + pcols].copy()
    
    # Filter to keep wavelengths within defined good bands
    pdata = pdata[pdata['wavelength'].apply(lambda x: is_within_goodbands(x, good_band_ranges))]

    # Extract concentration values and remove columns with no variance
    concs = np.array([float(col.split('_')[2].replace('X', '.')) / 100 for col in pcols])
    constants = [col for col in pcols if len(pdata[col].unique()) <= 1]
    pdata = pdata.drop(columns=constants)
    pcols = [col for col in pcols if col not in constants]

    # Transpose & standardize for PCA
    X = pdata[pcols].values.T
    X = np.hstack([X, concs.reshape(-1, 1)])
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Perform PCA and extract loadings
    pca = PCA(n_components=5).fit(X_std)
    loadings = pca.components_.T
    magnitudes = np.sqrt(np.sum(loadings[:, :1]**2, axis=1))
    angles = np.degrees(np.arctan2(loadings[:, 1], loadings[:, 0]))

    loading_df = pd.DataFrame({
        'wavelength': np.append(pdata['wavelength'].values, 'concentration'),
        'loading_magnitude': magnitudes,
        'angle': angles
    })

    # Determine the most correlated wavelengths with plastic concentration
    pconc_angle = loading_df.loc[loading_df['wavelength'] == 'concentration', 'angle'].values[0]
    loading_df['dangle'] = np.abs((loading_df['angle'] - pconc_angle + 180) % 360 - 180)
    threshold = 20  # Correlation threshold
    correlated = loading_df[(loading_df['dangle'] <= threshold) | (loading_df['dangle'] >= 180 - threshold)]
    correlated = correlated.sort_values(by='loading_magnitude', ascending=False).reset_index(drop=True)
    correlated = correlated.loc[correlated['wavelength'] != 'concentration']
    correlated['wavelength'] = correlated['wavelength'].astype(float)

    # Identify the most varying wavelengths within a defined range
    selected_wavelengths = []
    range_threshold = 20
    for _, row in correlated.iterrows():
        if not any(abs(row['wavelength'] - w) <= range_threshold for w in selected_wavelengths):
            selected_wavelengths.append(row['wavelength'])
    key_wvs = correlated[correlated['wavelength'].isin(selected_wavelengths)].reset_index(drop=True)

    # Find the nearest wavelength with negligible variation
    loading_df = loading_df.sort_values(by='loading_magnitude', ascending=False).reset_index(drop=True)
    least_var_df = loading_df.tail(len(loading_df) // 2)
    least_var_df = least_var_df.loc[least_var_df['wavelength'] != 'concentration']
    least_var_df['wavelength'] = pd.to_numeric(least_var_df['wavelength'])

    def find_nearest_negligible(wavelength):
        within_range_df = least_var_df[(least_var_df['wavelength'] - wavelength).abs() <= 50]
        if within_range_df.empty:
            return least_var_df.iloc[(least_var_df['wavelength'] - wavelength).abs().argsort()[:1]]['wavelength'].values[0]
        return within_range_df.sort_values(by='loading_magnitude').iloc[0]['wavelength']

    key_wvs['nearest_negligible'] = key_wvs['wavelength'].apply(find_nearest_negligible)
    key_wvs = key_wvs[key_wvs['wavelength'] != key_wvs['nearest_negligible']].reset_index(drop=True)

    # Determine correlation direction
    def correlation_direction(wavelength):
        col_100 = [col for col in pdata.columns if col.endswith('_100')]
        col_0X15 = [col for col in pdata.columns if col.endswith('_0X15')]
        mean_100 = pdata.loc[pdata['wavelength'] == wavelength, col_100].mean(axis=1).values[0]
        mean_0X15 = pdata.loc[pdata['wavelength'] == wavelength, col_0X15].mean(axis=1).values[0]
        return 'positive' if mean_100 > mean_0X15 else 'negative'

    key_wvs['correlation_direction'] = key_wvs['wavelength'].apply(correlation_direction)

    # Calculate weights based on reflectance differences
    weights = []
    for _, row in key_wvs.iterrows():
        key_refl = pdata.loc[round(pdata['wavelength']) == round(row['wavelength']), pdata.columns.str.endswith('_1X5')].mean(axis=1).values[0]
        stable_refl = pdata.loc[round(pdata['wavelength']) == round(row['nearest_negligible']), pdata.columns.str.endswith('_1X5')].mean(axis=1).values[0]
        weights.append(abs(key_refl - stable_refl))
    
    key_wvs['weight'] = weights
    key_wvs['weight'] /= sum(weights)
    key_wvs['material'] = p
    return key_wvs

bandeq_results = pd.DataFrame()
for p in plastic[1:-1]:
    bandeq_results = pd.concat([bandeq_results, key_wvs_band_eqs(p)], ignore_index=True)

# Generate band equations
def generate_band_eqs(bandeq_results, p):
    pdata = bandeq_results[bandeq_results['material'] == p]
    equations = []
    for _, row in pdata.iterrows():
        key_wv, stable_wv, direction, weight = row['wavelength'], row['nearest_negligible'], row['correlation_direction'], row['weight'] * 10000
        if direction == 'positive':
            equations.append(f'( ( {round(weight,4)} * (b{round(key_wv)} - b{round(stable_wv)}) ) /  {abs(round(key_wv) - round(stable_wv))} )')
        elif direction == 'negative':
            equations.append(f'( ( {round(weight,4)} * (b{round(stable_wv)} - b{round(key_wv)}) ) /  {abs(round(stable_wv) - round(key_wv))} )')

    final_eq = ' + '.join(equations)
    print(f'Material: {p}\nEquation: {final_eq}\n')

for p in plastic[1:-1]:
    generate_band_eqs(bandeq_results, p)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Band equations generated') # Band equations generated



# Apply band equations to spectral library to see co-registrations / mis-registrations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pindex = pd.DataFrame(index = [col for col in mean_ref.columns[1:] if 'SBR' not in col], columns = plastic[1:-1]) # Exclude SBR

for s in pindex.index:
    for p in pindex.columns:
        score = 0 # Initialize at 0
        pbandeq = bandeq_results.loc[ bandeq_results['material'] == p, :].reset_index(drop=True) # Extract band equation for specific plastic
        
        for index, row in pbandeq.iterrows():
            key, ref, weight, posneg = row['wavelength'], row['nearest_negligible'], row['weight'], row['correlation_direction']
            keyrefl = mean_ref.loc[mean_ref['wavelength'] == int(key), mean_ref.columns == s].values[0][0] # Extract spectral value at key wavelength
            refrefl = mean_ref.loc[mean_ref['wavelength'] == int(ref), mean_ref.columns == s].values[0][0] # Extract spectral value at ref wavelength
            
            diff = abs(keyrefl - refrefl)
            score += weight * diff
        pindex.loc[s, p] = score
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Band equations applied to all spectra in library') # Band equations applied



# Generate plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plastic Stack Subplots
labels = [f'{p}%'.replace('X','.') for p in pconc]
def plastic_stack_subplot(mean_ref, soil, p, ax_list, colores, pconc, label_text):
    for z, ax in enumerate(ax_list):  # For each row in the subplot
        s = soil[z]
        yyy = mean_ref.loc[:, mean_ref.columns.str.contains(f'{s}__0')].copy()
        cols = mean_ref.columns[mean_ref.columns.str.contains(f'{s}_{p}_')]
        yyy[cols] = mean_ref.loc[:, cols].copy()
        yyy[f'_{p}_100'] = mean_ref[f'_{p}_100'].copy()
        
        # Plot each concentration with corresponding color
        for j in range(len(pconc)):
            ax.plot(mean_ref['wavelength'], yyy.iloc[:, j], linewidth=3, color=colores[j], label=pconc[j])
        
        # Set individual subplot label
        ax.text(0.02, 0.25, f"{s}-{p}", transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='top')
        
        # Add label for each row and remove tick labels if not the last row
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(left=False)
        ax.set_facecolor('white')
        
        if z == len(ax_list) - 1:
            ax.set_xlabel('Wavelength [nm]', fontsize=18)
            ax.tick_params(axis='x', which='both', labelsize=12)
        else:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        

    # Set label text
    ax_list[0].text(0, 1.35, label_text, transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax_list[-1].set_xlabel('Wavelength [nm]', fontsize=18)
    [ax_list[i].set_xlabel([]) for i in range(len(ax_list[:-1]))]


# Main figure setup
fig, axs = plt.subplots(3, 2, figsize=(16, 18), dpi=300)

# Remove tick labels for the main 3x2 subplots
for ax in axs.flatten():
    ax.set_facecolor((1, 1, 1, 0))  # Fully transparent background
    ax.grid(False)
    for spine in ax.spines.values():  # Remove the frame (spines)
        spine.set_visible(False)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
    ax.set_yticks([])
    ax.set_ylabel('Reflectance', fontsize=18)

for i, p in enumerate(plastic[1:]):
    # Create a 6-row subplot for each plastic within the 3x2 grid
    sub_axs = [axs[i // 2, i % 2].inset_axes([0, y, 1, 0.15]) for y in np.linspace(0.8, 0, 6)]
    plastic_stack_subplot(mean_ref, soil[1:], p, sub_axs, colores, pconc, label_text=f"{chr(97 + i)}) {p}")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Spectral angle among spectra in library
def create_heatmap_subplot(subset_df, ax, label_text):
    mask = np.triu(np.ones_like(subset_df, dtype=bool), k=1)
    
    # Plot the heatmap with masking and customized settings
    sns.heatmap(subset_df, mask=mask, annot=False, cmap='Blues_r', fmt='.2f', linewidths=0, linecolor='black', ax=ax, cbar=False, vmin=0, vmax=1.3)

    
    # Add black rectangles around each non-masked cell in the lower triangle
    for i in range(subset_df.shape[0]):
        for j in range(i + 1):  # Only iterate over the lower triangle
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=1))
            
    thick_line_indices = [1, 8, 15, 22, 29]
    for idx in thick_line_indices:
        ax.axhline(y=idx, xmin=0, xmax=idx/subset_df.shape[0], color='black', linewidth=4)
        ax.axvline(x=idx, ymin=1 - (idx/subset_df.shape[0]), ymax=1 - subset_df.shape[0], color='black', linewidth=4)
            

    # Adjust tick labels
    tick_labels = subset_df.index
    ax.set_xticks(np.arange(len(tick_labels)) + 1)
    ax.set_yticks(np.arange(len(tick_labels)) + 0.5)
    ax.set_xticklabels(tick_labels, rotation=90, ha="right", fontsize=12)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=12)

    # Add top-right label
    ax.text(0.95, 0.95, label_text, transform=ax.transAxes, fontsize=18, fontweight='bold', ha='right', va='top')
    
    # Customize color bar (create it separately for each subplot)
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label('Spectral Angle [rad]', fontsize=18)
    cbar.ax.tick_params(labelsize=12)

# Function to extract rows and columns for a given condition
def extract_subset(spectralangle, condition):
    # Ensure all conditions are present in the spectralangle index
    valid_conditions = [c for c in condition if c in spectralangle.index]
    subset = spectralangle.loc[valid_conditions, valid_conditions]
    return subset

fig, axs = plt.subplots(3, 2, figsize=(20, 23), dpi=300)
for i, s in enumerate(soil[1:]):
    
    sample_order = [f'{s}__0'] # Initialize with pure substrate
    for p in plastic[1:-1]:
        sample_order.extend([f"{s}_{p}_{c}" for c in pconc[1:-1]]) # For each plastic, put in increasing concentration
        sample_order.append(f'_{p}_100') # Add pure plastic after 50% spectra
    
    subset = extract_subset(mean_specang, sample_order)
    create_heatmap_subplot(subset, axs[i // 2, i % 2], label_text=f"{chr(97 + i)}) {s}")
    
plt.tight_layout()
plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plastic index computed on spectral library
def plastic_index_heatmap_subplot(s_pindex, ax, label_text):
    # Plot the heatmap with custom colormap, colorbar, and masking
    sns.heatmap(s_pindex.astype(float), cmap=sns.color_palette("Reds", as_cmap=True), fmt=".2f", 
                cbar=False, vmin=0, vmax=0.65, linecolor='black', linewidth=1, ax=ax)

    # Add thicker lines to separate "blocks" within the heatmap
    for i in range(0, len(plastic)): # Add horizontal lines betwen each cluster of similar plastic
        ax.axhline(i * 7 + 1, color='black', linewidth=4)
    for i in range(1, 5): # Add vertical lines between each plastic index
        ax.axvline(i, color='black', linewidth=4)

    # Add labels and title formatting for the subplot
    ax.text(0, 1.035, label_text, transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')

    # Customize color bar separately for each subplot
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label('Plastic Index Score', fontsize=18)
    cbar.ax.tick_params(labelsize=12)


fig, axs = plt.subplots(2, 3, figsize=(18, 16), dpi=300)

for i, s in enumerate(soil[1:]):
    # Get the specific plastic index data for the current substrate
    sample_order = [f'{s}__0']  # Initialize with pure substrate
    for p in plastic[1:-1]:
        # Add all concentrations for the current plastic, except 0 and 100%
        sample_order.extend([f"{s}_{p}_{c}" for c in pconc[1:-1]])
        # Append the 100% plastic after the 50% concentration
        sample_order.append(f'_{p}_100')
    

    
    s_pindex = pindex.loc[sample_order]
    
    # Map each heatmap to the correct 2x3 position
    plastic_index_heatmap_subplot(s_pindex, axs[i // 3, i % 3], label_text=f"{chr(97 + i)}) {s}")
    axs[i // 3, i % 3].tick_params(axis='x', which='both', labelsize=12)
    axs[i // 3, i % 3].tick_params(axis='y', which='both', labelsize=12)

fig.text(0.5, -0.02, 'Plastic Index', ha='center', fontsize=24)
fig.text(-0.02, 0.5, 'Substrate-Plastic-Concentration', va='center', rotation='vertical', fontsize=24)

plt.tight_layout()
plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Spectral profiles of plastic structures in scene
spectral_data = []
for file_path in glob.glob(os.path.join(r"C:\Users\zachk\Desktop\RS\AVIRIS\NG", "*.txt")):
    wavelengths = []
    reflectances = []
    
    # Open and parse the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Skip header lines and extract data
        for line in lines[3:]:  # Skipping first 3 header lines
            parts = line.split()
            if len(parts) == 2:  # Ensure it has two columns of data
                try:
                    wavelength = float(parts[0])
                    reflectance = float(parts[1])
                    if wavelength >= 850: # only save if above 850
                        wavelengths.append(wavelength)
                        reflectances.append(reflectance)
                except ValueError:
                    pass  # Skip lines that can't be parsed as float
                    
    spectral_data.append((file_path, wavelengths, reflectances))

# Plotting
num_plots = len(spectral_data)
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True)
cs = [plt.get_cmap('icefire_r')(x) for x in np.linspace(.1, .9, num_plots)]

# If only one subplot, make sure axes is iterable
if num_plots == 1:
    axes = [axes]

names = ['Office Roof', 'Solar Panel', 'Tarp 1', 'Tarp 2', 'Tarp 3']
for i, (file_path, wavelengths, reflectances) in enumerate(spectral_data):
    axes[i].plot(wavelengths, reflectances, label=f"{chr(97 + i)}) {names[i]}", color = cs[i], linewidth = 4)
    axes[i].set_ylabel("Reflectance", fontsize = 18)
    axes[i].legend(loc="lower left", fontsize = 18)
    axes[i].grid(True)
    axes[i].xaxis.set_minor_locator(plt.MultipleLocator(100))
    axes[i].grid(True, which='minor', axis='x', linestyle='-', color='lightgray', linewidth=2)
    axes[i].grid(True, which='major', axis='x', linestyle='-', color='lightgray', linewidth=3)


# Set the x-axis label only on the last subplot
axes[-1].set_xlabel("Wavelength (nm)", fontsize = 18)

# Display the plot
plt.tight_layout()
plt.show()