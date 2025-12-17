
############################################################################
############################################################################
###                                                                      ###
###                   GALAXIES FALL 2025 FINAL PROJECT                   ###
###                                                                      ###
############################################################################
############################################################################


##---------------------------------------------------------------
##                Importing Necessary Packages:                 -
##---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import cm
import os
import subprocess
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

##----------------------------------------------------------------
##      Determining Which MaNGA DAP MAPS I Should Download       -
##----------------------------------------------------------------

data_directory = "/Users/RachelCampo/Desktop/CUNY Classes/Fall 2025 Galaxies/Homework/Data 300"
dpr_summary_file = os.path.join(data_directory, "drpall-v3_1_1.fits")

drpall = fits.open(dpr_summary_file)[1].data

# Finding the galaxies that sit in the redshift range I'm interested in
# Choosing this redshift range from Kai-Xing Liu et al. (2018):
# lowz_mask = drpall["NSA_Z"] < 0.35
# lowz_gals = drpall[lowz_mask]

# good_quality = (lowz_gals['MANGA_DAPQUAL'] == 0) if 'MANGA_DAPQUAL' in drpall.dtype.names else np.ones(len(lowz_gals), dtype=bool)

# quality_sample = lowz_gals[good_quality]

# # Randomly select 100 galaxies
# np.random.seed(50)  # For reproducibility
# random_indices = np.random.choice(len(quality_sample), size=min(300, len(quality_sample)), replace=False)
# final_sample = quality_sample["PLATEIFU"][random_indices]

# for i in final_sample:
#     plate, ifu = i.split("-")
#     url = f"rsync://dtn.sdss.org/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
#     subprocess.run(["rsync", "-avz", url, data_directory])

# test_spectra = fits.open(data_directory + "/manga-7992-9102-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz")
# test_spectra.info()

###########################################################################
###########################################################################
###                                                                     ###
###                            CLEANING DATA                            ###
###                                                                     ###
###########################################################################
###########################################################################


##---------------------------------------------------------------
##              Creating Functions To Clean Spectra             -
##---------------------------------------------------------------

def dust_extinction(h_alpha, h_beta, RA, DEC):
    """
    This function will handle dust corrections from both the Milky Way and the
    the spectra's own internal dust. 
    
    For the Milky Way dust: The function will use the Schlegel, Finkbeiner & 
    Davis 1998 (SFD98) Dust Maps to deredden locally.
    
    For the spectra's internal dust: This function will use Balmer Decrementing
    since that is an acceptable method for dereddening the galaxy (as seen in
    Kai-Xing et al. 2018).
    
    Inputs:
        h_alpha (array):
            The h_alpha emission line flux from a given spectra.
        
        b_beta (array):
            The h_beta emission line flux from a given spectra.
            
        RA (float):
            The right ascension of the spectra in degrees
        
        DEC (float):
            The declination of the spectra in degrees
       
    Outpus:
        full_dereddened_flux ():
            The dereddened flux from both the Milky Way and the interal spectra
            dust
        
        galactic_flux ():
            The flux that was dereddened with only the Milky Way dust
        
        internal_flux ():
            The flux that was dereddened with only the internal spectra dust
    
    """
    
    
    ##----------------------------------------------------------------
    ##                  Correct for Milky Way dust                   -
    ##----------------------------------------------------------------

    from dustmaps.sfd import SFDQuery
    from astropy.coordinates import SkyCoord
    
    # These are the extinction curve values from the CCM89 
    k_h_alpha = 2.53
    k_h_beta = 3.61
    
    sfd = SFDQuery()
    coordinates = SkyCoord(ra = RA,
                           dec = DEC,
                           unit = "deg",
                           frame = 'icrs')
    
    galactic_extinction = sfd(coordinates)
    
    adjusted_alpha_lambda = k_h_alpha * galactic_extinction
    adjusted_beta_lambda = k_h_beta * galactic_extinction
    
    # This applies a correction term (0.86) for low redshift galaxies. This
    # correction is found in Schlafly & Finkbeiner (2011)
    galactic_corrected = 0.86 * galactic_extinction
    adjusted_alpha_lambda = k_h_alpha * galactic_corrected
    adjusted_beta_lambda = k_h_beta * galactic_corrected
    
    galactic_alpha_flux = h_alpha * 10**(0.4 * adjusted_alpha_lambda)
    galactic_beta_flux = h_beta * 10**(0.4 * adjusted_beta_lambda)
    
    ##----------------------------------------------------------------
    ##            Now correcting for internal spectra dust           -
    ##----------------------------------------------------------------

    with np.errstate(divide = "ignore",
                     invalid = "ignore"):
        observed_ratio = galactic_alpha_flux / galactic_beta_flux
        
        # Balmer intrinsic ration = 2.86
        internal_flux = (2.5 / (k_h_alpha - k_h_beta)) * np.log10(observed_ratio / 2.86)
    
    # This will handle the case when the extinction is negative:
    internal_flux = np.where(internal_flux < 0, 0, internal_flux)
    
    ##----------------------------------------------------------------
    ##                    Correcting entire flux                     -
    ##----------------------------------------------------------------

    full_dereddened_flux = galactic_alpha_flux * 10**(0.4 * k_h_alpha * internal_flux)
    
    return full_dereddened_flux, galactic_corrected, internal_flux


# First, I'm going to create a list of all the files I want to store later
# into a pandas dataframe:
fits_files = [f for f in os.listdir(data_directory) if f.endswith(".gz")]

# Finding redhisft values:
redshift_lookup = dict(zip(drpall['PLATEIFU'], drpall['NSA_Z']))

# Now selecting columns I want to use later for analysis and plotting:

clean_fits = []

for file in fits_files:
    
    single_data = os.path.join(data_directory, file)
    
       
    with fits.open(single_data) as spectra:
        
        plateifu = spectra[0].header["PLATEIFU"]
        
        # Get redshift from DRPall
        redshift = redshift_lookup[plateifu]  
        
        # Selecting fluxes needed for calculation:
        halpha = spectra["EMLINE_GFLUX"].data[18]
        hbeta = spectra["EMLINE_GFLUX"].data[11]
        nii = spectra["EMLINE_GFLUX"].data[19]
        oiii = spectra["EMLINE_GFLUX"].data[13]
        
        halpha_ivar = spectra["EMLINE_GFLUX_IVAR"].data[18]
        hbeta_ivar = spectra["EMLINE_GFLUX_IVAR"].data[11]
        nii_ivar = spectra["EMLINE_GFLUX_IVAR"].data[19]
        oiii_ivar = spectra["EMLINE_GFLUX_IVAR"].data[13]
        
        # Calculate S/N for each line
        with np.errstate(divide='ignore', invalid='ignore'):
            halpha_snr = halpha * np.sqrt(halpha_ivar)
            hbeta_snr = hbeta * np.sqrt(hbeta_ivar)
            nii_snr = nii * np.sqrt(nii_ivar)
            oiii_snr = oiii * np.sqrt(oiii_ivar)
                       
        ra = spectra[0].header["OBJRA"]
        dec = spectra[0].header["OBJDEC"]
        x = spectra["SPX_SKYCOO"].data[0]
        y = spectra["SPX_SKYCOO"].data[1]
 
        # I am also going to create a mask that filters signal_to_noise ratios 
        # greater than 5 and filters out any bad spaxels that have negative flux
        # values:
        
        snr_threshold = 2.0
        
        valid_mask = (
            (halpha > 0) & (hbeta > 0) & (nii > 0) & (oiii > 0) &  # Positive fluxes
            (halpha_snr > snr_threshold) &  # Good S/N
            (hbeta_snr > snr_threshold) &
            (nii_snr > snr_threshold) &
            (oiii_snr > snr_threshold) &
            (halpha_ivar > 0) & (hbeta_ivar > 0) &  # Valid inverse variances
            (nii_ivar > 0) & (oiii_ivar > 0)            
        )
        
        # Because I am getting division by zero, I'm going to add in this line
        # to handle those problems. I will create an array filled with NaNs, 
        # then fill that array where all my fluxes are positive. Negative values
        # will remain as NaNs
        log_nii_halpha = np.full(halpha.shape, np.nan)
        log_oiii_hbeta = np.full(halpha.shape, np.nan)
        
        log_nii_halpha[valid_mask] = np.log10(nii[valid_mask] / halpha[valid_mask])
        log_oiii_hbeta[valid_mask] = np.log10(oiii[valid_mask] / hbeta[valid_mask])

        
        # Now deredden the spectra:
        dereddened_flux_vals, galactic_flux, internal_flux = dust_extinction(h_alpha = halpha,
                                                                             h_beta = hbeta,
                                                                             RA = ra,
                                                                             DEC = dec)
        
        # Take all my data, flatten it, then put it into a pandas dataframe for
        # easier analysis and plotting
        df = pd.DataFrame({
            "plateifu": plateifu,
            "redshift": redshift,
            "x": x.flatten().astype(float),
            "y": y.flatten().astype(float),
            "halpha_flux": halpha.flatten().astype(float),
            "hbeta_flux": hbeta.flatten().astype(float),
            "nii_flux": nii.flatten().astype(float),
            "oiii_flux": oiii.flatten().astype(float),
            "halpha_snr": halpha_snr.flatten().astype(float), 
            "hbeta_snr": hbeta_snr.flatten().astype(float), 
            "nii_snr": nii_snr.flatten().astype(float),       
            "oiii_snr": oiii_snr.flatten().astype(float),    
            "log_nii_halpha": log_nii_halpha.flatten().astype(float),
            "log_oiii_hbeta": log_oiii_hbeta.flatten().astype(float),
            "full_dereddened_flux" : dereddened_flux_vals.flatten().astype(float)})
        
        print(f"\nBefore filtering - columns: {df.columns.tolist()}")
        print(f"Before filtering - shape: {df.shape}")
        print(f"Redshift value: {redshift}")
        print(f"Sample of redshift column:\n{df['redshift'].head()}")
        print(f"NaN counts per column:\n{df.isna().sum()}")

        # If infinities exist (like if we divided by zero from earlier), turn
        # those values into a NAN. Then, drop all NA values and append the
        # cleaned data to the list we created earlier
        df = df[(df['halpha_flux'] > 0) & 
                (df['hbeta_flux'] > 0) & 
                (df['nii_flux'] > 0) & 
                (df['oiii_flux'] > 0) &
                (df['halpha_snr'] > snr_threshold) & 
                (df['hbeta_snr'] > snr_threshold) & 
                (df['nii_snr'] > snr_threshold) &  
                (df['oiii_snr'] > snr_threshold)]
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['halpha_flux', 'hbeta_flux', 'nii_flux', 'oiii_flux'])
        
        print(f"After dropna - shape: {df.shape}")
        print(f"After dropna - redshift column:\n{df['redshift'].head()}")
        
        
        clean_fits.append(df)

clean_data = pd.concat(clean_fits, ignore_index = True)


print(f"Total valid spaxels: {len(clean_data)}")
print(f"Sample of data:\n{clean_data.head()}")
print(f"\nColumns in dataframe:")
print(clean_data.columns.tolist())


############################################################################
############################################################################
###                                                                      ###
###                               ANALYSIS                               ###
###                                                                      ###
############################################################################
############################################################################


##------------------------------------------------------------------------------
##  I am now going to classify each galaxy to see if any of them have active   -
##  AGN                                                                        -
##------------------------------------------------------------------------------

# In order to classify my selected galaxies at each spaxel, I will have to 
# calculate the ionization of NII/Halpha, OIII/HBeta. The first and second 
# calculation are the Kauffmann and Kewley lines, respectively. Reference source: 
# https://sites.google.com/site/agndiagnostics/agn-optical-line-diagnostics/bpt-diagrams

def classify_bpt(log_nii_ha, log_oiii_hb):
    '''
    This function classifies each spaxel depending if that particular spaxel is
    star forming, composite, AGN, or unclassified.
    
    Inputs:
        log_nii_ha (float):
            The log ratio of NII / H Alpha at a given spaxel
            
        log_oiii_hb (float):
            The log ratio of OII / H Beta at a given spaxel
            
    Outputs:
        -1: Classifies that spaxel as Unclassified.
        0: Classifies that spaxel as Star Forming.
        1: Classifies that spaxel as Composite
        2: Classifies that spaxel as AGN
        
    '''


    ##---------------------------------------------------------------
    ##              Starting with Unclassified Spaxels:             -
    ##---------------------------------------------------------------
    
    if np.isnan(log_nii_ha) or np.isnan(log_oiii_hb):
        return -1
    
    ##----------------------------------------------------------------
    ##                    Handling Kauffmann Line:                   -
    ##----------------------------------------------------------------
    
    # Valid for log([NII]/Hα) < 0.05
    if log_nii_ha < 0.05:
        kauff_y = 0.61 / (log_nii_ha - 0.05) + 1.3
    else:
        kauff_y = -np.inf  
    
    ##---------------------------------------------------------------
    ##                    Handling Kewley Line:                     -
    ##---------------------------------------------------------------

    # This determines if the galaxy is Composite or AGN:
    if log_nii_ha < 0.47:
        kewley_y = 0.61 / (log_nii_ha - 0.47) + 1.19
    else:
        kewley_y = -np.inf
    
    
    ##----------------------------------------------------------------
    ##                  Now Classifying Each Spaxel:                 -
    ##----------------------------------------------------------------

    if log_oiii_hb < kauff_y:
        return 0  # Star-forming
    elif log_oiii_hb < kewley_y:
        return 1  # Composite
    else:
        return 2  # AGN  
        

# Now I will create a function that will classify if the galaxy is an active AGN
# or a non-active AGN with the nuclear spectra

def classify_if_agn(galaxy_df, agn_radius_kpc):
    '''
    This function will classify if a galaxy has an active AGN at the center of
    it. This function will use the physical distance of the nucleus of the
    galaxy. This is to take into account for redshift, making the overall
    calculation more robust.
    
    Inputs:
        galaxy_df (Pandas dataframe):
            The dataframe to be passed into the function.
            
        agn_radius_kpc (float):
            The radius in kiloparsecs at which the fucntion should test to see 
            if the galaxy has an active AGN within it. AGN radii can vary from 
            100 parsecs to 2 kiloparsecs from the central black hole. It really 
            is dependent on how narrow or broad you want to be.
            
    Outputs:
        Unclassified: This means that the nuclear spaxels were not classified
            from BPT calculations as either 2, 1, or 0. 
        
        AGN: This means that the BPT calculation of the nuclear spaxels were 2
        
        Composite: This means that the BPT calculation of the nuclear spaxels
            were 1
        
        Star Forming: This means that the BPT calculation of the nuclear spaxels
            were 0
    '''
    
    redshift = galaxy_df["redshift"].iloc[0]
    
    # Convert kiloparsecs to arcseconds at the given redshift:
    kpc_to_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc/u.arcsec).value
    agn_radius_arcsec = agn_radius_kpc / kpc_to_arcsec
    
    # Calculate the distance from the center:
    galaxy_df = galaxy_df.copy()
    galaxy_df['nuclear_radius'] = np.sqrt(galaxy_df['x']**2 + galaxy_df['y']**2)
    
    nuclear_spaxels = galaxy_df[galaxy_df["nuclear_radius"] <= agn_radius_arcsec]
    
    ##----------------------------------------------------------------
    ##                  Now Classifying Each Galaxy:                 -
    ##----------------------------------------------------------------

    if len(nuclear_spaxels) == 0:
        return "Unclassified"
    
    bpt_counts = nuclear_spaxels["bpt_classification"].value_counts()
    
    if 2 in bpt_counts.index:
        return 'AGN'
    elif 1 in bpt_counts.index:
        return "Composite"
    elif 0 in bpt_counts.index:
        return "Star Forming"
    else:
        return "Unclassified"
    
# Now I will create a function that will calculate the star formation rate by
# calculating the luminosity. I am able to calculate these rates thanks to
# the Kennicutt 1998 paper. The equation used in this function is found on page
# 7.

def calculate_sfr(halpha_flux_dereddened, redshift):
    '''
    This function calculates star formation rates using the equation found in
    Kennicutt (1998).
    
    Inputs:
        halpha_flux_dereddened (float):
            The dereddened flux value at a particular spaxel.
            
        redshift (float):
            The redshift value at a given spaxel.
            
    Outputs:
        SFR (float):
            The star formation rate at a given spaxel.
    '''

    luminosity_distance = cosmo.luminosity_distance(redshift).to(u.cm).value
    halpha_luminosity = halpha_flux_dereddened * 4 * np.pi * luminosity_distance**2
    SFR = 7.9e-42 * halpha_luminosity
    return SFR

# Now I will create a function that calculates SFR densities. This will be used
# to look at how the density changes over radius:

def sfr_density(sfr, z, spaxel_size_arcsec = 0.5):
    """
    This function calculates the star formation rate surface density at each
    spaxel. The size of each spaxel in MaNGA is 0.5 x 0.5 arcseconds.
    
    Inputs:
        sfr (float):
            The star formation rate at a given spaxel.
            
        z (float):
            The redshift value of the spaxel.
            
        spaxel_size_arcsec (float):
            The length of each spaxel. Hardcoded to 0.5
            
    Outputs:
        sfr / spaxel_area_kpc2 (float):
            The star formation rate per spaxel area.
    """
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
    spaxel_size_kpc = spaxel_size_arcsec * kpc_per_arcsec
    spaxel_area_kpc2 = spaxel_size_kpc**2
    return sfr / spaxel_area_kpc2

# To account for the fact that we have varying radii between each galaxy, I am
# going to create a function that calculates the effective radius. This will be
# used later for plotting:

def normalize_radii(data):
    """
    This function normalizes by the radius of each galaxy's maximum detected 
    radius.
    
    Inputs:
        data (PandasData Frame):
            Takes in the data frame that will be used to calculate the
            effective radius. 
            
    Outpus:
        A pandas dataframe with an additional column that calculated the 
        normalized radius.
    """
    normalized_data = []
    
    for plateifu in data['plateifu'].unique():
        galaxy_data = data[data['plateifu'] == plateifu].copy()
        
        # Get maximum radius for this galaxy
        r_max = galaxy_data['r_kpc'].max()
        
        # Normalize: r_norm = r / r_max
        galaxy_data['r_normalized'] = galaxy_data['r_kpc'] / r_max
        
        normalized_data.append(galaxy_data)
    
    return pd.concat(normalized_data, ignore_index=True)


##----------------------------------------------------------------
##                    Adding BPT Calculations:                   -
##----------------------------------------------------------------


clean_data['bpt_classification'] = clean_data.apply(
    lambda row: classify_bpt(row['log_nii_halpha'], row['log_oiii_hbeta']),
    axis = 1
)


##---------------------------------------------------------------
##            Adding Star Formation Rate Information:           -
##---------------------------------------------------------------


clean_data["SFR"] = clean_data.apply(
    lambda row: calculate_sfr(row["full_dereddened_flux"], row["redshift"]),
    axis = 1
)


##----------------------------------------------------------------
##                Adding in SFR Surface Density:                 -
##----------------------------------------------------------------


clean_data["sigma_SFR"] = clean_data.apply(
    lambda row: sfr_density(row["SFR"], row["redshift"]),
    axis = 1
)


##---------------------------------------------------------------
##                Calculating Radial Distances:                 -
##---------------------------------------------------------------


clean_data["r_arcsec"] = np.sqrt(clean_data["x"]**2 + clean_data["y"]**2)

def arcseconds_to_kpc(r_arcsec, redshift):
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec).value
    return kpc_per_arcsec * r_arcsec

clean_data["r_kpc"] = clean_data.apply(
    lambda row: arcseconds_to_kpc(row["r_arcsec"], row["redshift"]),
    axis = 1)


##---------------------------------------------------------------
##                Adding Galaxy Classification:                 -
##---------------------------------------------------------------

galaxy_classifications = {}

for plateifu in clean_data["plateifu"].unique():
    galaxy_data = clean_data[clean_data["plateifu"] == plateifu]
    
    # I am going to start with using agn_radius_kpc = 3.0 since these galaxies
    # are relatively close by. Using the standard radius of 1.5 may be too small
    # to properly classify the galaxy:
    galaxy_classifications[plateifu] = classify_if_agn(galaxy_data, agn_radius_kpc = 1.5)
    
clean_data["galaxy_classification"] = clean_data["plateifu"].map(galaxy_classifications)

##---------------------------------------------------------------
##                  Adding in Normalized Radii:                 -
##---------------------------------------------------------------


clean_data = normalize_radii(clean_data)


##----------------------------------------------------------------
##                  Printing Summary Statistics:                 -
##----------------------------------------------------------------

print(f"\nFinal dataframe with all classifications:")
print(clean_data.head(10))
print(f"\nColumns in dataframe:")
print(clean_data.columns.tolist())

print(f"Unique galaxies: {clean_data['plateifu'].nunique()}")
print(f"Unique plateifu values: {clean_data['plateifu'].unique()}")


###########################################################################
###########################################################################
###                                                                     ###
###                              PLOTTING                               ###
###                                                                     ###
###########################################################################
###########################################################################

# Make sure to put in a note about creating plotting functions!!!
def bin_by_radius(data, radial_bins, radius_type = "r_kpc"):
    """
    This function bins each spaxel by radius and then calculates the mean and
    median star formation rate surface density in each radial bin. Can take in 
    either physical or normalized radii.
    
    Inputs:
        data (PandasDataframe):
            The data frame that contains star formation rates to be passed to
            the function.
            
        radial_bins (NumpyArray):
            A range of radial bins that will be used to calculate bin centers.
            
        radius_type (str):
            The name of the column that holds which radius you want to use,
            either physical or normalized.
            
    Outputs:
        bin_centers (array):
            Returns an array of bin center radii
        
        median_sfr (NumpyArray):
            The median star formation rate surface density per bin
            
        mean_sfr (NumpyArray):
            The mean star formation rate surface density per bin
            
        std_sfr (NumpyArray):
            The standard deviation per bin.
            
        n_spaxels (NumpyArray):
            The number of spaxels per bin.
    """
    
    bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
    
    median_sfr = []
    mean_sfr = []
    std_sfr = []
    n_spaxels = []
    
    for i in range(len(radial_bins) - 1):
        r_min = radial_bins[i]
        r_max = radial_bins[i + 1]
        
        # Select spaxels in this radial bin
        in_bin = data[(data[radius_type] >= r_min) & (data[radius_type] < r_max)]
        
        if len(in_bin) > 0:
            median_sfr.append(in_bin['sigma_SFR'].median())
            mean_sfr.append(in_bin['sigma_SFR'].mean())
            std_sfr.append(in_bin['sigma_SFR'].std())
            n_spaxels.append(len(in_bin))
        else:
            median_sfr.append(np.nan)
            mean_sfr.append(np.nan)
            std_sfr.append(np.nan)
            n_spaxels.append(0)  
            
    return bin_centers, np.array(median_sfr), np.array(mean_sfr), np.array(std_sfr), np.array(n_spaxels)

# Now, I am filtering down to the data I want:
composite_data = clean_data[clean_data['galaxy_classification'] == 'Composite']
sf_data = clean_data[clean_data['galaxy_classification'] == 'Star Forming']

##----------------------------------------------------------------
##            Calculating the SFR Density per Spaxel             -
##----------------------------------------------------------------

# First calculating radial bins and their centers:
radial_bins = np.arange(0, 10, 0.5)  # 0 to 10 kpc in 0.5 kpc bins
bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

# Now I will bin both star forming and composite galaxies:

comp_bins, comp_median, comp_mean, comp_std, comp_n = bin_by_radius(composite_data, radial_bins)
sf_bins, sf_median, sf_mean, sf_std, sf_n = bin_by_radius(sf_data, radial_bins)

# Now creating the plot:

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Craeting the top panel to be the radial profiles:Top Panel:

# Plot composite galaxies
ax1.plot(comp_bins, comp_median, 
         marker='o', linewidth=2, markersize=8,
         label=f'Composite (n={composite_data["plateifu"].nunique()} galaxies)',
         color='orange', alpha=0.8)

# Add shaded region for standard deviation
ax1.fill_between(comp_bins, 
                  comp_median - comp_std/np.sqrt(comp_n), 
                  comp_median + comp_std/np.sqrt(comp_n),
                  alpha=0.3, color='orange')

# Plot star-forming galaxies
ax1.plot(sf_bins, sf_median,
         marker='s', linewidth=2, markersize=8,
         label=f'Star-forming (n={sf_data["plateifu"].nunique()} galaxies)',
         color='blue', alpha=0.8)

# Add shaded region
ax1.fill_between(sf_bins,
                  sf_median - sf_std/np.sqrt(sf_n),
                  sf_median + sf_std/np.sqrt(sf_n),
                  alpha=0.3, color='blue')

ax1.set_ylabel('SFR Surface Density', fontsize=14)
ax1.set_title('Radial SFR Surface Density Profiles', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_yscale('log')  # Log scale often better for SFR surface density

##--- Bottom Panel: Number of Spaxels ---##

ax2.plot(comp_bins, comp_n, marker='o', linewidth=2, 
         label='Composite', color='orange', alpha=0.8)
ax2.plot(sf_bins, sf_n, marker='s', linewidth=2,
         label='Star-forming', color='blue', alpha=0.8)

ax2.set_xlabel('Radius (kpc)', fontsize=14)
ax2.set_ylabel('Number of Spaxels', fontsize=14)
ax2.set_title('Sample Size per Radial Bin', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('radial_sfr_profile_composite_vs_sf.png', dpi=300, bbox_inches='tight')
plt.show()


##-------------------------------------------------------------------
##  Calculating the SFR Density Per Spaxel with Normalized Radii:   -
##-------------------------------------------------------------------

radial_bins_norm = np.linspace(0, 1, 11)

# Bin the data using normalized radius
comp_bins_norm, comp_median_norm, comp_mean_norm, comp_std_norm, comp_n_norm = bin_by_radius(
    composite_data, radial_bins_norm, radius_type ='r_normalized'
)
sf_bins_norm, sf_median_norm, sf_mean_norm, sf_std_norm, sf_n_norm = bin_by_radius(
    sf_data, radial_bins_norm, radius_type = 'r_normalized'
)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10), sharex = True)

# Top panel: SFR surface density
ax1.plot(comp_bins_norm, comp_median_norm, marker='o', linewidth=2, markersize=8,
         label=f'Composite (n={composite_data["plateifu"].nunique()} galaxies)',
         color='orange', alpha=0.8)

ax1.fill_between(comp_bins_norm, 
                  comp_median_norm - comp_std_norm/np.sqrt(comp_n_norm), 
                  comp_median_norm + comp_std_norm/np.sqrt(comp_n_norm),
                  alpha=0.3, color='orange')

ax1.plot(sf_bins_norm, sf_median_norm, marker='s', linewidth=2, markersize=8,
         label=f'Star-forming (n={sf_data["plateifu"].nunique()} galaxies)',
         color='blue', alpha=0.8)
ax1.fill_between(sf_bins_norm,
                  sf_median_norm - sf_std_norm/np.sqrt(sf_n_norm),
                  sf_median_norm + sf_std_norm/np.sqrt(sf_n_norm),
                  alpha=0.3, color='blue')

ax1.set_ylabel('SFR Surface Density', fontsize=14)
ax1.set_title('Radial SFR Surface Density Profiles (Normalized Radius)', 
              fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_yscale('log')

# Bottom panel: Number of spaxels
ax2.plot(comp_bins_norm, comp_n_norm, marker='o', linewidth=2, 
         label='Composite', color='orange', alpha=0.8)
ax2.plot(sf_bins_norm, sf_n_norm, marker='s', linewidth=2,
         label='Star-forming', color='blue', alpha=0.8)

ax2.set_xlabel('Normalized Radius', fontsize=14)
ax2.set_ylabel('Number of Spaxels', fontsize=14)
ax2.set_title('Sample Size per Radial Bin', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('radial_sfr_profile_normalized.png', dpi=300, bbox_inches='tight')
plt.show()


##---------------------------------------------------------------
##                Plotting BPT Diagram By Radius:               -
##---------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Kauffmann and Kewley lines
x_line = np.linspace(-2.0, 0.5, 1000)
kauffmann_y = 0.61 / (x_line - 0.05) + 1.3
kewley_y = 0.61 / (x_line - 0.47) + 1.19

for ax, data, title in zip(axes, 
                            [composite_data, sf_data],
                            ['Composite Galaxies', 'Star-Forming Galaxies']):
    
    # Plot spaxels colored by radius
    scatter = ax.scatter(data['log_nii_halpha'], 
                         data['log_oiii_hbeta'],
                         c=data['r_kpc'],  # Color by radius
                         cmap='viridis',
                         alpha=0.5,
                         s=20,
                         vmin=0, vmax=8)
    
    # Plot division lines
    mask = (x_line > -1.5) & (x_line < 0.05)
    ax.plot(x_line[mask], kauffmann_y[mask], 'k--', linewidth=2, 
            label='Kauffmann (2003)')
    
    mask = (x_line > -0.3) & (x_line < 0.47)
    ax.plot(x_line[mask], kewley_y[mask], 'k-', linewidth=2,
            label='Kewley (2001)')
    
    ax.set_xlabel('log([NII]/Hα)', fontsize=14)
    ax.set_ylabel('log([OIII]/Hβ)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(-1.0, 1.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Radius (kpc)', fontsize=12)

plt.tight_layout()
plt.savefig('bpt_diagram_by_radius.png', dpi=300, bbox_inches='tight')
plt.show()

##---------------------------------------------------------------
##            Plot: Total SFR Distribution                      -
##---------------------------------------------------------------

# Calculate total SFR per galaxy
comp_total_sfr = composite_data.groupby('plateifu')['SFR'].sum()
sf_total_sfr = sf_data.groupby('plateifu')['SFR'].sum()

fig, ax = plt.subplots(figsize=(10, 7))

# Histograms
ax.hist(np.log10(comp_total_sfr), bins=15, alpha=0.6, 
        label=f'Composite (n={len(comp_total_sfr)})', 
        color='orange', edgecolor='black')
ax.hist(np.log10(sf_total_sfr), bins=15, alpha=0.6,
        label=f'Star-Forming (n={len(sf_total_sfr)})',
        color='blue', edgecolor='black')

# Add median lines
ax.axvline(np.log10(comp_total_sfr.median()), 
           color='orange', linestyle='--', linewidth=2,
           label=f'Comp Median: {comp_total_sfr.median():.2f} M☉/yr')
ax.axvline(np.log10(sf_total_sfr.median()),
           color='blue', linestyle='--', linewidth=2,
           label=f'SF Median: {sf_total_sfr.median():.2f} M☉/yr')

ax.set_xlabel('log(Total SFR) [Change in Mass / Year]', fontsize=14)
ax.set_ylabel('Number of Galaxies', fontsize=14)
ax.set_title('Distribution of Total SFR', fontsize=16, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('total_sfr_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical test
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(comp_total_sfr, sf_total_sfr)
print(f"\nMann-Whitney U test:")
print(f"  p-value: {pval:.4f}")
if pval < 0.05:
    print(f"  → Distributions are significantly different!")
else:
    print(f"  → No significant difference")



##---------------------------------------------------------------
##            Plot: SFR vs Galaxy Size                          -
##---------------------------------------------------------------

# Calculate per-galaxy stats
galaxy_stats = clean_data.groupby('plateifu').agg({
    'SFR': 'sum',
    'r_kpc': 'max',
    'galaxy_classification': 'first'
})

comp_gals = galaxy_stats[galaxy_stats['galaxy_classification'] == 'Composite']
sf_gals = galaxy_stats[galaxy_stats['galaxy_classification'] == 'Star Forming']

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(comp_gals['r_kpc'], comp_gals['SFR'],
           s=100, alpha=0.7, color='orange', 
           edgecolors='black', linewidths=0.5,
           label=f'Composite (n={len(comp_gals)})')

ax.scatter(sf_gals['r_kpc'], sf_gals['SFR'],
           s=100, alpha=0.7, color='blue',
           edgecolors='black', linewidths=0.5,
           label=f'Star-Forming (n={len(sf_gals)})')

ax.set_xlabel('Maximum Radius (kpc)', fontsize=14)
ax.set_ylabel('Total SFR (Change of Mass / Year)', fontsize=14)
ax.set_title('SFR vs Galaxy Size', fontsize=16, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sfr_vs_size.png', dpi=300, bbox_inches='tight')
plt.show()












##---------------------------------------------------------------
##            Plotting SFRs vs Galaxy Classification:           -
##---------------------------------------------------------------


###  First, I am going to leverage the power of pandas by grouping by plateifu   
###  and summarizing the sum of star formation rates, finding the maximum        
###  radius of the galaxy, and taking the first value in the galaxy's            
###  classification and redshift since they are all the same value:              

summarize_galaxy_stats = clean_data.groupby("plateifu").agg({"SFR": "sum",
                                                             "r_kpc": "max",
                                                             "galaxy_classification": "first",
                                                            "redshift": "first"}).reset_index()

# Renaming the SFR column to be more intuitive:
summarize_galaxy_stats = summarize_galaxy_stats.rename(columns = {"SFR": "total_SFR"})

# Now adding in diameter information:
summarize_galaxy_stats["diameter_kpc"] = 2 * summarize_galaxy_stats["r_kpc"]

# Now, separate the galaxies by class to use in the plotting logic for later:
agn_gals = summarize_galaxy_stats[summarize_galaxy_stats["galaxy_classification"] == "AGN"]
sf_gals = summarize_galaxy_stats[summarize_galaxy_stats["galaxy_classification"] == "Star Forming"]
composite_gals = summarize_galaxy_stats[summarize_galaxy_stats["galaxy_classification"] == "Composite"]
unclass_gals = summarize_galaxy_stats[summarize_galaxy_stats["galaxy_classification"] == "Unclassified"]

# Now creating the plot:

fig, ax = plt.subplots(figsize = (10, 10))

# Starting by plotting the star forming galaxies:
if len(sf_gals) > 0:
    ax.scatter(sf_gals["diameter_kpc"],
                sf_gals["total_SFR"],
                label = "Star Forming Galaxies, n = " + str(len(sf_gals)),
                alpha = 0.7,
                s = 150,
                c = "blue",
                edgecolors = "black",
                linewidths = 0.5)
    
# Now plotting AGN galaxies:
if len(agn_gals) > 0:
    ax.scatter(agn_gals["diameter_kpc"],
                agn_gals["total_SFR"],
                label = "AGN Galaxies, n = " + str(len(agn_gals)),
                alpha = 0.7,
                s = 150,
                c = "red",
                edgecolors = "black",
                linewidths = 0.5)
    
# Now plotting composite galaxies:
if len(composite_gals) > 0:
    ax.scatter(composite_gals["diameter_kpc"],
                composite_gals["total_SFR"],
                label = "Composite Galaxies, n = " + str(len(composite_gals)),
                alpha = 0.7,
                s = 150,
                c = "yellow",
                edgecolors = "black",
                linewidths = 0.5)
    
ax.set_xlabel("Galaxy Diameter (kpc)", fontsize = 14)
ax.set_ylabel("Total Star Formation Rate Per Year", fontsize = 14)
ax.set_title("Galaxy Size vs. Total Star Formation Rate", fontsize = 16)
ax.legend(fontsize = 12, loc = "best")
ax.grid(True, alpha = 0.3, linestyle = "--")
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()

