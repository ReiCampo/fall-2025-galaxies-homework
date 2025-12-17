
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
lowz_mask = drpall["NSA_Z"] < 0.35
lowz_gals = drpall[lowz_mask]

good_quality = (lowz_gals['MANGA_DAPQUAL'] == 0) if 'MANGA_DAPQUAL' in drpall.dtype.names else np.ones(len(lowz_gals), dtype=bool)

quality_sample = lowz_gals[good_quality]

# Randomly select 100 galaxies
np.random.seed(50)  # For reproducibility
random_indices = np.random.choice(len(quality_sample), size=min(300, len(quality_sample)), replace=False)
final_sample = quality_sample["PLATEIFU"][random_indices]

# for i in final_sample:
#     plate, ifu = i.split("-")
#     url = f"rsync://dtn.sdss.org/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
#     subprocess.run(["rsync", "-avz", url, data_directory])


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
        
        # Selecting needed columns for analysis
        halpha = spectra["EMLINE_GFLUX"].data[18]
        hbeta = spectra["EMLINE_GFLUX"].data[11]
        nii = spectra["EMLINE_GFLUX"].data[19]
        oiii = spectra["EMLINE_GFLUX"].data[13]
        ra = spectra[0].header["OBJRA"]
        dec = spectra[0].header["OBJDEC"]
        x = spectra["SPX_SKYCOO"].data[0]
        y = spectra["SPX_SKYCOO"].data[1]
        
        # Going to filter out any bad spaxels since I only want positive fluxes:
        valid_mask = (halpha > 0) & (hbeta > 0) & (nii > 0) & (oiii > 0)
        
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
                (df['oiii_flux'] > 0)]
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print(f"\nAfter inf replacement - NaN counts:\n{df.isna().sum()}")
        
        
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

    def kauffmann_line(x):
        return 0.61 / (x - 0.05) + 1.3
    
    ##---------------------------------------------------------------
    ##                    Handling Kewley Line:                     -
    ##---------------------------------------------------------------

    def kewley_line(x):
        return 0.61 / (x - 0.47) + 1.19
    
    
    ##----------------------------------------------------------------
    ##                  Now Classifying Each Spaxel:                 -
    ##----------------------------------------------------------------

    if log_nii_ha < 0.05:
        if log_oiii_hb < kauffmann_line(log_nii_ha):
            return 0
        else:
            return 1
    else:
        if log_oiii_hb < kauffmann_line(log_nii_ha):
            return 0
        elif log_oiii_hb < kewley_line(log_nii_ha):
            return 1
        else:
            return 2
    
    return -1

# Now I will create a function that will classify if the galaxy is an active AGN
# or a non-active AGN with the nuclear spectra

def classify_if_agn(galaxy_df, agn_radius_kpc):
    '''
    This function will classify if a galaxy has an active AGN at the center of
    it. This function will use the physical distance of the nucleus of the
    galaxy. This is to take into account for redshift, making the overall
    calculation more robuts.
    
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

print(f"\nFinal dataframe with all classifications:")
print(clean_data.head(10))
print(f"\nColumns in dataframe:")
print(clean_data.columns.tolist())

print(f"Unique galaxies: {clean_data['plateifu'].nunique()}")
print(f"Unique plateifu values: {clean_data['plateifu'].unique()}")
print(f"Missing redshift: {clean_data['redshift'].isna().sum()}")





print("="*60)
print("DEBUGGING GALAXY CLASSIFICATION")
print("="*60)

# 1. Check overall BPT distribution
print("\n1. Overall BPT classification:")
print(clean_data['bpt_classification'].value_counts())

# 2. Check if ANY AGN spaxels exist
n_agn = (clean_data['bpt_classification'] == 2).sum()
print(f"\n2. Total AGN spaxels: {n_agn}")

###########################################################################
###########################################################################
###                                                                     ###
###                              PLOTTING                               ###
###                                                                     ###
###########################################################################
###########################################################################


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