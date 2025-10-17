import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict

from sklearn.model_selection import train_test_split
from astropy.stats import biweight_location, biweight_midvariance
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error
import math
import numpy as np


######################
# DATA PREPROCESSING #
######################


def import_photoz_data(path=None, version=None):
    """
    Import the data table of band magnitudes and spectroscopic redshift. Must
    provide a full path to data or the data version. Returns the raw loaded
    dataset as a DataFrame.
    
    path: str
        Full path. Dataset must have the original column names on retrieval from
        the HSC database. Takes precedence over version when both are provided.
    version: int
        Version number. Dataset must have the original column names on retrieval
        from the HSC database. Automatically fills out the pathname for you,
        assuming the data has been named correctly.
    """
    
    if (path is not None):
        pathname = path
    elif (version is not None):
        pathname = '/data/HSC/v' + str(version)
    else:
        sys.exit("Must provide a full path to data or the data version.")
    df = pd.read_csv(pathname)
    return df


def clean_photoz_data(df, errors=False, filters=None, scaled=False):
    """
    Clean the imported dataset. Columns are the band magnitudes, spectroscopic
    redshift, and optionally spectroscopic redshift error. Returns the cleaned
    dataset with specified cuts and NA filtering as a DataFrame.

    df: DataFrame
        Non-cleaned dataframe of photo-z data. Dataset must have the original
        column names on retrieval from the HSC database.
    errors: bool
        Whether to add error column 'specz_err' in the dataframe. Can be used
        later for probabilistic/interval redshift estimation.
    filters: list[int]
        List of ints corresponding to the filters to use. The filters.md file
        contains the list of the applicable filters. If None, no cuts applied.
    scaled: bool
        Whether to scale the magnitude data, min-max across all bands.
    """
    
    # DEFINE CUTS
    cut_1 = (df['specz_redshift'] < 4)
    cut_2 = (df['specz_redshift'] > 0.01)
    cut_3 = (0 < df['specz_redshift_err']) & (df['specz_redshift_err'] < 1)
    cut_4 = df['specz_redshift_err'] < 0.005*(1+df['specz_redshift'])
    cut_5 = (df['g_cmodel_mag'] > 0) & (df['g_cmodel_mag'] < 50)   \
            & (df['r_cmodel_mag'] > 0) & (df['r_cmodel_mag'] < 50) \
            & (df['i_cmodel_mag'] > 0) & (df['i_cmodel_mag'] < 50) \
            & (df['z_cmodel_mag'] > 0) & (df['z_cmodel_mag'] < 50) \
            & (df['y_cmodel_mag'] > 0) & (df['y_cmodel_mag'] < 50)

    # PERFORM CUTS
    if (filters is not None):
        cuts = (cut_1 if 1 in filters else True)    \
                & (cut_2 if 2 in filters else True) \
                & (cut_3 if 3 in filters else True) \
                & (cut_4 if 4 in filters else True) \
                & (cut_5 if 5 in filters else True)
        df = df[cuts]
    
    # NA CUTS
    df = df.replace([-99., -99.9, np.inf], np.nan)
    df = df.dropna()

    # SELECT COLUMNS
    df = df[['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag', 'specz_redshift']]
    df.columns = ['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag', 'z_spec']
    if (errors):
        df = df.assign(z_spec_err=df.loc[:,'specz_redshift_err'])
    
    # NORMALIZATION
    if (scaled):
        X = df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
        X_norm = (X - X.values.min()) / (X.values.max() - X.values.min())
        df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']] = X_norm
    
    return df


def split_photoz_data(df, test_size=0.2):
    """
    Perform a train-test split of the data. Returns tuple of training and test
    datasets.
    
    df: DataFrame
        The clean dataframe of photo-z data. Columns include five band
        magnitudes, spectroscopic redshift, and optionally spectroscopic
        redshift error. 
    test_size: float
        Fractional size of the test set.
    """

    # SELECT INPUTS AND OUTPUTS
    X = df[['g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
    z = df['z_spec']

    # SPLIT WITH ERROR
    if ('z_spec_err' in df.columns):
        e = df['z_spec_err']
        X_train, X_test, z_train, z_test, e_train, e_test = train_test_split(X, z, e, test_size=test_size)
        return X_train, X_test, z_train, z_test, e_train, e_test

    # SPLIT WITHOUT ERROR
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size)
    return X_train, X_test, z_train, z_test


def list_duplicates(seq):
    '''
    Returns a generator with the value and the counts of each
    non-unique element of the input array.

    Inputs:
    --------
    seq - array of values

    Outputs:
    --------
    an interator that returns a list  [item, indices] which are 
    the indices that each duplicate item can be found. 

    '''
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def find_duplicate_galaxies(tab,match_name='specz_name'):
    '''
    Takes in a data frame and determines which values are duplicates. 
    By default will look at the specz_name for duplicates. Will return
    boolean arrays indicating which rows are duplicates and which 
    unique values to use. 

    Inputs:
    -------
    tab - a pandas data frame 

    Keywords:
    ---------
    match_name - the column to use for finding duplicates (default: 'specz_name')

    Output:
    ------
    duplicate_bool, unique_bool - two boolean arrays that indicate 
    which rows are duplicates and which galaxies to use. For galaxies that
    are duplicates, it will assign the first instance of the galaxy to use
    
    '''

    duplicate_gen = list_duplicates(tab[match_name])
    
    duplicate_bool = np.zeros(len(tab),dtype=bool)   # to indicate the galaxy is duplicate
    use_unique_bool = np.ones(len(tab),dtype=bool)   # to indicate the galaxy to use that is unique

    # iterate over the duplicates and just assign one of them to be the true match to the specz
    for name, ind in duplicate_gen:

        duplicate_bool[ind] = True
        # arbitrarily choose the first duplicate to the T one, set all the others to False
        use_unique_bool[ind[1:]] = False
        

    return duplicate_bool, use_unique_bool


##########################
# POINT ESTIMATE METRICS #
##########################


def delz(z_photo, z_spec):
    """
    Returns a vector of residuals/errors in prediction scaled by redshift.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    dz = (z_photo-z_spec) / (1+z_spec)
    return dz


def calculate_bias(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Returns a single value. Bias is a measure of center of the
    distribution of prediction errors.

    Parameters:
    -----------
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    conventional: bool
        Whether to use the conventional bias or not. If true, use conventional
        bias, or the median of the errors. If false, use the biweight bias, or
        the biweight location of the errors.

    Returns:
    -------
    b: float
        Bias metric, a measure of the center of the distribution of prediction
        errors.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        b = np.median(dz)
    else:
        b = biweight_location(dz)
    return b


def calculate_scatter(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Scatter is a measure of deviation in the
    distribution of prediction errors.

    Parameters:
    -----------
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    conventional: bool
        Whether to use the conventional scatter or not. If true, use
        conventional scatter, or the normal MAD of the errors. If false, use the
        biweight bias, or the biweight midvariance of the errors.

    Returns:
    -------
    s: float
        Scatter metric, a measure of the deviation in the distribution of
        prediction errors.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        s = median_abs_deviation(dz, scale='normal') # normal scale multiplied MAD by 1.4826
    else:
        s = np.sqrt(biweight_midvariance(dz))
    return s


def calculate_outlier_rate(z_photo, z_spec, conventional=False):
    """
    HSC METRIC. Outlier rate is the fraction of prediction errors above a certain level.

    Parameters:
    -----------
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    conventional: bool
        Whether to use the conventional outlier rate or not. If true, use
        conventional outlier rate, or rate of absolute errors above 0.15. If
        false, use the biweight outlier rate, or rate of errors outside two
        deviations of the norm based on the distribution of errors.

    Returns:
    -------
    eta: float
        Outlier rate, the fraction of prediction errors above a certain level.
    """
    dz = delz(z_photo, z_spec)
    if (conventional):
        outlier_scores = abs(dz)
        eta = np.mean(outlier_scores > 0.15)
    else:
        b = calculate_bias(z_photo, z_spec)
        s = calculate_scatter(z_photo, z_spec)
        outlier_scores = abs(dz - b)
        eta = np.mean(outlier_scores > (2*s))
    return eta


def calculate_loss(z_photo, z_spec):
    """
    HSC METRIC. Returns an array. Loss is accuracy metric defined by HSC, meant
    to capture the effects of bias, scatter, and outlier all in one. This has
    uses for both point and density estimation. See Tanaka+18 for more information.

    Parameters:
    -----------
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.

    Returns:
    -------
    L: array
        Loss metric, a measure of the accuracy of the predictions.
    """
    dz = delz(z_photo, z_spec)
    gamma = 0.15
    denominator = 1.0 + np.square(dz/gamma)
    L = 1 - 1.0 / denominator
    return L

def calculate_rmse(z_photo, z_spec):
    """
    Alternative to the scatter metric, this is the root mean square of the
    prediction errors. This is a measure of the deviation in the distribution
    of prediction errors. It is more sensitive to outliers than the scatter
    
    Parameters:
    -----------
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    
    Returns:
    -------
    rms: float
        Root mean square of the prediction errors.
    """
    dz = delz(z_photo, z_spec)
    rms = np.sqrt(np.mean(np.square(dz)))
    return rms

def calculate_catastrophic_outliers(z_photo, z_spec):
    '''
    Returns the catastrophic outlier metric, which is the absolute difference
    between the predicted and spectroscopic redshifts. For more information on
    the metric, see Singal+20 and Singal+22 for more information.

    Inputs:
    --------
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    
    Output:
    --------
    co: float
        Catastrophic outlier fraction 
    '''
    co = np.mean(np.abs(z_photo - z_spec) > 1.0)
    return co

def calculate_percentage_change(i, j):
    """
    Returns percent increase in some value with respect to j when applying i.
        i: float
        j: float
    """
    if i / j == 1:
        return f'{0}%'
    else:
        return '{:.2f}'.format(100 * ((i - j) / j)) + ' %' # 2 decimal precision
        # measures percent increase
        
############################
# PROBABILISTIC METRICS #
############################


def calculate_PIT(z_photo_vectors, z_spec):
    """
    HSC METRIC. Returns an array. Probability integral transform is the CDF of
    the CDF of a sampled distribution, used as a measure of the balance between
    the galaxy PDF peak sharpness and the accuracy of the peak. For each galaxy
    PDF, we take the CDF value at the true redshift, that is, the percentile in
    which the true redshift lies in the array of predictions for that galaxy.
    The PIT is the resulting distribution of these galaxy CDF values over the
    whole set of galaxies. A well-calibrated model will have a uniform PIT. This
    would correspond to small unbiased errors with thin peaks. Slopes in the PIT
    correspond to biases in the PDFs.

    Parameters:
    -----------
    z_photo_vectors: array of arrays
        One array of predicted redshifts per galaxy.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    z_spec = np.array(z_spec)
    z_photo_vectors = np.array(z_photo_vectors)
    length = len(z_photo_vectors[0])
    PIT  = np.zeros(len(z_photo_vectors))
    for i in range (len(z_photo_vectors)):          
        PIT[i] += len(np.where(z_photo_vectors[i]<z_spec[i])[0])*1.0/length
    return PIT


def calculate_CRPS(z_photo_vectors, z_spec):
    """
    HSC METRIC. Returns an array. Continuous ranked probability score is a
    measure of error between the predicted galaxy redshift PDF and the actual
    PDF of galaxy redshifts.

    z_photo_vectors: array of arrays
        One array of predicted redshifts per galaxy.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    z_spec = np.array(z_spec)
    z_photo_vectors = np.array(z_photo_vectors)
    length = len(z_photo_vectors[0])
    crps = np.zeros(len(z_photo_vectors))
    for i in range(len(z_photo_vectors)):
        for j in range(200):
            z = 4.0*j/200
            if z < z_spec[i]:
                crps[i] += ((len(np.where(z_photo_vectors[i]<z)[0])*1.0/length)**2)*(4.0/200)
            else:
                crps[i] += ((len(np.where(z_photo_vectors[i]<z)[0])*1.0/length-1)**2)*(4.0/200)
    return crps

def calculate_3sigma_outlier_fraction(z_photo, e_zphoto, z_spec):
    """
    LSST METRIC. Returns the fraction of galaxies with photometric redshift
    errors greater than 3 sigma. For models with uncertainty estimates.

    Inputs:
    --------
    z_photom: array
        Photometric or predicted redshifts.
    e_zphoto: array
        Photometric or predicted redshift errors.
    z_spec: array
        Spectroscopic or actual redshifts.
    
    Output:
    --------
    frac: float
    """
    diff = np.abs(z_photo - z_spec)
    frac = np.mean(diff > 3 * e_zphoto)
    return frac


########################
# QUICK VIEW FUNCTIONS #
########################


def get_point_metrics(z_photo, z_spec, binrange=np.linspace(0, 4, 21)):
    """
    Get a dataframe of the point estimate metrics given predictions.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    binned: bool
        True to calculate metrics by bins. This creates bins of size Δz = 0.2 on
        the spectroscopic redshifts.
    """

    # CREATE BINS
    bins = pd.cut(z_spec, bins=binrange)
    true_grouped = z_spec.groupby(bins, observed=False)
    pred_grouped = z_photo.groupby(bins, observed=False)

    # METRICS PER BIN
    metrics_list = []
    for zspec_bin in true_grouped.groups:

        # GET BIN'S PREDICTIONS
        try:
            binned_z_true = true_grouped.get_group(zspec_bin)
        except KeyError:
            continue
        try:
            binned_z_pred = pred_grouped.get_group(zspec_bin)
        except KeyError:
            continue

        # BASIC STATISTICS
        count = len(binned_z_true)
        L = np.mean(calculate_loss(binned_z_pred, binned_z_true))

        # BIWEIGHT
        bias_bw = calculate_bias(binned_z_pred, binned_z_true)
        scatter_bw = calculate_scatter(binned_z_pred, binned_z_true)
        outlier_bw = calculate_outlier_rate(binned_z_pred, binned_z_true)

        # CONVENTIONAL
        bias_conv = calculate_bias(binned_z_pred, binned_z_true, conventional=True)
        scatter_conv = calculate_scatter(binned_z_pred, binned_z_true, conventional=True)
        outlier_conv = calculate_outlier_rate(binned_z_pred, binned_z_true, conventional=True)
        
        # MSE
        mse = mean_squared_error(binned_z_true,binned_z_pred)

        # RMS
        rmse = calculate_rmse(binned_z_pred, binned_z_true)

        # Catastrophic outliers
        catastrophic_outliers = calculate_catastrophic_outliers(binned_z_pred, binned_z_true)

        # ADD TO ROW
        metrics_list.append([
            zspec_bin, count, L, bias_bw, bias_conv, 
            scatter_bw, scatter_conv, outlier_bw, outlier_conv,mse, rmse, catastrophic_outliers])

    # DATAFRAME CONVERSION
    metrics_df = pd.DataFrame(metrics_list, columns=[
        'zspec_bin', 'count', 'L', 'bias_bw', 'bias_conv',
        'scatter_bw', 'scatter_conv', 'outlier_bw', 'outlier_conv',
        'mse', 'rmse', 'catastrophic_outliers'])
    return metrics_df


def get_density_metrics(z_photo_vectors, z_spec):
    """
    Get a dataframe of the PIT and CRPS given predictions.

    z_photo_vectors: array of arrays
        One array of predicted redshifts per galaxy.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    galaxies = z_spec.index
    PIT = calculate_PIT(z_photo_vectors, z_spec)
    CRPS = calculate_CRPS(z_photo_vectors, z_spec)
    metrics_df = pd.DataFrame({'galaxy': galaxies,
                               'PIT': PIT,
                               'CRPS': CRPS})
    return metrics_df

########################
#   Compare Models     #
########################

published = {
    'NN23':{
        'predictionFile': '/data2/predictions/published_models/nn_jones.csv',
        'Creator': 'Jones, Evan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoz',
        'trueKey': 'specz'      
    },
    'BNN23':{
        'predictionFile': '/data2/predictions/published_models/bnn_jones.csv',
        'Creator': 'Jones, Evan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoz',
        'trueKey': 'specz'   
    },
    'CNN24': {
        'predictionFile': '/data2/predictions/published_models/cnn_jones.csv',
        'Creator': 'Jones, Evan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoz',
        'trueKey': 'specz'
    },
    'BCNN24': {
        'predictionFile': '/data2/predictions/published_models/bcnn_jones.csv',
        'Creator': 'Jones, Evan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoz',
        'trueKey': 'specz'
    }, 
    'NN25': {
        'predictionFile': '/data2/predictions/soriano_25_ground_truth_models/nn_1_final/galaxiesml_results_conformal.csv',
        'Creator': 'Soriano, Jonathan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoZ_mean',
        'trueKey': 'trueZ'
    },
    'BNN25': {
        'predictionFile': '/data2/predictions/soriano_25_ground_truth_models/bnn_1_final_111121/galaxiesml_results_conformal.csv',
        'Creator': 'Soriano, Jonathan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoZ_mean',
        'trueKey': 'trueZ'
    },
    'NN-TL': {
        'predictionFile': '/data2/predictions/soriano_25_ground_truth_models/nn_tl_final/galaxiesml_results_conformal.csv',
        'Creator': 'Soriano, Jonathan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoZ_mean',
        'trueKey': 'trueZ'
    },
    'BNN-TL': {
        'predictionFile': '/data2/predictions/soriano_25_ground_truth_models/bnn_tl_final_411331_1/galaxiesml_results_conformal.csv',
        'Creator': 'Soriano, Jonathan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoZ_mean',
        'trueKey': 'trueZ'
    },
    'NN-Combo': {
        'predictionFile': '/data2/predictions/soriano_25_ground_truth_models/nn_combo_final/galaxiesml_results_conformal.csv',
        'Creator': 'Soriano, Jonathan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoZ_mean',
        'trueKey': 'trueZ'
    },
    'BNN-Combo': {
        'predictionFile': '/data2/predictions/soriano_25_ground_truth_models/bnn_combo_final_411331/galaxiesml_results_conformal.csv',
        'Creator': 'Soriano, Jonathan',
        'Dataset': 'GalaxiesML',
        'predKey': 'photoZ_mean',
        'trueKey': 'trueZ'
    },
}

def get_published_model_metrics(binrange=np.linspace(0,4,2)):
    """
    Get the point estimate metrics for the published models.

    Args:
    -----
    binrange: array-like
        The range of redshifts to bin the metrics by. Default is np.linspace(0, 4, 21).
    
    Returns:
    --------
    metrics: DataFrame
        DataFrame containing the point estimate metrics for each published model.
    """
    metrics = []
    # Loop through each published model and get the metrics
    for model_name, model_info in published.items():
        prediction_file = model_info['predictionFile']
        tab = pd.read_csv(prediction_file)

        z_photo = tab[model_info['predKey']]
        z_spec = tab[model_info['trueKey']]

        metrics_df = get_point_metrics(z_photo, z_spec, binrange=binrange)

        metrics_df.insert(loc=0, column='dataset', value= [model_info['Dataset']]*metrics_df.shape[0])
        metrics_df.insert(loc=0, column='creator', value= [model_info['Creator']]*metrics_df.shape[0])
        metrics_df.insert(loc=0, column='model', value=[model_name]*metrics_df.shape[0])
    
        metrics.append(metrics_df)

    # Concatenate all the metrics DataFrames
    metrics = pd.concat(metrics, ignore_index=True)
    
    return metrics

def compare_models(z_photo, z_spec, model_names=None, dataset=None, creator=None, binrange=np.linspace(0,4,2), published_models=True):
    """
    Compare the point estimate metrics of different models.

    Args:
    -----
    z_photo: List of Series or array
        Photometric or predicted redshifts for each model to compare.
    z_spec: List of Series or array
        Spectroscopic or actual redshift
    model_names: list of str, optional
        Names of the models to compare. If None, uses the default names e.g. 'model_1', 'model_2', etc.
    dataset: str, optional
        Name of the dataset used for the models. If None, no dataset name is added.
    creator: str, optional
        Name of the creator of the models. If None, no creator name is added.
    Returns:
    --------
    metrics: DataFrame
        DataFrame containing the point estimate metrics for each model.
    """

    if type(z_photo) == list:
        if len(z_photo) != len(z_spec):
            raise ValueError("z_photo and z_spec must have the same length.")
        
        # Check if one or more entries in list are not Series
        if not all(isinstance(z, pd.Series) for z in z_photo):
            z_photo = [pd.Series(z) for z in z_photo]
        if not all(isinstance(z, pd.Series) for z in z_spec):
            z_spec = [pd.Series(z) for z in z_spec]
    else:
        if type(z_photo) == pd.Series:
            z_photo = [z_photo]
            z_spec = [z_spec]
        elif type(z_photo) == np.ndarray:
            z_photo = [pd.Series(z_photo)]
            z_spec = [pd.Series(z_spec)]
        else:
            raise TypeError("z_photo and z_spec must be either a list, Series, or numpy array,")

    # Check if model_names is provided, if not, create default names
    if model_names is None:
        model_names = [f'model_{i+1}' for i in range(len(z_photo))]
    else:
        if type(model_names) != list:
            model_names = [model_names]
    
    if len(model_names) != len(z_photo):
        raise ValueError("model_names must have the same length as z_photo and z_spec.")

    # Get point metrics for each model
    metrics = []
    for i in range(len(z_photo)):
        metrics_df = get_point_metrics(z_photo[i], z_spec[i], binrange=binrange)

        metrics_df.insert(loc=0, column='model', value=[model_names[i]]*metrics_df.shape[0])
        metrics_df.insert(loc=0, column='dataset', value=[dataset]*metrics_df.shape[0] if dataset else [np.nan]*metrics_df.shape[0])
        metrics_df.insert(loc=0, column='creator', value=[creator]*metrics_df.shape[0] if creator else [np.nan]*metrics_df.shape[0])
        metrics.append(metrics_df)
    metrics = pd.concat(metrics, ignore_index=True)

    if published_models:
        # Get the published model metrics and concatenate with the current metrics
        published_metrics = get_published_model_metrics(binrange=binrange)
        metrics = pd.concat([published_metrics, metrics], ignore_index=True)

    return metrics

######################
# Outlier Analysis   #
#####################
def get_outlier_indices(z_photo, z_spec, threshold=0.15):
    """
    Get the indices of the outliers in the predictions.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    threshold: float
        The threshold for defining an outlier. Default is 0.15.
    
    Returns:
    --------
    outlier_indices: array
        Indices of the outliers in the predictions.
    """
    dz = delz(z_photo, z_spec)
    outlier_indices = np.where(np.abs(dz) > threshold)[0]
    return outlier_indices

def get_common_outliers(oultier_objectid_list_1, oultier_objectid_list_2):
    """
    Get the common outliers between two lists of outlier object IDs.
    Prediction file should contain a column 'object_id' for this to work.

    Parameters:
    oultier_objectid_list_1: list
        List of object IDs for the first set of outliers.
    oultier_objectid_list_2: list
        List of object IDs for the second set of outliers.
    
    Returns:
    --------
    common_outliers: list
        List of common object IDs between the two sets of outliers.
    """
    common_outliers = list(set(oultier_objectid_list_1) & set(oultier_objectid_list_2))
    return common_outliers
    

######################
# PLOTTING FUNCTIONS #
######################


def plot_predictions(z_photo, z_spec):
    """
    Plot predicted vs. true redshifts.

    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    
    sns.set(rc={'figure.figsize':(10,10)})

    sns.histplot(x=z_spec, y=z_photo, cmap='viridis', cbar=True)
    sns.lineplot(x=[0,4], y=[0,4])
    plt.xlabel('True redshift')
    plt.ylabel('Predicted redshift')
    
    
def plot_point_metrics(metrics):
    """
    Plot binned metrics. Must have already generated point metrics.

    metrics: DataFrame
        Binned point estimate metrics given predictions.
    """

    sns.set(rc={'figure.figsize':(18,8), 'lines.markersize':10})
    bin_lefts = metrics['zspec_bin'].apply(lambda x: x.left)
    sns.lineplot(x=[0,4], y=[0,0], linewidth=2, color='black')
    sns.scatterplot(x=bin_lefts, y=metrics['bias_bw'], marker = '.', edgecolor='none', label='bias')
    sns.scatterplot(x=bin_lefts, y=metrics['bias_conv'], marker = '.', facecolors='none', edgecolor='black')
    sns.scatterplot(x=bin_lefts, y=metrics['scatter_bw'], marker = "s", edgecolor='none', label='scatter')
    sns.scatterplot(x=bin_lefts, y=metrics['scatter_conv'], marker = "s", facecolors='none', edgecolor='black')
    sns.scatterplot(x=bin_lefts, y=metrics['outlier_bw'], marker = "v", edgecolor='none', label='outlier')
    sns.scatterplot(x=bin_lefts, y=metrics['outlier_conv'], marker = "v", facecolors='none', edgecolor='black')
    sns.scatterplot(x=bin_lefts, y=metrics['L'], marker = 'x', label = "loss", linewidth=4)
    plt.xlabel('Redshift')
    plt.ylabel('Statistic value')

    
def plot_density_metrics(metrics):
    """
    Plot density metrics. Must have already generated density metrics.

    metrics: DataFrame
        Density estimate metrics given predictions.
    """
    
    sns.set(rc={'figure.figsize':(18,8), 'lines.markersize':10})

    PIT = metrics['PIT']
    CRPS = metrics['CRPS']
    fig, axes = plt.subplots(1,2)
    sns.histplot(PIT, bins = 50, ax=axes[0])
    sns.histplot(CRPS, bins = 50, ax=axes[1])
    plt.yscale('log')

    
def compare_point_metrics_scatter(metrics_array, legends, desc, markers, zmax=4, marker_size=130):
    """
    Pass in a list where the elements are the output of binned get_point_metrics() from photoz_utils.py and compare some of the metrics on each plot.
        metrics_array: list
            A list of metric DataFrames acquired from binned get_point_metrics() output.
        legends: list
            Legend labels for the models in the metrics_array as strings.
        desc: str
            Plot title.
        markers: list
            A list of strings where you indicate the markers for models in metrics_array.
        zmax: float
            Maximum redshift to be displayed.
        marker_size: int
            Size of the markers displayed.
    """
    index = len(metrics_array)
    sns.set(rc={'figure.figsize':(16,24), 'lines.markersize':10})
    plt.suptitle(f'{desc}')
    plt.subplots_adjust(hspace=0.3)
    metrics = ['bias_conv', 'scatter_conv', 'outlier_conv', 'loss', 'mse']
    for i in enumerate(metrics):
        plt.subplot(len(metrics), 1, i[0] + 1)
        for j in range(0,index):
            
            sns.lineplot(x=[0, zmax], y=[0, 0], linewidth=2, color='black')
            bin_lefts = metrics_array[j]['zspec_bin'].apply(lambda x: x.left)
            sns.scatterplot(
                x=bin_lefts, 
                y=metrics_array[j][i[1]], 
                marker = markers[j], 
                edgecolor='none', 
                label=f'{legends[j]}', 
                s=marker_size)
            
            plt.xlabel('Redshift', fontsize=18)
            plt.ylabel(f'{metrics[i[0]]}', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16)
            
def compare_point_metrics_scatter2(metrics_array, legends, metrics = None,
    markers=None, zmax=4, marker_size=130,xlim=None,label_plots=True):
    """
    Pass in a list where the elements are the output of binned get_point_metrics() from photoz_utils.py and compare some of the metrics on each plot.
        metrics_array: list
            A list of metric DataFrames acquired from binned get_point_metrics() output.
        legends: list
            Legend labels for the models in the metrics_array as strings.
        zmax: float
            Maximum redshift to be displayed.
        marker_size: int
            Size of the markers displayed.
    Keywords:
        markers: list
            A list of strings where you indicate the markers for models in metrics_array.
        meters: list
            A list of strings where you indicate the metrics to be displayed. (Default: all)
        xlim: tuple
            Tuple of the form (xmin,xmax) to set the x limits of the plot.
        label_plots: bool 
            Label the corners of the plots with a letter. (Default: True)

    Return:
        fig,axs: tuple 
            Tuple of the figure and the axes of the plot.

    """
    index = len(metrics_array)
    
    # assume the metrics are the same as the column names
    if metrics is None:
        metrics = metrics_array[0].columns[1:]
    #metrics = ['bias_conv', 'scatter_conv', 'outlier_conv', 'loss', 'mse']
    if markers is None:
        markers = ['o','X','s','v','^','+','d','*','p','h','X','P','D']
    fig, axs = plt.subplots(math.ceil(len(metrics)/2), 2, figsize=(17,16))
    plt.subplots_adjust(hspace=0.3)

    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M']
    for i in enumerate(metrics):
        plt.subplot(math.ceil(len(metrics)/2), 2, i[0] + 1)
        for j in range(0,index):
            
            sns.lineplot(x=[0, zmax], y=[0, 0], linewidth=2, color='black')
            bin_lefts = (metrics_array[j]).iloc[:,0].apply(lambda x: x.left)
            bins_rights = (metrics_array[j]).iloc[:,0].apply(lambda x: x.right)
            bin_center = (bin_lefts + bins_rights)/2
            sns.scatterplot(
                x=bin_center, 
                y=metrics_array[j][i[1]], 
                marker = markers[j], 
                edgecolor='none', 
                label=f'{legends[j]}', 
                s=marker_size)
            if xlim is not None:
                plt.xlim(xlim)
            else:
                # get xlim from the plot
                xlim = plt.xlim()

                
            # using the xlim calculate the ylim of the plot from the data within that x range
            # Find the minimum and maximum within the x range from xlim
            x_range_mask = (bin_center >= xlim[0]) & (bin_center <= xlim[1])
            y_values = metrics_array[j][i[1]][x_range_mask]
            min_value = np.min(y_values)
            max_value = np.max(y_values)
            delta = max_value - min_value
            plt.ylim(min_value-delta*0.1,max_value+delta*0.1)


    # label the corner of the plot
    if label_plots:
        for i in range(len(metrics)):
            current_ax = axs.flat[i]
            plt.text(-0.25,0.95,labels[i]+')',transform=current_ax.transAxes,fontsize=18,weight='bold')
        

    return fig,axs
            


def compare_point_metrics_table(metrics_array1, metrics_array2, caption):
    """
    Pass in the output of get_point_metrics() from photoz_utils.py and get a table comparing the percent increase that metrics_array1
    had compared to metrics_array2 for each metric for a given bin. Has issues displays more than a few bins. Does not display biweight
    metrics.
        metrics_array1: DataFrame
            Binned metrics DataFrame acquired from get_point_metrics().
        metrics_array2: DataFrame
            Binned metrics DataFrame acquired from get_point_metrics().
        caption: str
            Describe what metrics are being displayed.
    """
    metrics = ['bias_conv', 'scatter_conv', 'outlier_conv', 'loss', 'mse']
    percs = []
    title = ['Metric']
    zspec_ranges = list(metrics_array1.loc[:,'zspec_bin'].apply(lambda x: f'z: {x.left} - {x.right}'))
    heading = title + zspec_ranges
    percs.append(heading)
    
    for i in enumerate(metrics):
        array1_metric = np.asarray(metrics_array1.loc[:, i[1]])
        array2_metric = np.asarray(metrics_array2.loc[:, i[1]])
        perc = [calculate_percentage_change(i, j) for i, j in zip(array1_metric, array2_metric)]
        perc = [i[1] + ' percent change'] + perc # add metric name
        percs.append(perc)
    print(caption, '\n')
    print(tabulate(percs, headers='firstrow', tablefmt='fancy_grid'))
    
    
def compare_point_metrics_bar(model_arr, model_names, fig_size=(10,5), conventional_only=True, palette='colorblind'):
    """
    list: model_arr
        List of the Pandas DataFrames which are the output of unbinned get_point_metrics().
    list: models_names
        List of strings which represent the models in the model_arr in order.
    fig_size: tuple
        Size of the barplots.
    conventional_only: bool
        Determines which metrics are displayed. If conventional, none of the biweight metrics are shown.
    palette: str
        Determines the color palette of the barplots.
        
    """
    if conventional_only:
        metrics = [model_arr[0].columns[i] for i in [2,4,6,8,9]] # List of conv metrics names
    else:
        metrics = list(model_arr[0].columns[2:10]) # List of metrics names
        
    bar_colors = sns.color_palette(palette=palette, n_colors=len(model_arr)) 
     
    model_metrics_list = []
    
    for metric in metrics:
        model_metrics = []
        
        for i in range(0, len(model_arr)):
            model_metrics.append(model_arr[i].loc[0][metric]) # Add each model's value for a certain metric its own list.
            
        model_metrics_dict = dict(zip(model_names, model_metrics)) # Add model names to the corresponding models.
        model_metrics_list.append(model_metrics_dict) # Add model metric values for all metrics
    # print(model_metrics_list)
    
    for i in range(0, len(model_metrics_list)):
        
        idx = list(model_metrics_list[i].keys())
        vals = np.abs(list(model_metrics_list[i].values()))
        fig = plt.figure(figsize=fig_size)
        ax = plt.bar(idx, vals, width=0.4, color=bar_colors)
        plt.xlabel(f'Model', fontsize=18)
        plt.ylabel(f'{metrics[i]}', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(iter(ax), model_names, fontsize=16) # must iterate over the bars for the legend to work
        
        
###############
# DATA SAVING #
###############


def save_with_oid_std(name, object_id, specz, photoz, photoz_err):
    """
    Saves predictions providing object id labels. 
    """
    predictions = np.transpose(np.vstack((object_id, specz, photoz, photoz_err)))
    now = datetime.now()
    t_string = now.strftime('%Y_%m_%d_%H:%M:%S')
    m_string = now.strftime('%Y_%m')
    if os.path.exists('/predictions/'+name+'/'+m_string)==False:
        os.makedirs('/predictions/'+name+'/'+m_string)
    df = pd.DataFrame(predictions, columns=['object_id', 'specz', 'photoz', 'photoz_err'])
    df.to_csv('/predictions/'+name+'/'+m_string+'/'+t_string+'_testing.csv')
    

def save_with_oid(name, object_id, specz, photoz):
    """
    Saves predictions providing object id labels, for non variational models. 
    """
    photoz_err = [nan] * len(object_id)
    predictions = np.transpose(np.vstack((object_id, specz, photoz, photoz_err)))
    now = datetime.now()
    t_string = now.strftime('%Y_%m_%d_%H:%M:%S')
    m_string = now.strftime('%Y_%m')
    if os.path.exists('/predictions/'+name+'/'+m_string)==False:
        os.makedirs('/predictions/'+name+'/'+m_string)
    df = pd.DataFrame(predictions, columns=['object_id', 'specz', 'photoz', 'photoz_err'])
    df.to_csv('/predictions/'+name+'/'+m_string+'/'+t_string+'_testing.csv')

    
def save_train_with_oid_std(name, object_id, specz, photoz, photoz_err):
    """
    Saves predictions on training set providing object id labels. 
    """
    predictions = np.transpose(np.vstack((object_id, specz, photoz, photoz_err)))
    now = datetime.now()
    t_string = now.strftime('%Y_%m_%d_%H:%M:%S')
    m_string = now.strftime('%Y_%m')
    if os.path.exists('/predictions/'+name+'/'+m_string)==False:
        os.makedirs('/predictions/'+name+'/'+m_string)
    df = pd.DataFrame(predictions, columns=['object_id', 'specz', 'photoz', 'photoz_err'])
    df.to_csv('/predictions/'+name+'/'+m_string+'/'+t_string+'_training.csv')

    
def save_train_with_oid(name, object_id, specz, photoz):
    """
    Saves predictions on training set providing object id labels. 
    """
    photoz_err = [nan] * len(object_id)
    predictions = np.transpose(np.vstack((object_id, specz, photoz, photoz_err)))
    now = datetime.now()
    t_string = now.strftime('%Y_%m_%d_%H:%M:%S')
    m_string = now.strftime('%Y_%m')
    if os.path.exists('/predictions/'+name+'/'+m_string)==False:
        os.makedirs('/predictions/'+name+'/'+m_string)
    df = pd.DataFrame(predictions, columns=['object_id', 'specz', 'photoz', 'photoz_err'])
    df.to_csv('/predictions/'+name+'/'+m_string+'/'+t_string+'_training.csv')

    
def save_validation_with_oid_std(name, object_id, specz, photoz, photoz_err):
    """
    Saves predictions on training set providing object id labels. 
    """
    predictions = np.transpose(np.vstack((object_id, specz, photoz, photoz_err)))
    now = datetime.now()
    t_string = now.strftime('%Y_%m_%d_%H:%M:%S')
    m_string = now.strftime('%Y_%m')
    if os.path.exists('/predictions/'+name+'/'+m_string)==False:
        os.makedirs('/predictions/'+name+'/'+m_string)
    df = pd.DataFrame(predictions, columns=['object_id', 'specz', 'photoz', 'photoz_err'])
    df.to_csv('/predictions/'+name+'/'+m_string+'/'+t_string+'_validation.csv')

    
def save_validation_with_oid(name, object_id, specz, photoz):
    """
    Saves predictions on training set providing object id labels. 
    """
    photoz_err = [nan] * len(object_id)
    predictions = np.transpose(np.vstack((object_id, specz, photoz, photoz_err)))
    now = datetime.now()
    t_string = now.strftime('%Y_%m_%d_%H:%M:%S')
    m_string = now.strftime('%Y_%m')
    if os.path.exists('/predictions/'+name+'/'+m_string)==False:
        os.makedirs('/predictions/'+name+'/'+m_string)
    df = pd.DataFrame(predictions, columns=['object_id', 'specz', 'photoz', 'photoz_err'])
    df.to_csv('/predictions/'+name+'/'+m_string+'/'+t_string+'_validation.csv')

