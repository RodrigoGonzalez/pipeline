import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats as scs
from scipy.stats import skew, pearsonr, spearmanr, kendalltau
import cPickle as pickle
import string

from edatools import compare_missing_values, count_data, summary_stats, printall

class data_transporter(object):
    """
    A 'data' class that will contain all of the data, for ease of use. This class can be used to quickly access the different types of variables (continuous, discrete, ordinal, nominal). Determined by examining data.
    """

    def __init__(self, filename):
        """
        Instantiate class and initialize dataframe, id, and labels if training data set.
        """
        print 'Instantiate data class\n'
        self.package = self.load_data(filename)
        self.unpack(self.package)

        # Load feature labels
        self.load_feature_labels()

    def load_data(self, filename):
        """
        Load data and add variables to dataframes,
        """
        print "Reading in data from csv\n"

        df_train = pd.read_csv(filename[0])
        df_test  = pd.read_csv(filename[1])

        # Replace individual data points
        # The sample with ID 666 has GarageArea, GarageCars, and GarageType
        # but none of the other fields, so use the mode and median to fill them in.
        df_test.loc[666, 'GarageFinish'] = 'Unf'
        df_test.loc[666, 'GarageCond'] = 'TA'
        df_test.loc[666, 'GarageQual'] = 'TA'

        df = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                        df_test.loc[:,'MSSubClass':'SaleCondition']))

        df.reset_index(inplace=True, drop=True)



        df_train_id = df_train['Id']
        df_test_id  = df_test['Id']

        y = df_train['SalePrice'].get_values()

        # Package that will be unpacked
        package = [df, df_train_id, df_test_id, y]

        return package

    def unpack(self, package):
        """
        INPUT: Package of raw data
        OUPUT: No output returned simply defining instance variables
        """
        print "Loading data on transporter\n"
        self.df = package[0]
        self.df_train_id = package[1]
        self.df_test_id = package[2]
        self.y = package[3]
        self.original_features = self.df.columns.unique()
        self.X = 0
        self.X_pred = 0
        self.df_train = 0
        self.df_test = 0
        self.df_pretransform = 0


    def continuous(self):
        """
        Returns all of the continuous variable labels
        """

        c = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal']

        self.continuous_feat = c

    def discrete(self):
        """
        Returns all of the discrete variable labels
        """

        d = ['YearBuilt', 'YearRemodAdd','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

        self.discrete_feat = d

    def nominal(self):
        """
        Returns all of the nominal variable labels
        """

        n = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',  'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'PavedDrive', 'MiscFeature']

        self.nominal_feat = n

    def ordinal(self):
        """
        Returns all of the ordinal variable labels
        """

        o = ['LotShape', 'LandContour', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'SaleType', 'SaleCondition']

        self.ordinal_feat = o

    def constructed(self):
        """
        Returns all constructed variable labels
        """

        self.constructed_feat = []
        self.mapped_feat = []
        self.removed_feat = []
        self.binary_feat = []

    def load_feature_labels(self):
        """
        Load labels onto data class
        """

        print 'Load class labels\n'

        self.continuous()
        self.discrete()
        self.nominal()
        self.ordinal()
        self.constructed()



def missing_values(dt):
    """
    Replacing all of the NaN's with values that were determined by examining the individual rows that were missing values as well as surrounding rows and the number of values for each label in a feature
    """
    # Find null values using all_data[pd.isnull(all_data['PoolQC'])]

    print 'Replace missing values\n'

    df = dt.df

    # Replace garage year with year built if null
    df.GarageYrBlt.fillna(df.YearBuilt, inplace=True)

    # MSZoning is replaced with the most common value within each Neighborhood as most
    zoning = {}

    for N in df.Neighborhood.unique().tolist():
        zoning[N] = df[df['Neighborhood'] == N]['MSZoning'].mode()[0]

    mask = df['MSZoning'].isnull()
    df.loc[mask, 'MSZoning'] = df.loc[mask, 'Neighborhood'].map(zoning)

    # Replace nulls with most common values
    D = {
            'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd',
            'Utilities': 'AllPub', 'Electrical': 'SBrkr',
            'Functional': 'Typ', 'SaleType': 'WD'

            }

    for k, v in D.iteritems():
        df[k].fillna(value=v, inplace=True)

    # Replace null with None
    col_none = [
                'Alley', 'MasVnrType',
                'Fence', 'MiscFeature',
                'GarageType', 'BsmtExposure',
                'PoolQC', 'BsmtQual',
                'BsmtCond', 'BsmtFinType1',
                'BsmtFinType2', 'KitchenQual',
                'FireplaceQu', 'GarageQual',
                'GarageCond', 'GarageFinish'

                ]

    for col in col_none:
        df[col].fillna(value='None', inplace=True)


    # Replace null with 0
    col_0 = [
                'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
                'GarageCars', 'GarageArea', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'MasVnrArea',
                'PoolArea', 'MiscVal', 'LotFrontage',
                'PoolArea', 'TotalBsmtSF'

                ]

    for col in col_0:
        df[col].fillna(value=0, inplace=True)

    # Convert to floats
    df['OverallQual'] = df['OverallQual'].astype(float)
    df['OverallCond'] = df['OverallCond'].astype(float)

    dt.df = df

    return dt

    # New variable construction

    ### Use self.function to place functions below inside the class

def load_new_features(dt):
    """
    Add new columns to data frame
    """

    print 'Load new features\n'
    df = dt.df

    # Additions of new features
    # The total square feet of the house: float64
    df['TotSqFt'] = (df['TotalBsmtSF'] + df['GrLivArea']).astype(float)

    # The time since the house was sold, 2010 base year
    df['TimeSinceSold'] = ((2010 - df['YrSold']) * 1).astype(float)

    # The number of bathroooms in the house: float64
    df['TotBath'] = (df['BsmtHalfBath'] + df['BsmtFullBath'] + df['FullBath'] + df['HalfBath']).astype(float)

    # How old the house was at the time of sale 0, 1
    df['SaleAge'] = (df['YrSold'] - df['YearBuilt']).astype(float)
    df['SaleAge'].replace(to_replace=-1, value=0, inplace=True)

    # How many years has it been since the remodel: 0,1
    df['YrSinceRemodel'] = (df['YrSold'] - df['YearRemodAdd']).astype(float)
    df['YrSinceRemodel'].replace({-2:0, -1:0}, inplace=True)

    # Is the square footage greater than two standard deviations from the mean?  sq ft: 0, 1
    PremiumSQ = df.TotSqFt.mean() + 2 * df.TotSqFt.std()
    df['Premium'] = (df['TotSqFt'] > PremiumSQ) * 1

    # Is the garage detached: 0, 1
    df['IsGarageDetached'] = (df['GarageType'] == 'Detchd') * 1

    # Most have a paved drive so treat dirt/gravel and partial pavement as 'not paved': 0,1
    df['IsPavedDrive'] = (df['PavedDrive'] == 'Y') * 1

    # The only interesting 'misc. feature' is the presence of a shed: 0,1
    df['HasShed'] = (df['MiscFeature'] == 'Shed') * 1

    # If YearRemodAdd != YearBuilt, then a remodeling took place at some point : 0,1
    df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']) * 1

    # Did a remodeling happen in the year the house was sold?: 0,1
    df['RecentRemodel'] = (df['YearRemodAdd'] == df['YrSold']) * 1

    # Was house sold in the year it was built?: 0,1
    df['VeryNewHouse'] = (df['YearBuilt'] == df['YrSold']) * 1

    # Features a result of specific labels 0, 1
    df['Has2ndFloor'] = (df['2ndFlrSF'] == 0) * 1

    df['HasMasVnr'] = (df['MasVnrArea'] == 0) * 1

    df['HasWoodDeck'] = (df['WoodDeckSF'] == 0) * 1

    df['HasOpenPorch'] = (df['OpenPorchSF'] == 0) * 1

    df['HasEnclosedPorch'] = (df['EnclosedPorch'] == 0) * 1

    df['Has3SsnPorch'] = (df['3SsnPorch'] == 0) * 1

    df['HasScreenPorch'] = (df['ScreenPorch'] == 0) * 1

    # Is the house in a residential district: 1, 0
    df['Residential'] = df['MSZoning'].isin(['C (all)', 'FV']) * 1

    # Is the house level: 1, 0
    df['Level'] = df['LandContour'].isin(['Bnk', 'Low', 'HLS']) * 1

    # Does the house have Shingles: 0, 1
    df['HasShingles'] = df['RoofMatl'].isin(['ClyTile', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl']) * 1

    # Does the house have a gas furnace: 0, 1
    df['GasFurance'] = df['Electrical'].isin(['FuseA', 'FuseF', 'FuseP', 'Mix']) * 1

    # Circuit Breakers: 0, 1
    df['CircuitBreaker'] = df['LandContour'].isin(['Bnk', 'Low', 'HLS']) * 1

    # Typical home functionality? :0, 1
    df['TypHomeFunc'] = df['Functional'].isin(['Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'Sev']) * 1

    # Is the dive paved?:  0, 1
    df['Paved'] = df['PavedDrive'].isin(['N', 'P']) * 1

    # There is no fence: 0, 1
    df['NoFence'] = df['Fence'].isin(['GdPrv', 'GdWo', 'MnPrv', 'MnWw']) * 1

    # Is it a conventional warranty deed? 0, 1
    df['ConvWarrantyDeed'] = df['SaleType'].isin(['COD','CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth']) * 1

    # Was the sale condition normal?: 0, 1
    df['NormalSaleCondition'] = df['SaleCondition'].isin(['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']) * 1

    # Regular lot shape": 0, 1
    df['RegLotShape'] = df['LotShape'].isin(['IR3', 'IR2', 'IR1']) * 1

    # Worst time to buy July/Aug/Nov/Dec/Jan/Feb: 0, 1
    df['BestBuyTime'] = df['MoSold'].isin([3, 4, 5, 6, 9, 10]) * 1

    # Append constructed feature names
    dt.constructed_feat = ['TotSqFt', 'TimeSinceSold', 'TotBath', 'SaleAge',
                            'YrSinceRemodel', 'Premium', 'IsPavedDrive',
                            'HasShed', 'Remodeled', 'RecentRemodel',
                            'VeryNewHouse', 'Has2ndFloor', 'HasMasVnr',
                            'HasWoodDeck', 'HasOpenPorch', 'HasEnclosedPorch',
                            'Has3SsnPorch', 'HasScreenPorch', 'TimeSinceSold',
                            'Residential',  'Level', 'HasShingles',
                            'GasFurance', 'CircuitBreaker', 'TypHomeFunc',
                            'Paved', 'NoFence', 'ConvWarrantyDeed',
                            'NormalSaleCondition', 'RegLotShape', 'BestBuyTime']


    binary_features = df[dt.constructed_feat].select_dtypes(include=['int64']).columns.unique()

    dt.binary_feat = binary_features.tolist()

    dt.df = df

    return dt


def map_ordinal_feat(dt):
    """
    Ordinal features are mapped to numerical representations based on the labels of the individual columns, both the new features and old features are kept and will be fed to feature selection models
    """
    df = dt.df

    print 'Mapping ordinal features\n'

    # MSSubClass map digits to alphabetic labels for type of dwelling so models dont treat as numerical data

    subclass = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]
    alpha =  list(string.ascii_uppercase)
    sub_list = zip(subclass, alpha[:len(subclass)])
    sub_mapping = dict(sub_list)

    df['MSSubClass'] = df['MSSubClass'].map(sub_mapping)


    # Slope Mapping
    slope = ['Gtl', 'Mod', 'Sev']
    s_list = zip(slope, xrange(0, len(slope)))
    sloped_mapping = dict(s_list)

    df['map_LandSlope'] = df['LandSlope'].map(sloped_mapping).astype(float)

    # Exposure
    exposure = ['None', 'No', 'Mn', 'Av', 'Gd']
    ex_list = zip(exposure, xrange(0, len(exposure)))
    ex_mapping = dict(ex_list)

    df['BsmtExposure'] = df['BsmtExposure'].map(ex_mapping)

    # Garage mapping
    garage_mapping = {'None': 0.0, 'Unf': 1.0, 'RFn': 2.0, 'Fin': 3.0}
    df['map_GarageFinish'] = df['GarageFinish'].map(garage_mapping).astype(float)

    # Fence mapping
    fence_mapping = {'None': 0.0, 'MnWw': 1.0, 'GdWo': 2.0, 'MnPrv': 3.0, 'GdPrv': 4.0}
    df['map_Fence'] = df['Fence'].map(fence_mapping).astype(float)
    df['map_Fence'].fillna(value=0, inplace=True)

    ordinals_quality = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

    quality = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    q_list = zip(quality, xrange(0, len(quality)+1))
    quality_mapping = dict(q_list)

    for column in iter(ordinals_quality):
        new_col = '{}'.format(column)
        df[new_col] = df[column].map(quality_mapping)
        df[new_col].fillna(value=0, inplace=True)

    dt.mapped_feat = ['map_LandSlope', 'map_BsmtExposure', 'map_GarageFinish', 'map_Fence']

    return dt


def encode_labels(dt):
    """
    Creating new dummy features from all categorical feature labels using pandas get_dummies
    """

    print 'Encoding labels of nominal and ordinal features\n'

    features = dt.ordinal_feat + dt.nominal_feat

    remove_feat = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtExposure']

    features = [x for x in features if x not in remove_feat]

    columns_before = set(dt.df.columns.unique())

    # Get dummies and drop the first column
    dt.df = pd.get_dummies(dt.df, columns=features, drop_first=True)

    # Generate list of binary columns to exclude from standarization
    columns_after = set(dt.df.columns.unique())
    dt.binary_feat = dt.binary_feat + list(columns_after.difference(columns_before))

    return dt

def remove_missing_features(dt):
    """
    The following features do not appear in the test data set, in order to avoid over fitting we will remove them (found by running missing values in edatools)
    """
    print 'Dropping features that dont show up in training set\n'

    D = compare_missing_values()

    # generate column names from missing values dictionary, clean up
    drop_feats = []

    for k,v in D.iteritems():
        if v:
            [drop_feats.append([k, x]) for x in v]

    drop_columns = ['{}_{}'.format(k, v) for k,v in drop_feats]
    drop_columns.append('MSSubClass_M')

    # Check if feature column was generated when encoding and then drop due to missing value in missing features
    [dt.df.drop(column, axis=1, inplace=True) for column in drop_columns if column in dt.df.columns]

    remove_binary = [x for x in drop_columns if x in dt.binary_feat]

    [dt.binary_feat.remove(bin_feat) for bin_feat in remove_binary]

    return dt


def preprocessing(dt):
    """
    Take log1 transformations of numerical values with absolute skew greater than 0.5 to help normalize data. Then standarize data (mu = 0, sigma = 1) in order to more efficiently train the machine learning algorithms
    """

    numeric_features = dt.continuous_feat + dt.discrete_feat + dt.ordinal_feat + ['TotSqFt', 'TotBath', 'SaleAge', 'YrSinceRemodel', 'TimeSinceSold', 'map_LandSlope', 'map_GarageFinish', 'map_Fence']

    # These were encoded in encode_labels -> not numerical values
    remove_feat = [ 'LotShape',
                    'LandContour',
                    'Utilities',
                    'LandSlope',
                    'BsmtFinType1',
                    'BsmtFinType2',
                    'GarageFinish',
                    'Fence',
                    'SaleType',
                    'SaleCondition',
                    'MoSold']

    # Following transformations determined by plotting feature vs log1p(SalePrice) in PlotsPreTransformedData

    print "Begin transformations\n"

    # The following features show that although skew is big, log transformations lead to lower Pearson correlation
    low_pearson_feat = ['BsmtCond', 'BsmtUnfSF', 'PoolQC']

    # Special log transformation of the form log(x/mean(x)+k) applied
    mean_log_feat = ['YearBuilt_log', 'MasVnrArea_log', 'BsmtFinSF1_log', 'TotalBsmtSF_log', 'GarageQual_log', 'SaleAge_log']

    feat_log = ['YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageQual', 'SaleAge']

    k_s = [1, 1, 10, 1, 10, 1]

    mean_log_dict = zip(mean_log_feat, k_s, feat_log)

    for feats in mean_log_dict:
        mean = dt.df[feats[2]].mean()
        dt.df[feats[0]] = np.log(dt.df[feats[2]] / mean + feats[1])


    # Power transformations (squared and cubed)
    feats = ['BsmtQual', 'BsmtQual', 'BsmtQual', '2ndFlrSF', '2ndFlrSF', '2ndFlrSF', 'BsmtHalfBath', 'BsmtHalfBath', 'BsmtHalfBath']
    power_feat = ['BsmtQual1', 'BsmtQual2', 'BsmtQual3', '2ndFlrSF', '2ndFlrSF2', '2ndFlrSF3', 'BsmtHalfBath1', 'BsmtHalfBath2', 'BsmtHalfBath3']
    powers = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    power_dict = zip(feats, power_feat, powers)

    for feats in power_dict:
        dt.df[feats[1]] = np.power(dt.df[feats[0]], feats[2])

    # Features to remove from skew analysis
    remove_feat = remove_feat + low_pearson_feat + mean_log_feat + power_feat

    # Remaining numerical features can be transformed to address skew
    numeric_features = [x for x in numeric_features if x not in remove_feat]

    # Transform the skewed numeric features by taking log(feature + 1).
    # This will make the features more normal.

    print 'Correct for data skew\n'

    # Make sure we are performing transformations only on training data, we don't want data leakage of future information

    skewed_features = dt.df.loc[:1459][numeric_features].apply(lambda x: skew(x.dropna().astype(float)))

    # This is a hyperparameter that can be tuned (0.75 most common on kaggle), .6 identifies the best division between whether to log transform depending on the pearson correl coeff as shown in the jupyter notebook PlotsPreTransformedData

    skewed_features = skewed_features[abs(skewed_features) > 0.75]

    skewed_features = skewed_features.index

    dt.df[skewed_features] = np.log1p(dt.df[skewed_features])


    # Log transform y values
    dt.y = np.log1p(dt.y)

    return dt

def standarization(dt):
    """
    Here we standardize numerical features that are not binary. Since binary features aren't meaningful with respect to sizes or shapes, moreover we would lose interpretations that can be made from the dummy predictors
    """

    print 'Standarize features\n'

    from sklearn.preprocessing import StandardScaler

    dt.df_pretransform = dt.df.copy()

    # look to see if there is a way to mask or do something easier
    features_stand = [x for x in dt.df.columns.unique() if x not in dt.binary_feat]

    # Make array of remaining feature values to standardize
    X_array      = dt.df.loc[:1459][features_stand].get_values()
    X_pred_array = dt.df.loc[1460:][features_stand].get_values()

    # Standarize
    scaler = StandardScaler(with_mean=True).fit(X_array)
    X_scaled      = scaler.transform(X_array)
    X_pred_scaled = scaler.transform(X_pred_array)

    # Reload onto data transporter
    df_X        = pd.DataFrame(data=X_scaled, index=None, columns=features_stand, copy=True)
    df_X_pred   = pd.DataFrame(data=X_pred_scaled, index=None, columns=features_stand, copy=True)

    df_standarized = pd.concat([df_X, df_X_pred],  ignore_index=True)
    df_binary      = dt.df[dt.binary_feat]
    dt.df = pd.concat([df_standarized, df_binary], axis=1, copy=True)

    dt.df_train = dt.df.loc[:1459]
    dt.df_test  = dt.df.loc[1460:]

    # dt.X contains the training data set
    dt.X      = dt.df_train.get_values()
    dt.X_pred = dt.df_test.get_values()

    # dt.df_train.columns[dt.df_train.max() == 1] old way of getting binary features

    return dt

def load_data():
    """
    Load data and add constructed variables to dataframe
    """
    print "Loading data to data transporter object\n"

    data_location = ['../data/train.csv', '../data/test.csv']
    dt = data_transporter(data_location)
    dt = missing_values(dt)
    dt = load_new_features(dt)
    dt = map_ordinal_feat(dt)
    dt = encode_labels(dt)
    dt = remove_missing_features(dt)
    dt = preprocessing(dt)
    dt = standarization(dt)

    # Features that we are going to test on
    dt.features = dt.df_train.columns.tolist()
    dt.n_features = dt.df_train.shape[1]

    print "Done Loading Data\n"
    print ""

    return dt

if __name__ == '__main__':

    all_data = load_data()

    file_name = 'processed_data.pkl'

    with open(file_name,'wb') as fileObject:
        pickle.dump(all_data, fileObject)
