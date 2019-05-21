# Author: Akshara Balachandra
# Date: 05/15/2019
# Description: Connectome modeling class

from sklearn.model_selection import StratifiedKFold
import pickle
import pandas as pd
import numpy as np
from os import path
import sys

class ConnectomeModel:

    def __init__(self, connectome_dir, neuropsych_dir, verbose = False):

        self.connectome_dir = connectome_dir
        self.neuropsych_dir = neuropsych_dir
        self.kfold = False
        self.verbose = verbose
        self.train = None
        self.test = None
        self.folds = None
        self.n_folds = 5

    def read_impair(self, neuropsych_file, col_name, limit_strength = True, connectome = True):
        """Reads cognitive data from the given file.

        Reads the cognitive data from a csv file for the given test. Subjects are automatically
        screened for 3T or 1.5T imaging. By default, the subject list is limited to those who also
        have good diffusion data (for connectome based analyses).

        Args:
            1. colname              -- name of test column
            2. limit_strength       -- True (default): limit to 3T, False: include 1.5T & 3T
            3. connectome           -- True (default): limit to only those with good diffusion;
                                       False: all patients with neuropsych

        Returns:
            Tuple containing subject labels and list of subjects included.
        """

        neuropsych_data = pd.read_csv(self.neuropsych_dir + '/' + neuropsych_file, index_col = 0)

        # limit scanner strength
        if limit_strength:
            neuropsych_data = neuropsych_data[neuropsych_data['Magnet Strength'] == 3.0]

        labels = neuropsych_data[col_name].dropna()
        np_subjs = list(labels.index)

        common_subjs = np_subjs
        if connectome == 'connectome':
            common_subjs = [subj for subj in np_subjs if subj in DIFF_SUBJS]


        labels = labels.astype(int)

        if self.verbose:
            print(len(common_subjs))
            print(len(labels))
            print(sum(labels))
            print(list(labels))

        self.labels = labels
        self.subjects = common_subjs

        return (labels, common_subjs)

    def read_data(self, measure, temp_subnet = True,
            temporal_connections_file = '../temporal_connections_only.pickle',
            tract_file = 'WM_tracts.csv', tract_regex = 'fiber_FA*',
            hcv_file = 'Hipp_Vol.csv', icv_colname = 'ICV',
            lhcv_name = 'Left Hippocampus', rhcv_name = 'Right Hippocampus'):
        """Read imaging data.

        Reads the imaging data for the given measure. Returns a numpy array with the data
        subjects (rows) by imaging data (columns).

        Args:
            1. measure                   -- name of imaging modality (connectome, tract, hcv, clinical)
            2. temp_subnet               -- subset the connectome to temporal connections (Default: True)
            3. temporal_connections_file -- filename for temporal connections
            4. tract_file                -- filename for tract-based measures
            5. tract_regex               -- tract names regex
            6. hcv_file                  -- filename for HCV measures (not normalized!)
            7. icv_colname               -- column name for ICV
            8. lhcv_name                 -- column name for left HCV
            9. rhcv_name                 -- column name for right HCV

        Returns:
            A numpy array (n_subjects, n_features) with the features that will be included in the model.
        """
        # do connectome based predictions
        if measure == 'connectome':
            return self._read_connectomes(temporal_connections_file, temp_subnet)
        # do hcv based predictions
        elif measure == 'hcv':
            return self._read_hcv_data(hcv_file, icv_colname, lhcv_name, rhcv_name)
        # do tract-based predictions
        elif measure == 'tract':
            return self._read_tract_data(tract_file, tract_regex)

        # do clinical variable based predictions
        return self._read_clinical_data()

    def split_data(self, kfold = False, num_folds = 5):
        """Split the dataset into train/test or k-folds.

        Splits the dataset into a train/test paradigm based on ucsd vs ucsf or into k-folds for k-fold validation scheme.

        Args:
            1. kfold      -- Do k-fold (True) or train/test (False). Default: True
            2. num_folds  -- number of folds to do. Default: 5

        Returns:
            Tuple of train/test indices or list of tuples of train/test indices
        """
        self.kfold = kfold

        if kfold:
            return self._create_k_split(num_folds, shuffle = True)
        return self._split_ucsd_ucsf()

    ###############################################
    #########   PRIVATE FUNCTIONS   ###############
    ###############################################

    def _read_connectomes(self, temporal_connections_file, temp_subnet = True):
        # read in all connectomes
        connectomes = []
        reshaped = []

        for subj in self.subjects:
            conn = pd.read_csv(path.join(self.connectome_dir,
                '{}_norm.csv'.format(subj)), index_col= 0).values
            connectomes.append(conn)

        connectomes = np.stack(connectomes)

        for subj in connectomes:
            reshaped.append(self._upper_tri_masking(subj))

        # each subject now has  98 * 49 - 49 number of features
        reshaped = np.array(reshaped)

        if temp_subnet:
            temp_connections = pickle.load(
                    open(temporal_connections_file, 'rb'))
            reshaped = reshaped[:,temp_connections]

        return reshaped

    def _read_tract_data(self, tract_file, regex):
        tract_df = pd.read_csv(path.join(NEUROPSYCH_DIR, tract_file), index_col = 0)
        tract_df = tract_df.loc[self.subjects]

        return tract_df.filter(regex = regex).values

    def _read_hcv_data(self, hcv_file, icv_colname, lhcv_name, rhcv_name):
        hcv_df = pd.read_csv(path.join(NEUROPSYCH_DIR, hcv_file),
                index_col = 0)
        hcv_df = hcv_df.loc[self.subjects]
        hcv_df['Norm_Hipp.L'] = hcv_df[lhcv_name] / hcv_df[icv_colname]
        hcv_df['Norm_Hipp.R'] = hcv_df[rhcv_name] / hcv_df[icv_colname]
        return hcv_df.filter(regex = 'Norm_Hipp').values

    def _upper_tri_masking(self, A):
        m = A.shape[0]
        r = np.arange(m)
        mask = r[:,None] < r
        return A[mask]

    def _create_k_split(self, k = 10, shuffle = True):
        self.folds = StratifiedKFold(k, shuffle).split(self.subjects, self.labels)
        self.n_folds = k

        return k,self.folds

    def _split_ucsd_ucsf(self):
        ucsf_ind = [i for i,k in enumerate(self.subjects) if 'ucsf' in k]
        ucsd_ind = [i for i in range(len(self.subjects)) if i not in ucsf_ind]

        self.train = ucsd_ind
        self.test = ucsf_ind

        return ucsd_ind, ucsf_ind
