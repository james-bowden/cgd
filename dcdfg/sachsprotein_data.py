import os
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class ProteinInterventionDataset(Dataset):
    def __init__(
            self, 
            file_map, 
            base_dir, 
            intervention_targets=None, 
            fraction_regimes_to_ignore=None,
            regimes_to_ignore=None,
            load_ignored=False,
            normalize=False,
    ) -> None:
        super(ProteinInterventionDataset, self).__init__()

        # Define the protein measurement columns explicitly before calling _load_data
        self.protein_columns = ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3', 
                                'p44/42', 'pakts473', 'PKA', 'PKC', 'P38', 'pjnk']
        self.data = pd.DataFrame()  # To store combined data
        self.base_dir = base_dir
        self.intervention_targets = intervention_targets or {}
        self.file_map = file_map
        self.normalize = normalize  # Add normalize option
        
        # load data
        all_data, all_masks, all_regimes, adata = self.load_data()
        self.adata = adata
        # index of all regimes, even if not used in the regimes_to_ignore case
        self.all_regimes_list = np.unique(all_regimes)
        '''
        obs_regime = np.unique(
            all_regimes[np.where([mask == [] for mask in all_masks])[0]]
        )
        assert len(obs_regime) == 1
        obs_regime = obs_regime[0]
        '''

        if fraction_regimes_to_ignore is not None or regimes_to_ignore is not None:
            if fraction_regimes_to_ignore is not None and regimes_to_ignore is not None:
                raise ValueError("either fraction or list, not both")
            if fraction_regimes_to_ignore is not None:
                np.random.seed(0)
                '''
                # make sure observational regime is in the training, and not in the testing
                sampling_list = self.all_regimes_list[
                    self.all_regimes_list != obs_regime
                ]
                '''
                sampling_list = self.all_regimes_list
                self.regimes_to_ignore = np.random.choice(
                    sampling_list,
                    int(fraction_regimes_to_ignore * len(sampling_list)),
                )
            else:
                self.regimes_to_ignore = regimes_to_ignore
            to_keep = np.array(
                [
                    regime not in self.regimes_to_ignore
                    for regime in np.array(all_regimes)
                ]
            )
            if not load_ignored:
                data = all_data[to_keep]
                masks = [mask for i, mask in enumerate(all_masks) if to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(all_regimes) if to_keep[i]]
                )
            else:
                data = all_data[~to_keep]
                masks = [mask for i, mask in enumerate(all_masks) if ~to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(all_regimes) if ~to_keep[i]]
                )
        else:
            data = all_data
            masks = all_masks
            regimes = all_regimes

        self.data = data
        self.regimes = regimes
        self.masks = np.array(masks, dtype=object)
        self.intervention = True
        
        # Extract basic dataset properties
        self.num_regimes = len(set(self.regimes))
        self.num_samples = self.data.shape[0]
        self.dim = len(self.protein_columns)

    def load_data(self):
        """
        Load and concatenate data from multiple files, adding intervention labels and processing them as needed.
        """
        combined_data = []
        interventions = []  # To hold the intervention conditions (regimes)
        masks = []          # To hold the intervention masks

        for intervention_label, file_name in self.file_map.items():
            file_path = os.path.join(self.base_dir, file_name)
            try:
                # Load and ensure consistent protein columns across files
                data = pd.read_excel(file_path)

                # Rename 'pip2' and 'pip3' to 'PIP2' and 'PIP3' if they exist
                data.rename(columns={'pip2': 'PIP2', 'pip3': 'PIP3'}, inplace=True)

                data['intervention'] = intervention_label
                
                intervention_label_fixed = intervention_label.replace('_', '+')
                intervention_list = re.split(r'\+', intervention_label_fixed)

                # Check if "cd3" and "cd28" are present, and combine them into "cd3_cd28"
                if "cd3" in intervention_list and "cd28" in intervention_list:
                    # Remove "cd3" and "cd28" from the list and add "cd3_cd28"
                    intervention_list = [x for x in intervention_list if x not in ["cd3", "cd28"]]
                    intervention_list.append("cd3_cd28")

                # Assign the processed intervention list to the data
                data['intervention_list'] = [intervention_list] * len(data)


                # Use _create_mask to generate the mask for this intervention
                mask = self._create_mask(intervention_list)

                # Append the generated mask for each sample in the dataset
                masks.extend([mask] * len(data))
                interventions.extend([intervention_label] * len(data))

                # Append processed data
                combined_data.append(data[self.protein_columns + ['intervention', 'intervention_list']])
            except FileNotFoundError:
                print(f"File not found: {file_path}")

        # Combine all data into a single DataFrame
        adata = pd.concat(combined_data, ignore_index=True)

        # Normalize protein data if specified
        if self.normalize:
            adata[self.protein_columns] = (adata[self.protein_columns] - 
                                               adata[self.protein_columns].mean()) / adata[self.protein_columns].std()
        
        # Convert data to final format, returning masks, regimes, and full adata structure
        data = adata[self.protein_columns].values.astype(np.float32)
        regimes = np.array(interventions)

        return data, masks, regimes, adata
    
    '''
    def _create_mask(self, intervention_list):
        """
        Create a binary mask for each sample based on its intervention, considering activators and inhibitors.
        :param list intervention_list: Intervention labels for the sample.
        :return list mask: Binary mask indicating inhibited (0) and active (1) proteins.
        """
        # Create an initial mask with all proteins set to active (1)
        mask = [1] * len(self.protein_columns)

        # Iterate through the interventions and modify the mask accordingly
        for intervention in intervention_list:
            if intervention in self.intervention_targets:
                intervention_info = self.intervention_targets[intervention]
                for target in intervention_info['targets']:
                    if target in self.protein_columns:
                        target_index = self.protein_columns.index(target)
                        # Set to 0 for any intervention, regardless of whether activator or inhibitor
                        mask[target_index] = 0

        return mask
    '''
    def _create_mask(self, intervention_list):
        """
        Create a binary mask for each sample based on its intervention, considering activators and inhibitors.
        :param list intervention_list: Intervention labels for the sample.
        :return list mask: Binary mask indicating inhibited (0) and active (1) proteins.
        """
        # Create an initial mask with all proteins set to active (1)
        mask = [1] * len(self.protein_columns)

        # print(f"Intervention list: {intervention_list}")  # Debug: Print intervention list

    # Iterate through the interventions and modify the mask accordingly
        for intervention in intervention_list:
            if intervention in self.intervention_targets:
                # print(f"Processing intervention: {intervention}")  # Debug: Print current intervention being processed
                intervention_info = self.intervention_targets[intervention]
                for target in intervention_info['targets']:
                    if target in self.protein_columns:
                        target_index = self.protein_columns.index(target)
                    # Set to 0 for any intervention, regardless of whether activator or inhibitor
                        mask[target_index] = 0
                        # print(f"Setting protein '{target}' at index {target_index} to 0")  # Debug: Print which protein is being set to 0

        # print(f"Generated mask: {mask}")  # Debug: Print generated mask
        return mask


    def __getitem__(self, idx):
        """
        Return a single sample from the dataset in a unified format.
        :param int idx: Index of the sample.
        :return tuple: (data, mask, regime) where data is the protein measurements,
                       mask is the binary mask, and regime is the intervention condition.
        """
        # Extract the protein data from the NumPy array
        protein_data = self.data[idx].astype(np.float32)

        binary_mask = self.masks[idx]  # Should be the mask generated in load_data()

        # Map the regime to an integer index
        unique_regimes = sorted(list(set(self.regimes)))
        regime_mapping = {reg: i for i, reg in enumerate(unique_regimes)}
        regime = regime_mapping[self.regimes[idx]]

        return protein_data, np.array(binary_mask, dtype=np.float32), regime

    def __len__(self):
        return self.num_samples


# Test of output format on a sample
'''
# Define file mappings and intervention targets
file_map = {
    "cd3_cd28": "1. cd3cd28.xls",
    "cd3_cd28_icam2": "2. cd3cd28icam2.xls",
    "cd3_cd28+aktinhib": "3. cd3cd28+aktinhib.xls",
    "cd3_cd28+g0076": "4. cd3cd28+g0076.xls",
    "cd3_cd28+psitect": "5. cd3cd28+psitect.xls",
    "cd3_cd28+u0126": "6. cd3cd28+u0126.xls",
    "cd3_cd28+ly": "7. cd3cd28+ly.xls",
    "pma": "8. pma.xls",
    "b2camp": "9. b2camp.xls",
    "cd3_cd28_icam2+aktinhib": "10. cd3cd28icam2+aktinhib.xls",
    "cd3_cd28_icam2+g0076": "11. cd3cd28icam2+g0076.xls",
    "cd3_cd28_icam2+psit": "12. cd3cd28icam2+psit.xls",
    "cd3_cd28_icam2+u0126": "13. cd3cd28icam2+u0126.xls",
    "cd3_cd28_icam2+ly": "14. cd3cd28icam2+ly.xls",
}

intervention_targets = {
    'cd3_cd28': {'type': 'activator', 'targets': ['ZAP70', 'Lck', 'plcg', 'praf', 'pmek', 'Erk', 'PKC']},
    'icam2': {'type': 'activator', 'targets': ['LFA-1']},
    'b2camp': {'type': 'activator', 'targets': ['PKA']},
    'pma': {'type': 'activator', 'targets': ['PKC']},
    'aktinhib': {'type': 'inhibitor', 'targets': ['pakts473']},
    'g0076': {'type': 'inhibitor', 'targets': ['PKC']},
    'psitect': {'type': 'inhibitor', 'targets': ['P38']},
    'u0126': {'type': 'inhibitor', 'targets': ['pmek', 'Erk']},
    'ly': {'type': 'inhibitor', 'targets': ['PI3K', 'pakts473']}
}

# Initialize the dataset
dataset = ProteinInterventionDataset(file_map, base_dir='data/2005_sachs_protein', intervention_targets=intervention_targets, normalize=True)

# Access a sample to test
print(dataset[8600])
'''