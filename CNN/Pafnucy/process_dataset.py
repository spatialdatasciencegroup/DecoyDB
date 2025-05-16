import numpy as np
import pandas as pd
import h5py


import pybel
from tfbio.data import Featurizer

import warnings

import matplotlib.pyplot as plt
import seaborn as sns

path = './data/v2013'

%%bash -s $path --out missing

path=$1

# Save binding affinities to csv file

echo 'pdbid,-logKd/Ki' > affinity_data.csv
cat $path/PDBbind_2016_plain_text_index/index/INDEX_general_PL_data.2016 | while read l1 l2 l3 l4 l5; do
    if [[ ! $l1 =~ "#" ]]; then
        echo $l1,$l4
    fi
done >> affinity_data.csv


# Find affinities without structural data (i.e. with missing directories)

cut -f 1 -d ',' affinity_data.csv | tail -n +2 | while read l;
    do if [ ! -e $path/general-set-except-refined/$l ] &&  [ ! -e $path/refined-set/$l ]; then
        echo $l;
    fi
done

affinity_data = pd.read_csv('affinity_data.csv', comment='#')
affinity_data = affinity_data[~np.in1d(affinity_data['pdbid'], list(missing))]
affinity_data.head()

# Separate core, refined, and general sets

core_set = ! grep -v '#' $path/PDBbind_2016_plain_text_index/index/INDEX_core_data.2016 | cut -f 1 -d ' '
core_set = set(core_set)

refined_set = ! grep -v '#' $path/PDBbind_2016_plain_text_index/index/INDEX_refined_data.2016 | cut -f 1 -d ' '
refined_set = set(refined_set)

general_set = set(affinity_data['pdbid'])


assert core_set & refined_set == core_set
assert refined_set & general_set == refined_set

len(general_set), len(refined_set), len(core_set)

# Exclude v 2013 core set - it will be used as another test set

core2013 = ! cat core_pdbbind2013.ids
core2013 = set(core2013)

affinity_data['include'] = True
affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core2013 & (general_set - core_set))), 'include'] = False

affinity_data.loc[np.in1d(affinity_data['pdbid'], list(general_set)), 'set'] = 'general'

affinity_data.loc[np.in1d(affinity_data['pdbid'], list(refined_set)), 'set'] = 'refined'

affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core_set)), 'set'] = 'core'

affinity_data.head()

affinity_data[['pdbid']].to_csv('pdb.ids', header=False, index=False)
affinity_data[['pdbid', '-logKd/Ki', 'set']].to_csv('affinity_data_cleaned.csv', index=False)

dataset_path = {'general': 'general-set-except-refined', 'refined': 'refined-set', 'core': 'refined-set'}

%%bash -s $path

# Prepare pockets with UCSF Chimera - pybel sometimes fails to calculate the charges.
# Even if Chimera fails to calculate several charges (mostly for non-standard residues),
# it returns charges for other residues.

path=$1

for dataset in general-set-except-refined refined-set; do
    echo $dataset
    for pdbfile in $path/$dataset/*/*_pocket.pdb; do
        mol2file=${pdbfile%pdb}mol2
        if [[ ! -e $mol2file ]]; then
            echo -e "open $pdbfile \n addh \n addcharge \n write format mol2 0 tmp.mol2 \n stop" | chimera --nogui
            # Do not use TIP3P atom types, pybel cannot read them
            sed 's/H\.t3p/H    /' tmp.mol2 | sed 's/O\.t3p/O\.3  /' > $mol2file
        fi
    done 
done > chimera_rw.log


featurizer = Featurizer()
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

with h5py.File('%s/core2013.hdf' % path, 'w') as g:
    j = 0

    for dataset_name, data in affinity_data.groupby('set'):

        print(dataset_name, 'set')
        i = 0
        ds_path = dataset_path[dataset_name]


        with h5py.File('%s/%s.hdf' % (path, dataset_name), 'w') as f:
            for _, row in data.iterrows():

                name = row['pdbid']
                affinity = row['-logKd/Ki']

                ligand = next(pybel.readfile('mol2', '%s/%s/%s/%s_ligand.mol2' % (path, ds_path, name, name)))
                # do not add the hydrogens! they are in the strucutre and it would reset the charges

                try:
                    pocket = next(pybel.readfile('mol2', '%s/%s/%s/%s_pocket.mol2' % (path, ds_path, name, name)))
                    # do not add the hydrogens! they were already added in chimera and it would reset the charges
                except:
                    warnings.warn('no pocket for %s (%s set)' % (name, dataset_name))
                    continue

                ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
                assert (ligand_features[:, charge_idx] != 0).any()
                pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
                assert (pocket_features[:, charge_idx] != 0).any() 

                centroid = ligand_coords.mean(axis=0)
                ligand_coords -= centroid
                pocket_coords -= centroid

                data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                                       np.concatenate((ligand_features, pocket_features))), axis=1)

                if row['include']:
                    dataset = f.create_dataset(name, data=data, shape=data.shape, dtype='float32', compression='lzf')
                    dataset.attrs['affinity'] = affinity
                    i += 1
                else:
                    dataset = g.create_dataset(name, data=data, shape=data.shape, dtype='float32', compression='lzf')
                    dataset.attrs['affinity'] = affinity
                    j += 1

        print('prepared', i, 'complexes')
    print('excluded', j, 'complexes')
    
with h5py.File('%s/core.hdf' % path, 'r') as f, \
    h5py.File('%s/core2013.hdf' % path, 'r+') as g:
for name in f:
    if name in core2013:
        dataset = g.create_dataset(name, data=f[name])
        dataset.attrs['affinity'] = f[name].attrs['affinity']
# we assume that PDB IDs are unique
assert ~protein_data['pdbid'].duplicated().any()

protein_data = protein_data[np.in1d(protein_data['pdbid'], affinity_data['pdbid'])]

# check for missing values
protein_data.isnull().any()

for idx, row in protein_data[protein_data['name'].isnull()].iterrows():
    uniprotid = row['uniprotid'][:6]
    name = row['uniprotid'][7:]
    protein_data.loc[idx, ['uniprotid', 'name']] = [uniprotid, name]

protein_data.isnull().any()
protein_data.to_csv('protein_data.csv', index=False)