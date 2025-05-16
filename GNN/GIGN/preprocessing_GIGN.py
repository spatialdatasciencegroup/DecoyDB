# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# %%
def generate_pocket(data_dir, distance=5):
    complex_id = os.listdir(data_dir)
    for cid in complex_id:
        try:
            # print(cid)
            complex_dir = os.path.join(data_dir, cid)
            if(len(os.listdir(complex_dir))==0):
                continue
            lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
            protein_path= os.path.join(complex_dir, f"{cid}_protein.pdb")
    
            if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
                continue
            # print(pocket.GetNumAtoms())
            pymol.cmd.load(protein_path)
            pymol.cmd.remove('resn HOH')
            pymol.cmd.load(lig_native_path)
            pymol.cmd.remove('hydrogens')
            pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
            pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
            pymol.cmd.delete('all')
        except:
            continue

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        try:
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            try:
                if(len(os.listdir(complex_dir))==0):
                    continue
            except FileNotFoundError:
                print(complex_dir)
                continue
            pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
            if input_ligand_format != 'pdb':
                ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
                ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
                os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
            else:
                ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')
    
            save_path = os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")
            ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
            if ligand == None:
                print(f"Unable to process ligand of {cid}")
                continue
    
            pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
            # print(pocket.GetNumAtoms()+ligand.GetNumAtoms())
            conf = pocket.GetConformer()

            if pocket == None:
                print(f"Unable to process protein of {cid}")
                continue
    
            complex = (ligand, pocket)
            with open(save_path, 'wb') as f:
                pickle.dump(complex, f)
        except:
            continue
        pbar.update(1)

if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'mol2'
    data_root = './data/v2020-other-PL/'
    data_dir = data_root
    data_df = pd.read_csv(os.path.join('./data', 'alldata.csv'))

    ## generate pocket within 5 Ångström around ligand 
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)



# %%
