# # %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import time
from multiprocessing import Pool
# %%
def generate_pocket(data_dir, distance=5,ligand_dir=None,protein_name=None):
    os.makedirs(ligand_dir+"_save", exist_ok=True)
    complex_id = os.listdir(ligand_dir)


    for cid in complex_id:
        obj = cid.split('.')[0]
        save_num = obj.split('_')[-1]
        if int(save_num)>10:
            continue
        # print(obj[-3:])
        complex_dir = os.path.join(ligand_dir, cid)
        lig_native_path = complex_dir
        # lig_native_path = os.path.join(complex_dir, f"{cid}")
        protein_path= os.path.join(data_dir, f"{protein_name}_target.pdbqt")
        save_path = ligand_dir+"_save"
        if os.path.exists(os.path.join(save_path, f'Pocket_{distance}A.pdb')):
            continue
        
        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {obj} around {distance}')
        pymol.cmd.save(os.path.join(save_path, f'Pocket_{distance}A{save_num}.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2',protein_name='10GS-VWW'):
    os.makedirs(os.path.join(data_dir,protein_name)+"_savecomplex", exist_ok=True)
    cid, pKa = str(0), float(0)
    # print()
    complex_dir = os.path.join(data_dir,protein_name)
    pocket_path = os.path.join(complex_dir+'_save',f'Pocket_{distance}A{0}.pdb')
    if input_ligand_format != 'pdb':
        ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
        ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
        os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
    else:
        ligand_path = os.path.join(complex_dir, f'molecule_{cid}.pdb')
    save_path = os.path.join(complex_dir+'_savecomplex', f"{cid}_{distance}A.rdkit")
    ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    if ligand == None:
        print(f"Unable to process ligand of {cid}")
        # continue

    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
    if pocket == None:
        print(f"Unable to process protein of {cid}")
        # continue

    complex = (ligand, pocket)
    with open(save_path, 'wb') as f:
        pickle.dump(complex, f)
    # pbar = tqdm(total=10)
    
    for i, row in data_df.iterrows():

        cid, pKa = str(i+1), float(row['rmsd'])
        if i+1>10:
            break
        complex_dir = os.path.join(data_dir,protein_name)

        pocket_path = os.path.join(complex_dir+'_save',f'Pocket_{distance}A{cid}.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(complex_dir, f'molecule_{cid}.pdb')
        save_path = os.path.join(complex_dir+'_savecomplex', f"{cid}_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(complex_dir)
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        # pbar.update(1)
def find_files_with_decoys(directory):
    decoy_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'decoys' in file:
                decoy_files.append(os.path.join(root, file))
    
    return decoy_files
def process(i,path):
    distance = 5
    input_ligand_format = 'pdb'
    root_path = './PDBdata/'
    data_root = os.path.join(root_path,path)
    complex_path = find_files_with_decoys(data_root)
    for complex_name in complex_path:
        data_name = complex_name.split('_')[0]
        ligand_dir = os.path.join(data_root,data_name)

        data_df = pd.read_csv(os.path.join(data_root, f'{data_name}_decoy_scores.csv'))
    
        ## generate pocket within 5 Ångström around ligand 
        generate_pocket(data_dir=data_root, distance=distance,ligand_dir=ligand_dir,protein_name=data_name)

        generate_complex(data_root, data_df, distance=distance, input_ligand_format=input_ligand_format,protein_name=data_name)
if __name__ == '__main__':
    root_path = './PDBdata/'
    paths = os.listdir(root_path)
    with Pool(processes=80) as pool:
        pool.starmap(process, enumerate(paths))

