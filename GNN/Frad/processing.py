import os
import numpy as np
import pandas as pd
from openbabel import openbabel

# Function to remove hydrogens and HETATM entries, returning atom count, coordinates, and atomic numbers
def process_pdb(file_path):
    obconversion = openbabel.OBConversion()
    obconversion.SetInFormat("pdb")

    mol = openbabel.OBMol()
    obconversion.ReadFile(mol, file_path)

    non_h_atoms = [atom for atom in openbabel.OBMolAtomIter(mol) if atom.GetAtomicNum() != 1 and not atom.IsHetAtom()]
    atom_count = len(non_h_atoms)
    coordinates = np.array([[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in non_h_atoms])
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in non_h_atoms])

    return atom_count, coordinates, atomic_numbers

def process_sdf(file_path):
    obconversion = openbabel.OBConversion()
    obconversion.SetInFormat("sdf")

    mol = openbabel.OBMol()
    obconversion.ReadFile(mol, file_path)

    non_h_atoms = [atom for atom in openbabel.OBMolAtomIter(mol) if atom.GetAtomicNum() != 1]
    atom_count = len(non_h_atoms)
    coordinates = np.array([[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in non_h_atoms])
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in non_h_atoms])

    return atom_count, coordinates, atomic_numbers

def pad_array(array, max_length, pad_value=0):
    """Pad a 2D or 3D array to the desired length along the first axis."""
    if array.ndim == 2:  # For atomic numbers
        padded = np.pad(array, ((0, max_length - array.shape[0]),), constant_values=pad_value)
    elif array.ndim == 3:  # For coordinates
        padded = np.pad(array, ((0, max_length - array.shape[0]), (0, 0)), constant_values=pad_value)
    return padded

def main(directory, csv_file):
    pdbids = []
    affinities = []

    combined_atom_counts = []
    combined_coordinates = []
    combined_atomic_numbers = []

    pocket_atom_counts = []
    ligand_atom_counts = []

    # Load affinity data
    affinity_data = pd.read_csv(csv_file)
    affinity_dict = dict(zip(affinity_data['pdbid'], affinity_data['-logKd/Ki']))

    # Filter subfolders by pdbid in the CSV file
    pdbid_list = set(affinity_data['pdbid'])

    max_atoms = 0
    num = 0
    # First pass to determine max_atoms
    for pdbid in pdbid_list:
        subfolder_path = os.path.join(directory, pdbid)
        if os.path.isdir(subfolder_path):
            pocket_file = None
            ligand_file = None

            for file in os.listdir(subfolder_path):
                if file.endswith("_pocket.pdb"):
                    pocket_file = os.path.join(subfolder_path, file)
                elif file.endswith("_ligand.sdf"):
                    ligand_file = os.path.join(subfolder_path, file)

            if pocket_file and ligand_file:
                pocket_count, _, _ = process_pdb(pocket_file)
                ligand_count, _, _ = process_sdf(ligand_file)

                max_atoms = max(max_atoms, pocket_count + ligand_count)
        if num>5000:
            break
        num+=1
        
    num = 0
    # Second pass to process data
    for pdbid in pdbid_list:
        subfolder_path = os.path.join(directory, pdbid)
        if os.path.isdir(subfolder_path):
            pocket_file = None
            ligand_file = None

            for file in os.listdir(subfolder_path):
                if file.endswith("_pocket.pdb"):
                    pocket_file = os.path.join(subfolder_path, file)
                elif file.endswith("_ligand.sdf"):
                    ligand_file = os.path.join(subfolder_path, file)

            if pocket_file and ligand_file:
                pocket_count, pocket_coords, pocket_numbers = process_pdb(pocket_file)
                ligand_count, ligand_coords, ligand_numbers = process_sdf(ligand_file)

                # Combine protein and ligand data
                combined_count = pocket_count + ligand_count
                combined_coords = np.vstack((pocket_coords, ligand_coords))
                combined_numbers = np.hstack((pocket_numbers, ligand_numbers))

                # Pad to max_atoms
                padded_coords = np.pad(combined_coords, ((0, max_atoms - combined_coords.shape[0]), (0, 0)), constant_values=0)
                padded_numbers = np.pad(combined_numbers, (0, max_atoms - combined_numbers.shape[0]), constant_values=0)

                pdbids.append(pdbid)
                affinities.append(affinity_dict.get(pdbid, np.nan))

                pocket_atom_counts.append(pocket_count)
                ligand_atom_counts.append(ligand_count)

                combined_atom_counts.append(combined_count)
                combined_coordinates.append(padded_coords)
                combined_atomic_numbers.append(padded_numbers)
        if num>5000:
            break
        num+=1
    # Create index
    indices = np.arange(len(pdbids))

    # Create dictionary
    dataset = {
        'index': indices,
        'num_atoms': np.array(combined_atom_counts),
        'charges': np.array(combined_atomic_numbers),
        'positions': np.array(combined_coordinates),
        'pocket_atoms': np.array(pocket_atom_counts),
        'ligand_atoms': np.array(ligand_atom_counts),
        'neglog_aff': np.array(affinities)
    }

    # Save the dictionary as a NumPy file
    np.save(os.path.join("./pdbbind2013", "lba_val.npy"), dataset)
    print(np.array(dataset).shape)
    print(np.array(dataset).item()['index'])
if __name__ == "__main__":
    main("./data/v2013", "./data/valid2013.csv")
