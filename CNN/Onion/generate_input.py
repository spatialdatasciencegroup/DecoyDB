import pandas as pd

csv_file_path = './data/train2013.csv'

df = pd.read_csv(csv_file_path)

if 'pdbid' in df.columns:
    df['path'] = df['pdbid'].apply(lambda x: f"./data/v2013/{x}/{x}_complex2.pdb")
    
    output_file = 'input.data'
    

    df['path'].to_csv(output_file, index=False, header=False)
    print("File saved successfully.")
else:
    print("Error: 'pdbid' column not found in the CSV file.")