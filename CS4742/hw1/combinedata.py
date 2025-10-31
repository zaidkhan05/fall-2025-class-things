#script to combine data from multiple sources
import pandas as pd
def combine_data(file_list, output_file):
    combined_df = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df.to_csv(output_file, index=False)
if __name__ == "__main__":
    files_to_combine = ['test.csv', 'train.csv']
    combine_data(files_to_combine, 'combined_data.csv')