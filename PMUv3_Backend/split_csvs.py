import os
import sys
import pandas as pd

def unique_directory(base_path, name):
    count = 1
    new_path = os.path.join(base_path, name)
    while os.path.exists(new_path):
        new_path = os.path.join(base_path, f"{name}_{count}")
        count += 1
    return new_path

def split_and_save_csvs(base_dirs, base_filename, num_bundles):
    for base_dir in base_dirs:
        base_dir = base_dir.rstrip(os.sep)
        # Output directory for the current base directory
        base_output_dir = f"{os.path.basename(base_dir)}_split_files"
        output_dir = unique_directory(os.getcwd(), base_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for bundle_num in range(num_bundles):
            filename = os.path.join(base_dir, base_filename.format(bundle_num))
            if not os.path.exists(filename):
                print(f"File {filename} does not exist. Skipping...")
                continue

            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()

            first_column_name = df.columns[0]
            grouped = df.groupby(first_column_name)

            for name, group in grouped:
                # Create a directory named after the first column's unique value within the output directory
                split_output_dir = os.path.join(output_dir, f"{name}_split")
                os.makedirs(split_output_dir, exist_ok=True)

                # Drop first column
                group = group.drop(columns=[first_column_name])
                output_filename = os.path.join(split_output_dir, f"bundle{bundle_num}.csv")
                group.to_csv(output_filename, index=False)
                print(f"Saved {output_filename} in {split_output_dir}")

if __name__ == "__main__":
    # Example usage:
    # python split_csvs.py base_dir1 base_dir2 ... base_dirN
    if len(sys.argv) < 2:
        print("Usage: python split_csvs.py <base_dir1> <base_dir2> ... <base_dirN>")
        sys.exit(1)

    base_dirs = sys.argv[1:]  # List of base directories passed as arguments

    # Define your base_filename and num_bundles as per your requirement
    base_filename = "bundle{}.csv"  # Replace with your filename pattern
    num_bundles = 15  # Set the number of bundles you have

    split_and_save_csvs(base_dirs, base_filename, num_bundles)

