import argparse
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PyPDF2 import PdfMerger

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def plot_clustered_bar(df, column, title, ax):
    """Plot a clustered bar graph for the specified column."""
    bar_plot = df[column].plot(kind='bar', ax=ax, width=0.8)
    legend_labels = ['N1', 'G2', 'G3']
    ax.legend(legend_labels, title='50th Percentile', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Index')  # Update the xlabel since there's no BG, LS, and CB_LEN
    ax.set_ylabel('Values')
    ax.set_title(title)
    
    # Extract columns
    n1_column = df.iloc[:, 0]  # Update the column index as per your data
    g3_column = df.iloc[:, -1]  # Update the column index as per your data

    # Calculate percentage difference
    percent_difference = ((n1_column - g3_column) / n1_column) * 100
    pd_val = percent_difference.iloc[0]
    # Find the highest bar
    max_height = max(bar_plot.patches, key=lambda x: x.get_height()).get_height()
    ax.annotate(f'{pd_val:.2f}%', 
                xy=(0, max_height), 
                xytext=(0, 1),  # Offset of the annotation from the top of the bar
                textcoords="offset points",
                ha='center', va='bottom', 
                fontsize=10,
                color='black')

def extract_row_as_new_row_from_csv_files(csv_files, row_number):
    # Initialize an empty list to store dictionaries of column names and values
    column_data = []
    # Loop through each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Extract the specified row
        selected_row = df.iloc[row_number]
        
        # Split the row into column names and values
        columns = selected_row.index
        values = selected_row.values
        
        # Append column names and values to the list
        for column, value in zip(columns, values):
            column_data.append({'Column Name': column, 'Value': value})
    
    # Create DataFrame from the list of dictionaries
    column_names_values_df = pd.DataFrame(column_data)
    return column_names_values_df

def calculate_metrics(df, metrics):
    # Initialize a list to store the metric information
    metric_info = []
    # Iterate over each metric
    for metric_name, metric_columns in metrics:
        # Extract column names for the current metric
        column1, column2 = metric_columns

        # Get the values corresponding to the two columns for each row
        values1 = df.loc[df['Column Name'] == column1, 'Value'].tolist()
        values2 = df.loc[df['Column Name'] == column2, 'Value'].tolist()
        graph_xlabel = f"{metric_name}={column1}/{column2}"

        # Calculate the metric for each pair of values
        for value1, value2 in zip(values1, values2):
            # Calculate the metric
            if value2 != 0:  # To avoid division by zero
                metric_value = value1 / value2
            else:
                metric_value = float('nan')  # Handling division by zero
            
            # Append metric information to the list
            metric_info.append({
                'Metrics': metric_name,
                'Name_1': column1,
                'Event_1': value1,
                'Name_2': column2,
                'Event_2': value2,
                'Event_1/Event_2': metric_value,
                'Graph_Xlabel': graph_xlabel
            })
    
    # Convert the list of metric information into a DataFrame
    metric_df = pd.DataFrame(metric_info)
    #print(metric_df)
    return metric_df

def create_full_paths(base_dirs, kpi_file_groups):
    full_paths = []
    for dir_info in base_dirs:
        dir_path = dir_info['path']
        file_groups = [
            [os.path.join(dir_path, filename) for filename in file_group]
            for file_group in kpi_file_groups
        ]
        full_paths.append(file_groups)
    return full_paths

#Calculate percentage difference between max and min bars in a cluster. 
def calculate_percentage_difference(max_vals, min_vals):
    percentage_differences = []
    for max_val, min_val in zip(max_vals, min_vals):
        if min_val == 0:
            percentage_difference = float('inf')  # Handle division by zero
        else:
            percentage_difference = ((max_val - min_val) / min_val) * 100
        percentage_differences.append(percentage_difference)
    return percentage_differences

#Include only metrics containing "miss" but not "TLB" in it as a keyword/keystring
def filter_and_plot_miss(plot_num, data, output_dir, scenario, context):
    filtered_data = [df[df['Metrics'].str.contains('miss') & ~df['Metrics'].str.contains('TLB')] for df in data]

    for df in filtered_data:
        #df['Event_1/Event_2'] *= 100
        df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 100

    plt.figure(plot_num)
    fig, ax = plt.subplots(figsize=(18, 12))

    bar_width = 0.14  # Adjust spacing

    # Array of indices for the y-coordinates of bars
    indices = np.arange(len(filtered_data[0]))
    
    #Hardcoded jor jupyter
    #ue_values = ['CPU51', 'CPU55']
    
    # Use ax.barh for horizontal bars & use height instead of width
    for i, df in enumerate(filtered_data):
        bars = ax.barh(  
            indices + i * bar_width,
            df['Event_1/Event_2'],
            height=bar_width,  
            label=f'{scenario[i]}',
        )
        for j, bar in enumerate(bars):
            value = df['Event_1/Event_2'].iloc[j]  # Get the value of the metric
            if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = value  # Keep the text at the end of the bar
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                    f'{value:.2f}', 
                    #f'{round(value, 2)}',
                    #ha='left', va='center', fontsize=9, color='black')
                    ha='left', va='center', fontsize=9, color='black', fontweight='bold')
    # Lists to store max and min values of the bar cluster of every metric 
    max_values = [0] * len(filtered_data[0])
    min_values = [float('inf')] * len(filtered_data[0])
    metric_names = []
    for i, df in enumerate(filtered_data):
        metric_name = df['Metrics'].iloc[0]
        values = df['Event_1/Event_2']

        for j, value in enumerate(values):
            if value > max_values[j]:
                max_values[j] = value
            if value < min_values[j]:
                min_values[j] = value

        metric_names.append(metric_name)

    formatted_title = f'Bar Plot for "miss" Metrics (Excluding "TLB")- {context}'
    ax.set_ylabel('Metrics', fontsize=14)
    ax.set_xlabel('Rate in Percentage', fontsize=14)
    ax.set_title(formatted_title,fontsize=16)
    ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    # y-axis labels
    ax.set_yticklabels(filtered_data[0]['Graph_Xlabel'], rotation=0, ha='right', fontsize=14)  

    ax.set_xlim(0, 20)  # Adjust xlim for horizontal chart
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    # Calculate and annotate the percentage difference based on max and min values on the plot
    '''percentage_differences = calculate_percentage_difference(max_values, min_values)
    for i, diff in enumerate(percentage_differences):
        # Annotate or display to the right side of the max value
        annotate_x = max(max_values[i], min_values[i]) + 0
        ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)

    #for i, diff in enumerate(percentage_differences):
        #ax.annotate(f'{diff:.2f}%', (indices[i] + bar_width / 2, max(max_values[i], min_values[i])), ha='center')
'''
    ax.legend()
    custom_legend = ax.legend(loc='lower right', title="Cores")
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.pdf')
    plt.savefig(output_path)
    plt.close(fig)
    
# Include only metrics containing "TLB" but not "context_swap"
def filter_and_plot_tlb(plot_num, data, output_dir, scenario, context):
    # Include only metrics containing "TLB"
    #filtered_data = [df[df['Metrics'].str.contains('TLB')] for df in data]
    
    filtered_data = [df[df['Metrics'].str.contains('TLB') & ~df['Metrics'].str.contains('context_swap')] for df in data]
    #print(filtered_data)
    for df in filtered_data:
        #df['Event_1/Event_2'] *= 100
        df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 100
    #print(df['Event_1/Event_2'])
    plt.figure(plot_num)
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Adjust spacing
    bar_width = 0.14 

    # Array of indices for the y-coordinates of bars
    indices = np.arange(len(filtered_data[0]))


    for i, df in enumerate(filtered_data):
        bars=ax.barh(  
            indices + i * bar_width,
            df['Event_1/Event_2'],
            height=bar_width,  
            label=f'{scenario[i]}',
        )
        for j, bar in enumerate(bars):
            value = df['Event_1/Event_2'].iloc[j]  # Get the value of the metric
            if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = value  # Keep the text at the end of the bar
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                    f'{value:.2f}', 
                    #f'{round(value, 2)}',
                    #ha='left', va='center', fontsize=9, color='black')
                    ha='left', va='center', fontsize=9, color='black', fontweight='bold')
    max_values = [0] * len(filtered_data[0])
    min_values = [float('inf')] * len(filtered_data[0])
    metric_names = []

    for i, df in enumerate(filtered_data):
        metric_name = df['Metrics'].iloc[0]
        values = df['Event_1/Event_2']

        for j, value in enumerate(values):
            if value > max_values[j]:
                max_values[j] = value
            if value < min_values[j]:
                min_values[j] = value
        metric_names.append(metric_name)

    formatted_title = f'Bar Plot for "TLB" Metrics - {context}'
    ax.set_ylabel('Metrics', fontsize=14)
    ax.set_xlabel('Rate in Percentage', fontsize=14)
    ax.set_title(formatted_title, fontsize=16)
    ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    #y-axis labels
    ax.set_yticklabels(filtered_data[0]['Graph_Xlabel'], rotation=0, ha='right', fontsize=14)  
    ax.set_xlim(0, 7)  # Adjust xlim for horizontal chart
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    
    '''percentage_differences = calculate_percentage_difference(max_values, min_values)
    for i, diff in enumerate(percentage_differences):
        annotate_x = max(max_values[i], min_values[i]) + 0
        ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)'''
    ax.legend()
    custom_legend = ax.legend(loc='upper right', title="Cores")
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.pdf')
    plt.savefig(output_path)
    plt.close(fig)
    
# Include only metrics containing "rate_per_instruction"
def filter_and_plot(key_string, plot_num, data, output_dir, scenario, context):
   
    filtered_data = [df[df['Metrics'].str.contains(key_string)] for df in data]
    #To not throw an error if the metric is not present in csv. May be applicable for N2.
    #if len(filtered_data) == 0 or len(filtered_data[0]) == 0:
      #print(f"No data found for '{', '.join(key_string)}'. Skipping. . . .")
    #return

    if key_string == 'rate_per_instruction' or key_string == 'miss' or key_string == 'read_rate|write_rate' or key_string == 'walk_rate' or key_string == 'eviction_rate' or key_string == 'stall_rate' or key_string == 'rate_over_time' or key_string == 'exclusive_store':
        for df in filtered_data:
            #df['Event_1/Event_2'] *= 100
            df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 100
    elif key_string == 'MPKI':
        for df in filtered_data:
            #df['Event_1/Event_2'] *= 1000
            df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 1000
    else:
        for df in filtered_data:
            df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce')

    plt.figure(plot_num)
    if key_string == 'miss':
        fig, ax = plt.subplots(figsize=(20, 16))
    else:
        fig, ax = plt.subplots(figsize=(16, 12))

    bar_width = 0.14  
    indices = np.arange(len(filtered_data[0]))

    for i, df in enumerate(filtered_data):
        bars = ax.barh( 
            indices + i * bar_width,
            df['Event_1/Event_2'],
            height=bar_width,  
            label=f'{scenario[i]}',
        )
        for j, bar in enumerate(bars):
            value = df['Event_1/Event_2'].iloc[j]  # Get the value of the metric
            if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = value  # Keep the text at the end of the bar
            if key_string == "walk_rate":
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                        f'{value:.6f}', 
                        #f'{round(value, 2)}',
                        #ha='left', va='center', fontsize=9, color='black')
                        ha='left', va='center', fontsize=9, color='black', fontweight='bold')
            else:
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                        f'{value:.2f}', 
                        #f'{round(value, 2)}',
                        #ha='left', va='center', fontsize=9, color='black')
                        ha='left', va='center', fontsize=9, color='black', fontweight='bold')

    max_values = [0] * len(filtered_data[0])
    min_values = [float('inf')] * len(filtered_data[0])
    metric_names = []
    for i, df in enumerate(filtered_data):
        metric_name = df['Metrics'].iloc[0]
        values = df['Event_1/Event_2']
        #df['Event_1/Event_2'] = df['Event_1/Event_2'].astype(float)
        #print(values)
        #values = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 1

        for j, value in enumerate(values):
            if value > max_values[j]:
                max_values[j] = value
            if value < min_values[j]:
                min_values[j] = value

        metric_names.append(metric_name)

    formatted_title = f'Bar Plot for Metrics with "{key_string}" - {context}'
    ax.set_ylabel('Metrics', fontsize=14)
    if key_string == 'MPKI':
        ax.set_xlabel('MPKI', fontsize=14)
    elif key_string == 'misses_per_context_swap' or key_string == 'miss':
        ax.set_xlabel('Number of Misses')
    elif key_string == 'IPC':
        ax.set_xlabel('IPC', fontsize=14)
    else:
        ax.set_xlabel('Rate in Percentage',fontsize=14)
    ax.set_title(formatted_title,fontsize=16)
    ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    #y-axis labels
    ax.set_yticklabels(filtered_data[0]['Graph_Xlabel'], rotation=0, ha='right', fontsize=14)  
    if key_string == 'IPC':
        ax.set_xlim(0,3)  # xlim for horizontal chart

        #ideal_value = 4
        #ax.axvline(ideal_value, color='red', linestyle='--', label=f'Ideal: {ideal_value}', linewidth=2)
        '''ax.annotate(
            f'Ideal: {ideal_value}',
            xy=(ideal_value + 0.1, 0.02),  #text pos can be adjusted
            color='red',
            fontsize=12,
            fontweight='bold',
        )'''
        #ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))

    if key_string == 'rate_per_instruction':
        ax.set_xlim(1, 60)  # xlim for horizontal chart
        ax.xaxis.set_major_locator(plt.MultipleLocator(3))

    if key_string == 'read_rate|write_rate':
        ax.set_xlim(1, 75)  # xlim for horizontal chart
        ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))

    '''percentage_differences = calculate_percentage_difference(max_values, min_values)
    #cluster_width = bar_width * len(ue_values)  # Calculate the total width of each cluster
    for i, diff in enumerate(percentage_differences):
        annotate_x = max(max_values[i], min_values[i]) + 0
        #annotate_x = max(max_values[i], min_values[i]) + 0.5 * (max(max_values[i], min_values[i]) - min(max_values[i], min_values[i]))

        # Find the position to print/annotate at the center of the cluster
        #annotate_x = min(max_values[i], min_values[i]) + cluster_width / 2
        ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)
'''
    ax.legend()
    custom_legend = ax.legend(loc='upper right', title="Cores")
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.pdf')
    plt.savefig(output_path)
    plt.close(fig)


def merge_pdfs(input_dir, output_file):
    pdf_merger = PdfMerger()

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    pdf_files.sort()  

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        pdf_merger.append(pdf_path)

    with open(output_file, 'wb') as output_pdf:
        pdf_merger.write(output_pdf)
    pdf_merger.close()   

def main():
    epilog = (
        'Examples:\n'
        '  pmuv3_terminal.py -config config.yaml\n'
    )

    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write(f'error: {message}\n')
            self.print_help()
            sys.stderr.write(f'\n{epilog}\n')
            sys.exit(2)

    parser = CustomArgumentParser(
        description='Process CSV files and generate plots.',
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-config',
        metavar='CONFIG',
        type=str,
        required=True,
        help='Path to the YAML configuration file. Example: -config config.yaml'
    )

    args = parser.parse_args()

    config = load_config(args.config)

    base_dirs = config.get('base_dirs')
    output_dir = config.get('output_dir')
    base_filename = config.get('base_filename')
    num_bundles = config.get('num_bundles')
    kpi_file_groups = config["kpi_file_groups"]
    kpi_metrics = config["kpi_metrics"]
    scenarios = config["scenarios"]
    context = config["context"]

    if len(base_dirs) != len(scenarios):
        print("ERROR: Scenarios must be same as number of output files")
        exit()
    
    if not base_dirs or not output_dir or not base_filename or num_bundles is None or context is None:
        parser.error('Missing required arguments in the configuration file.')
    
    merged_dfs = []
    bundles_cols = []

    for bundle_num in range(num_bundles):
        dfs = []
        for base_dir_info in base_dirs:
            base_dir = base_dir_info.get('path')
            if not base_dir:
                parser.error('Missing base directory path in the configuration file.')

            filename = os.path.join(base_dir, base_filename.format(bundle_num))
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            
        merged_df = pd.concat(dfs, axis=1)
        merged_dfs.append(merged_df)
        bundles_cols.append(dfs[0].columns)
    
    #Plot events graph
    for bundle_num, merged_df in enumerate(merged_dfs):
        bundle_cols = bundles_cols[bundle_num]
        fig, axs = plt.subplots(len(bundle_cols), 1, figsize=(10, 25))
        for i, column in enumerate(bundle_cols):
            plot_clustered_bar(merged_df, column, f'Clustered Bar Graph - {column}', axs[i])

        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}bundle{bundle_num}.pdf')
        #plt.show()
        plt.close(fig)   
    
    #Make list of files to parse to compute KPI
    kpi_files_list = create_full_paths(base_dirs, kpi_file_groups)
    row_number = 0    
    kpi_metrics_data = []

    for idx, base_dir_info in enumerate(base_dirs):
        path = base_dir_info['path']
        output_file = base_dir_info['output_file']
        
        kpi_files = kpi_files_list  # Use kpi_files_list directly
        
        instance_kpi_metrics_df = pd.DataFrame()
        
        for i, (csv_files, metrics) in enumerate(zip(kpi_files[idx], kpi_metrics)):
            result_df = extract_row_as_new_row_from_csv_files(csv_files, row_number)
            metrics_df = calculate_metrics(result_df, kpi_metrics[i])
            instance_kpi_metrics_df = pd.concat([instance_kpi_metrics_df, metrics_df], ignore_index=True)
        
        # Save to CSV
        instance_kpi_metrics_df.to_csv(f'{output_dir}{output_file}', index=False, na_rep='NaN')
        kpi_metrics_data.append(instance_kpi_metrics_df)
        #n_kpi_metrics_df = pd.concat([n_kpi_metrics_df, instance_kpi_metrics_df], ignore_index=True)
    
    #Plot KPIs
    filter_and_plot('rate_per_instruction', 1, kpi_metrics_data, output_dir, scenarios, context)
    filter_and_plot('MPKI', 2, kpi_metrics_data, output_dir, scenarios, context)
    filter_and_plot('IPC', 3, kpi_metrics_data, output_dir, scenarios, context)
    filter_and_plot('stall_rate', 4, kpi_metrics_data, output_dir, scenarios, context)

    filter_and_plot_miss("1m", kpi_metrics_data, output_dir, scenarios, context)
    filter_and_plot_tlb("2t", kpi_metrics_data, output_dir, scenarios, context)
    output_pdf_path = os.path.join(output_dir, 'merged_output.pdf')
    merge_pdfs(output_dir, output_pdf_path) 
    

if __name__ == "__main__":
    main()
