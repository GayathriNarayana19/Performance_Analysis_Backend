import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from PIL import Image

'''#Hardcoded here, will be given as inputs in cmdline later
data = []
csv_files = ['/Users/gayyeg01/Documents/JUPYTER NOTEBOOK FILES/separate_csv_files/CPU51.csv', '/Users/gayyeg01/Documents/JUPYTER NOTEBOOK FILES/separate_csv_files/CPU55.csv']
for file in csv_files:

    df = pd.read_csv(file)
    data.append(df)
'''    
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

    ax.set_xlim(0, 120)  # Adjust xlim for horizontal chart
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))

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
    output_path = os.path.join(output_dir, f'{plot_num}.png')
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
    #ax.set_xlim(0, 1)  # Adjust xlim for horizontal chart
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    
    '''percentage_differences = calculate_percentage_difference(max_values, min_values)
    for i, diff in enumerate(percentage_differences):
        annotate_x = max(max_values[i], min_values[i]) + 0
        ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)'''
    ax.legend()
    custom_legend = ax.legend(loc='upper right', title="Cores")
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.png')
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
#    elif key_string == 'rate_per_instruction': #will change if limit/ y axis labels are messed. 
 #       fig, ax = plt.subplots(figsize=(32,24))
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
        ax.set_xlim(0, 40)  # xlim for horizontal chart
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.5))
        #for label in ax.get_yticklabels():
         # label.set_rotation(0)
         # label.set_horizontalalignment('right')

    if key_string == 'read_rate|write_rate':
        ax.set_xlim(0, 75)  # xlim for horizontal chart
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
    
#    plt.tight_layout(rect=[0.2, 0, 1, 1])  # Adjust left margin if needed
    custom_legend = ax.legend(loc='upper right', title="Cores")
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.png')
    plt.savefig(output_path)
    plt.close(fig)
    #plt.show()

#Include Name_1 column and extract corresponding event_1 values to generate bar plots.
def filter_and_plot_Name_1(key_strings, plot_num, data, output_dir, scenario, context):
  
        
      #To club two Name_1 key_strings in the same plot
      key_strings_len = len(key_strings)
      if key_strings_len > 1:
        #filtered_data = [df[df['Name_1'].str.contains(key)].any(axis=0) for df in data for key in key_strings]
        filtered_data = [df[df['Name_1'].str.contains(key_strings[0]) | df['Name_1'].str.contains(key_strings[1])] for df in data]
        formatted_title = f'Bar Plot for Metrics with "{key_strings[0]},{key_strings[1]}" - {context}'
      else:
        for key in key_strings:
          filtered_data = [df[df['Name_1'].str.contains(key)] for df in data]
          formatted_title = f'Bar Plot for Metrics with "{key}"- {context}'
      for df in filtered_data:
            df['Event_1'] = pd.to_numeric(df['Event_1'], errors='coerce')
      #for key in key_strings:
        # Filter data to include only rows with 'Name_1' matching the key string
        #filtered_data = [df[df['Name_1'] == key] for df in data]

        #filtered_data = [df[df['Name_1'].str.contains(key)] for df in data]
        # Deduplicate based on 'Name_1' and keep only the first occurrence
      for i, df in enumerate(filtered_data):
          df.drop_duplicates(subset=['Name_1'], keep='first', inplace=True)
      #to avoid errors when event is missing
      #if len(filtered_data) == 0 or len(filtered_data[0]) == 0:
        #print(f"No data found for '{', '.join(key)}'. Skipping. . . .")
      # return
      #print(f"Key: {key}")
      #print(f"Filtered Data for Key {key}:")
      plt.figure(plot_num)
      fig, ax = plt.subplots(figsize=(20, 16))
      bar_width = 0.14  
      indices = np.arange(len(filtered_data[0]))
      #ue_values = ['CPU51', 'CPU55']
      

      for i, df in enumerate(filtered_data):
          unique_values = df['Name_1'].unique()
          #print("Unique Values in 'Name_1' Column:", unique_values)
          #print(f"CSV {i + 1}:")
          #print(df)
          bars = ax.barh(  
              indices + i * bar_width,
              df['Event_1'],  # Use values from "Event_1" column
              height=bar_width,  
              label=f'{scenario[i]}',
          )
          for j, bar in enumerate(bars):
            value = df['Event_1'].iloc[j]  # Get the value of the metric
            if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = max(value,0.01)  # Keep the text at the end of the bar
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                    f'{value:.2f}', 
                    #f'{round(value, 2)}',
                    #ha='left', va='center', fontsize=9, color='black')
                    ha='left', va='center', fontsize=10, color='black', fontweight='bold')

      # Create lists to store max and min values for each metric
      max_values = [0] * len(filtered_data[0])
      min_values = [float('inf')] * len(filtered_data[0])
      metric_names = []
      for i, df in enumerate(filtered_data):
          metric_name = df['Name_1'].iloc[0]
          values = df['Event_1'] #Event_1/Event_2 values are wrong. 

          for j, value in enumerate(values):
              if value > max_values[j]:
                  max_values[j] = value
              if value < min_values[j]:
                  min_values[j] = value

          metric_names.append(metric_name)

      #formatted_title = f'Bar Plot for Metrics with "{key}"'
      ax.set_ylabel('Event_1', fontsize=14)
      ax.set_xlabel('Values', fontsize=14)  # Adjust x-axis label to match "Event_1" column
      ax.set_title(formatted_title, fontsize=16)
      ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
      ax.set_yticklabels(filtered_data[0]['Name_1'], rotation=0, ha='right', fontsize=14)  # Adjust labels for y-axis
     
      percentage_differences = calculate_percentage_difference(max_values, min_values)
      '''for i, diff in enumerate(percentage_differences):
          annotate_x = max(max_values[i], min_values[i]) + 5
          #annotate_x = max(max_values[i], min_values[i]) + 0.5 * (max(max_values[i], min_values[i]) - min(max_values[i], min_values[i]))
          #annotate_x = min(max_values[i], min_values[i]) + cluster_width / 2
          ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)'''
          
          
          
      ax.legend()
      custom_legend = ax.legend(loc='upper right', title="Cores")   
      #custom_legend = ax.legend(loc='upper right', title="3 CELL DIFFERENT UE CASES")
      ax.add_artist(custom_legend)
      plt.tight_layout()
      output_path = os.path.join(output_dir, f'{plot_num}.png')
      plt.savefig(output_path)
      plt.close(fig)
      # Display the values on top of the bars
      
      #plt.show()
 #   plt.tight_layout()
 #   plt.show()
def merge_pngs_to_pdf(input_dir, output_file):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: The input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    # List and sort PNG files
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    png_files.sort()
    
    if not png_files:
        print(f"Error: No PNG files found in the directory '{input_dir}'.")
        sys.exit(1)
    
    # Load and convert images to RGB
    images = []
    for png_file in png_files:
        png_path = os.path.join(input_dir, png_file)
        try:
            image = Image.open(png_path).convert('RGB')
            images.append(image)
        except Exception as e:
            print(f"Error processing '{png_path}': {e}")
            sys.exit(1)
    
    # Save images as a single PDF
    try:
        images[0].save(output_file, save_all=True, append_images=images[1:])
        print(f"Successfully created the PDF '{output_file}'")
    except Exception as e:
        print(f"Error saving the PDF to '{output_file}': {e}")
        sys.exit(1) 

def main():
    usage = ("plotting_perf_stat.py --csv /path_to_csv/CSV1 /path_to_csv/CSV2 -o /dir_path_for_output_plots/ -s Scenario_for_CSV1 Scenario_for_CSV2 -c CSV1_CSV2_COMPARISON\n"
    "Enter 'python3 plotting_perf_stat.py -h' to know the description for arguments -o, -s and -c")
    parser = argparse.ArgumentParser(usage=usage, description="Process some CSV files and generate plots.")
    #parser = argparse.ArgumentParser()
    parser.add_argument("--csv_files", "-csv", nargs="+", type=str, help="Path to the CSV file")
    parser.add_argument("--op_dir", "-o", type=str, help="Path to the directory to store the plots")
    parser.add_argument("--scenario", "-s", type=str, nargs="+", help="Custom Scenarios (default: Core1, Core2)")
    parser.add_argument("--context", "-c", type=str, help="Specify context - This will appear as 'title' in output plot")
    #parser.add_argument("--cores", "-c", type=str, nargs="+", help="Custom Scenarios (default: [3UE, 16UE, 32UE, 64UE, 128UE])")
    args = parser.parse_args()
    #print(args)
    if not args.csv_files:
        parser.error("Please provide the path to the CSV file using --csv_file or -csv")
    if not args.op_dir:
        parser.error("Please provide the path to the output directory using --op_dir or -o")
    if not os.path.exists(args.op_dir):
      os.makedirs(args.op_dir)
      print(f"Created new output directory : {args.op_dir}")
    csv_files = args.csv_files
    data = []
   
    for file in csv_files:
      df = pd.read_csv(file)
      data.append(df)
    # Remove data that has Event_2 value as 0.
    #for df in data:
     # df.drop(df[df[df.columns[5]] == 0].index, inplace=True)

    filter_and_plot('rate_per_instruction', 1, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('MPKI', 2, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('IPC', 3, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('read_rate|write_rate', 4, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('eviction_rate', 5, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('stall_rate', 6, data, args.op_dir, args.scenario, args.context)
    #filter_and_plot('miss', 7, data, args.op_dir, args.scenario)
    filter_and_plot('rate_over_time', 8, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('misses_per_context_swap', 9, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('exclusive_store', 10, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('walk_rate', 11, data, args.op_dir, args.scenario, args.context)
    
    filter_and_plot_miss("1m", data, args.op_dir, args.scenario, args.context)
    filter_and_plot_tlb("1t", data, args.op_dir, args.scenario, args.context)
      
    filter_and_plot_Name_1(['TLB_REFILL'], 11, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['CACHE_REFILL'], 12, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['RETIRED'], 13, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['CACHE_RD', 'CACHE_WR'],14, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['SPEC'], 15, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['CACHE_WB_VICTIM'], 16, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['WALK'], 17, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['MEM_ACCESS'], 18, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['STALL'], 19, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_1(['EXC_TAKEN','CACHE_MISS'], 20, data, args.op_dir, args.scenario, args.context)
      #plot_number = plot_number + 1
      #data_per_core = []
    output_pdf_path = os.path.join(args.op_dir, 'merged_output.pdf')
    merge_pngs_to_pdf(args.op_dir, output_pdf_path)
if __name__ == '__main__':
    main()
