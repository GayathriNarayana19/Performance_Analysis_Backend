import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os
from PIL import Image
import seaborn as sns
import matplotlib.font_manager as fm
import textwrap
mpl.rcParams['figure.max_open_warning'] = 30
'''#Hardcoded here, will be given as inputs in cmdline later
data = []
csv_files = ['/Users/gayyeg01/Documents/JUPYTER NOTEBOOK FILES/separate_csv_files/CPU51.csv', '/Users/gayyeg01/Documents/JUPYTER NOTEBOOK FILES/separate_csv_files/CPU55.csv']
for file in csv_files:

    df = pd.read_csv(file)
    data.append(df)
'''

def dynamic_wrap_width(label, max_width=40, min_width=20):
    return min(max_width, max(len(label) // 2, min_width))

def wrap_after_newline_and_dash(label):
    if 'bound' in label:
        if '\n' in label:
            first_part, second_part = label.split('\n', 1)

            wrap_width = dynamic_wrap_width(second_part)

            if '-' in second_part:
                before_dash, after_dash = second_part.split('-', 1)
                
                before_dash_wrapped = textwrap.fill(before_dash, wrap_width)
                after_dash_wrapped = textwrap.fill(after_dash, wrap_width)
                
                second_part_wrapped = f'{before_dash_wrapped} - {after_dash_wrapped}'
            else:
                second_part_wrapped = textwrap.fill(second_part, wrap_width)
            
            return f'{first_part}\n{second_part_wrapped}'
    return label    
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
    if len(filtered_data) == 0:
        print("No data to plot.")
        return

    # Seaborn color palette
    try:
        sns_palette = sns.color_palette("Set2")  
        #colors = [sns_palette[2], sns_palette[0]]  # Choose green and blue from the palette
        colors = [sns_palette[1],sns_palette[2],sns_palette[0],sns_palette[6]]  
        colors = colors[:len(filtered_data)]
        #colors = sns.color_palette("muted", len(filtered_data))  # Ensure 'colors' is defined
#        colors = ['#228B22', '#8B4513']  # Green and brown hex codes
    except Exception as e:
        print(f"Error creating color palette: {e}")
        return
    for df in filtered_data:
        #df['Event_1/Event_2'] *= 100
        df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 100

    plt.figure(plot_num)
    fig, ax = plt.subplots(figsize=(18, 12))
    #fig, ax = plt.subplots(figsize=(18, len(filtered_data[0]) * 0.6))  # Increase height based on number of metrics
    # Extract Y-axis labels and shorten them by keeping only the part before '='
    #short_labels = [label.split('=')[0] for label in filtered_data[0]['Graph_Xlabel']]
    # Update Y-axis labels to split at '=' into two lines
    short_labels = [label.split('=')[0] + '\n' + label.split('=')[1] for label in filtered_data[0]['Graph_Xlabel']]

    # Set color cycle before plotting
    ax.set_prop_cycle('color', colors)
 #   bar_width = 0.14  # Adjust spacing
   
    # Array of indices for the y-coordinates of bars
  #  indices = np.arange(len(filtered_data[0]))
    indices = np.arange(len(filtered_data[0]['Graph_Xlabel']))
    bar_width = 0.2 
    #Hardcoded jor jupyter
    #ue_values = ['CPU51', 'CPU55']
    # Use ax.barh for horizontal bars & use height instead of width
    max_value = 0
    for i, df in enumerate(filtered_data):
        bars = ax.barh(  
            indices + i * bar_width,
            df['Event_1/Event_2'],
            height=bar_width,  
            label=f'{scenario[i]}',
            alpha=0.8,  # Add transparency
            #edgecolor='black',  # Add black borders for better contrast
            linewidth=1.5,  # Thicker edges for better visibility
            capstyle='round'  # Rounded bar edges
        )
        max_value = max(max_value, df['Event_1/Event_2'].max())

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
                    ha='left', va='center', fontsize=12, color='black', fontweight='bold')
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
#Seaborn changes
    colors = sns.color_palette("muted", len(filtered_data))  # Custom palette
    formatted_title = f'Bar Plot for "miss" Metrics (Excluding "TLB")- {context}'
    ax.set_ylabel('Metrics', fontsize=14, weight='bold')
    ax.set_xlabel('Rate in Percentage', fontsize=14, weight='bold')
    ax.set_title(formatted_title,fontsize=16, weight='bold')
    ax.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
    ax.tick_params(axis='x', labelsize=14, labelcolor='black', width=2, length=5)  # Customize x-axis numbers' size and color
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set major ticks to integers if needed
    #ax.set_facecolor('lightgrey')
    ax.set_facecolor('#F0F0F0')
    ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    # y-axis labels
    ax.set_yticklabels(short_labels, rotation=0, ha='right', fontsize=14)  
    #for label in ax.get_yticklabels():
     #   label.set_fontproperties(fm.FontProperties(weight='bold'))
    # Adding gridlines for clarity
    ax.grid(axis='x', linestyle='--', linewidth=2, alpha=0.7, color='white')
    #ax.set_xlim(0, 120)  # Adjust xlim for horizontal chart
    #Dynamically adjust x-axis limits based on the max value
    ax.set_xlim(0, max_value * 1.1) 
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
    #ax.legend()
    #custom_legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title='Cores', title_fontsize=16)

    custom_legend = ax.legend(loc='lower right', title="Cores",fontsize=12, title_fontsize=14, frameon=True)
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
# Include only metrics containing "TLB" but not "context_swap"
def filter_and_plot_tlb(plot_num, data, output_dir, scenario, context):
    # Include only metrics containing "TLB"
    #filtered_data = [df[df['Metrics'].str.contains('TLB')] for df in data]
    
    filtered_data = [df[df['Metrics'].str.contains('TLB') & ~df['Metrics'].str.contains('context_swap')] for df in data]
    #print(filtered_data)
    short_labels = [label.split('=')[0] + '\n' + label.split('=')[1] for label in filtered_data[0]['Graph_Xlabel']]
    if len(filtered_data) == 0:
        print("No data to plot.")
        return

    # Seaborn color palette
    try:
        sns_palette = sns.color_palette("Set2")  # A good balanced palette
      #  colors = [sns_palette[2], sns_palette[0]]  # Choose green and blue from the palette
        colors = [sns_palette[1],sns_palette[2],sns_palette[0],sns_palette[6]]  # Choose green and blue from the palette
        colors = colors[:len(filtered_data)]
        #colors = sns.color_palette("muted", len(filtered_data))  # Ensure 'colors' is defined
#        colors = ['#228B22', '#8B4513']  # Green and brown hex codes
    except Exception as e:
        print(f"Error creating color palette: {e}")
        return

    for df in filtered_data:
        #df['Event_1/Event_2'] *= 100
        df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') * 100
    #print(df['Event_1/Event_2'])
    plt.figure(plot_num)
    fig, ax = plt.subplots(figsize=(16, 12))
    #fig, ax = plt.subplots(figsize=(16, len(filtered_data[0]) * 0.6))  # Increase height based on number of metrics
    ax.set_prop_cycle('color', colors) 
    # Adjust spacing
    bar_width = 0.2

    # Array of indices for the y-coordinates of bars
    indices = np.arange(len(filtered_data[0]))


    for i, df in enumerate(filtered_data):
        bars=ax.barh(  
            indices + i * bar_width,
            df['Event_1/Event_2'],
            height=bar_width,  
            label=f'{scenario[i]}',
        )
        max_value = max(df['Event_1/Event_2'].max() for df in filtered_data)
        for j, bar in enumerate(bars):
            value = df['Event_1/Event_2'].iloc[j]  # Get the value of the metric
            '''if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = value  # Keep the text at the end of the bar'''
            
            # Set the maximum position for text to avoid going out of the grid
            if bar.get_width() > max_value * 0.95:  # If the bar width is close to the right edge
                x_pos = bar.get_width() - 0.05 * max_value  # Move text slightly inside the bar
                ha_value = 'right'  # Align text to the right
            else:
                x_pos = bar.get_width()  # Default x position (bar width)
                ha_value = 'left'  # Align text to the left for shorter bars
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                    f'{value:.2f}', 
                    #f'{round(value, 2)}',
                    #ha='left', va='center', fontsize=9, color='black')
                    ha='left', va='center', fontsize=12, color='black', fontweight='bold')
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

    colors = sns.color_palette("muted", len(filtered_data))  # Custom palette
    formatted_title = f'Bar Plot for "TLB" Metrics - {context}'
    ax.set_ylabel('Metrics', fontsize=14, weight='bold')
    ax.set_xlabel('Rate in Percentage', fontsize=14, weight='bold')
    ax.set_title(formatted_title, fontsize=16, weight = 'bold')
    ax.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
    ax.tick_params(axis='x', labelsize=14, labelcolor='black', width=2, length=5)  # Customize x-axis numbers' size and color
    ax.set_xlim(0, max_value * 1.1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set major ticks to integers if needed
    ax.set_facecolor('#F0F0F0')
    ax.grid(axis='x', linestyle='--', linewidth=2, alpha=0.7, color='white')
    ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    #y-axis labels
    ax.set_yticklabels(short_labels, rotation=0, ha='right', fontsize=14)  
    #ax.set_xlim(0, 1)  # Adjust xlim for horizontal chart
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    
    '''percentage_differences = calculate_percentage_difference(max_values, min_values)
    for i, diff in enumerate(percentage_differences):
        annotate_x = max(max_values[i], min_values[i]) + 0
        ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)'''
    ax.legend()
    custom_legend = ax.legend(loc='upper right', title="Cores",fontsize=12, title_fontsize=14, frameon=True)
    ax.add_artist(custom_legend)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    
# Include only metrics containing "rate_per_instruction"
def filter_and_plot(key_string, plot_num, data, output_dir, scenario, context):
   
    filtered_data = [df[df['Metrics'].str.contains(key_string)] for df in data]
    short_labels = [label.split('=')[0] + '\n' + label.split('=')[1] for label in filtered_data[0]['Graph_Xlabel']]
    if len(filtered_data) == 0:
        print("No data to plot.")
        return

    # Seaborn color palette
    try:
        sns_palette = sns.color_palette("Set2")  # A good balanced palette
        #colors = [sns_palette[2], sns_palette[0]]  # Choose green and blue from the palette
        colors = [sns_palette[1],sns_palette[2],sns_palette[0],sns_palette[6]]  # Choose green and blue from the palette
        colors = colors[:len(filtered_data)]
        #colors = sns.color_palette("muted", len(filtered_data))  # Ensure 'colors' is defined
#        colors = ['#228B22', '#8B4513']  # Green and brown hex codes
    except Exception as e:
        print(f"Error creating color palette: {e}")
        return

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
    #elif key_string == 'bound':
        #filtered_data = [df[df['Metrics'].str.contains('frontend_bound') | df['Metrics'].str.contains('backend_bound')] for df in data]
        #for df in filtered_data:
            #df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') 

    # Handle retiring separately
    #elif key_string == 'retiring':
        #for df in filtered_data:
            #df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce') 

    else:
        for df in filtered_data:
            df['Event_1/Event_2'] = pd.to_numeric(df['Event_1/Event_2'], errors='coerce')

    plt.figure(plot_num)
    if key_string == 'miss':
        fig, ax = plt.subplots(figsize=(20, 16))
        #fig, ax = plt.subplots(figsize=(20, len(filtered_data[0]) * 0.6))  # Increase height based on number of metrics
#    elif key_string == 'rate_per_instruction': #will change if limit/ y axis labels are messed. 
 #       fig, ax = plt.subplots(figsize=(32,24))
    else:
        fig, ax = plt.subplots(figsize=(16, 12))
        #fig, ax = plt.subplots(figsize=(16, len(filtered_data[0]) * 0.6))  # Increase height based on number of metrics
    ax.set_prop_cycle('color', colors)

    bar_width = 0.2
    indices = np.arange(len(filtered_data[0]))

    for i, df in enumerate(filtered_data):
        if key_string == 'retiring' or 'bound' in key_string:
          bars = ax.bar( 
              indices + i * bar_width,
              df['Event_1/Event_2'],
              width=bar_width,  
              label=f'{scenario[i]}',
          )
          for j, bar in enumerate(bars):
            value = df['Event_1/Event_2'].iloc[j]
            x_pos = bar.get_x() + bar.get_width() / 2  # Center of the bar
            y_pos = bar.get_height()  # Height of the bar (top position)
            
            ax.text(x_pos, y_pos, f'{value:.2f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

        # Swap labels: x-axis becomes y-axis and vice versa
          ax.set_xlabel('Metrics', fontsize=14, weight='bold')  # Now the vertical axis
          ax.set_ylabel('Rate in percentage', fontsize=14, weight='bold')
          ax.set_xticks(indices + i * bar_width)
          ax.set_xticklabels(short_labels, rotation=0, ha='right', fontsize=14)  # Short labels on the x-axis
          y_max_value = max(df['Event_1/Event_2'].max() for df in filtered_data)
          ax.set_ylim(0, y_max_value * 1.1)  # Adjust y-axis limit slightly above the maximum value
          ax.set_yticks(np.arange(0, y_max_value + 1, step=1))  # Set tick values (adjust step based on data range)
        else:
          bars = ax.barh( 
              indices + i * bar_width,
              df['Event_1/Event_2'],
              height=bar_width,  
              label=f'{scenario[i]}',
          )
        #max_value = max(df['Event_1/Event_2'].max() for df in filtered_data if df['Event_1/Event_2'].notnull().all())

          for j, bar in enumerate(bars):
            value = df['Event_1/Event_2'].iloc[j]  # Get the value of the metric
            '''# Set the maximum position for text to avoid going out of the grid
            if bar.get_width() > max_value * 0.95:  # If the bar width is close to the right edge
                x_pos = bar.get_width() - 0.05 * max_value  # Move text slightly inside the bar
                ha_value = 'right'  # Align text to the right
            else:
                x_pos = bar.get_width()  # Default x position (bar width)
                ha_value = 'left'  # Align text to the left for shorter bars'''
            if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = value  # Keep the text at the end of the bar
            if key_string == "walk_rate":
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                        f'{value:.6f}', 
                        #f'{round(value, 2)}',
                        #ha='left', va='center', fontsize=9, color='black')
                        ha='left', va='center', fontsize=12, color='black', fontweight='bold')
            else:
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                        f'{value:.2f}', 
                        #f'{round(value, 2)}',
                        #ha='left', va='center', fontsize=9, color='black')
                        ha='left', va='center', fontsize=12, color='black', fontweight='bold')

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
    colors = sns.color_palette("muted", len(filtered_data))  # Custom palette

    formatted_title = f'Bar Plot for Metrics with "{key_string}" - {context}'
    ax.set_ylabel('Metrics', fontsize=14, weight = 'bold')
    if key_string == 'MPKI':
        ax.set_xlabel('MPKI', fontsize=14, weight = 'bold')
    elif key_string == 'misses_per_context_swap' or key_string == 'miss':
        ax.set_xlabel('Number of Misses', fontsize=14, weight = 'bold')
    elif key_string == 'IPC':
        ax.set_xlabel('IPC', fontsize=14, weight = 'bold')
    else:
        ax.set_xlabel('Rate in Percentage',fontsize=14, weight = 'bold')
    
    if key_string == "retiring" or "bound" in key_string:
      ax.tick_params(axis='y', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
      ax.tick_params(axis='y', labelsize=14, labelcolor='black', width=2, length=5)  # Customize y-axis numbers' size and color
    #ax.set_xlim(0, max_value * 1.1)
      ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set major ticks to integers if needed
      ax.set_facecolor('#F0F0F0')
      ax.grid(axis='y', linestyle='--', linewidth=2, alpha=0.7, color='white')
      ax.set_title(formatted_title,fontsize=16, weight = 'bold')
      ax.set_xticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    #x-axis labels
      wrapped_labels = [wrap_after_newline_and_dash(label) for label in short_labels]
      ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=14) 
    else:
      ax.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
      ax.tick_params(axis='x', labelsize=14, labelcolor='black', width=2, length=5)  # Customize x-axis numbers' size and color
    #ax.set_xlim(0, max_value * 1.1)
      ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set major ticks to integers if needed
      ax.set_facecolor('#F0F0F0')
      ax.grid(axis='x', linestyle='--', linewidth=2, alpha=0.7, color='white')
      ax.set_title(formatted_title,fontsize=16, weight = 'bold')
      ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
    #y-axis labels
      ax.set_yticklabels(short_labels, rotation=0, ha='right', fontsize=14)  
    if key_string == 'IPC':
        ax.set_xlim(0,3)  # xlim for horizontal chart
        ax.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
        ax.tick_params(axis='x', labelsize=14, labelcolor='black', width=2, length=5)  # Customize x-axis numbers' size and color
        ax.set_facecolor('#F0F0F0')
        ax.grid(axis='x', linestyle='--', linewidth=2, alpha=0.7, color='white')
        
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
        ax.set_xlim(0, 56)  # xlim for horizontal chart
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
    if key_string == "retiring" or "bound" in key_string:
      ax.legend()
      y_max_value = max(df['Event_1/Event_2'].max() for df in filtered_data)
      
      ax.set_xlabel('Metrics', fontsize=14, weight='bold')  # Now the vertical axis
      ax.set_ylabel('Rate in percentage', fontsize=14, weight='bold')
      ax.set_ylim(0, y_max_value * 1.1)  # Adjust y-axis limit slightly above the maximum value
      ax.set_yticks(np.arange(0, y_max_value + 1, step=1))  # Set tick values (adjust step based on data range)
      y_max_value = max(df['Event_1/Event_2'].max() for df in filtered_data)
      ax.set_ylim(0, y_max_value * 1.1)  # Adjust y-axis limit slightly above the maximum value
      ax.set_yticks(np.arange(0, y_max_value + 1, step=1))  # Set tick values (adjust step based on data range)
      ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
      custom_legend = ax.legend(loc='upper right', title="Cores",fontsize=12, title_fontsize=14, frameon=True)
      ax.add_artist(custom_legend)
      plt.tight_layout()
    else:
      ax.legend()
      ax.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
      ax.tick_params(axis='x', labelsize=14, labelcolor='black', width=2, length=5)  # Customize x-axis numbers' size and color
      ax.set_facecolor('#F0F0F0')
      ax.grid(axis='x', linestyle='--', linewidth=2, alpha=0.7, color='white')
#    plt.tight_layout(rect=[0.2, 0, 1, 1])  # Adjust left margin if needed
      custom_legend = ax.legend(loc='upper right', title="Cores",fontsize=12, title_fontsize=14, frameon=True)
      ax.add_artist(custom_legend)
      plt.tight_layout()
    output_path = os.path.join(output_dir, f'{plot_num}.png')
    plt.savefig(output_path)
    plt.close(fig)
    #plt.show()

#Include Name_1 column and extract corresponding event_1 values to generate bar plots.
def filter_and_plot_Name_n(plot_num, data, output_dir, scenario, context, num, key_strings=None):
    # Seaborn color palette
      try:
        sns_palette = sns.color_palette("Set2")  # A good balanced palette
        colors = [sns_palette[1],sns_palette[2],sns_palette[0],sns_palette[6]]  # Choose green and blue from the palette
        #colors = sns.color_palette("muted", len(filtered_data))  # Ensure 'colors' is defined
#        colors = ['#228B22', '#8B4513']  # Green and brown hex codes
      except Exception as e:
        print(f"Error creating color palette: {e}")
        return
  
      #To club two Name_1 key_strings in the same plot
      key_strings_len = len(key_strings) if key_strings is not None else 0
      if key_strings_len > 1:
        #filtered_data = [df[df['Name_1'].str.contains(key)].any(axis=0) for df in data for key in key_strings]
        filtered_data = [df[df[f'Name_{num}'].str.contains(key_strings[0]) | df[f'Name_{num}'].str.contains(key_strings[1])] for df in data]
        formatted_title = f'Bar Plot for Metrics with "{key_strings[0]},{key_strings[1]}" - {context}'
      elif key_strings is None:
        filtered_data = [df[df[f'Name_{num}'].notna()] for df in data]
        formatted_title = f'Bar Plot for Metrics with "SLOT"- {context}'
      else:
        for key in key_strings:
          filtered_data = [df[df[f'Name_{num}'].str.contains(key)] for df in data]
          formatted_title = f'Bar Plot for Metrics with "{key}"- {context}'
      for df in filtered_data:
            df[f'Event_{num}'] = pd.to_numeric(df[f'Event_{num}'], errors='coerce')
      #for key in key_strings:
        # Filter data to include only rows with 'Name_1' matching the key string
        #filtered_data = [df[df['Name_1'] == key] for df in data]

        #filtered_data = [df[df['Name_1'].str.contains(key)] for df in data]
        # Deduplicate based on 'Name_1' and keep only the first occurrence
      
      colors = colors[:len(filtered_data)]
      for i, df in enumerate(filtered_data):
          df.drop_duplicates(subset=[f'Name_{num}'], keep='first', inplace=True)
      #to avoid errors when event is missing
      #if len(filtered_data) == 0 or len(filtered_data[0]) == 0:
        #print(f"No data found for '{', '.join(key)}'. Skipping. . . .")
      # return
      #print(f"Key: {key}")
      #print(f"Filtered Data for Key {key}:")
      plt.figure(plot_num)
      fig, ax = plt.subplots(figsize=(20, 16))
      ax.set_prop_cycle('color', colors)
      bar_width = 0.2  
      indices = np.arange(len(filtered_data[0]))
      #ue_values = ['CPU51', 'CPU55']
      

      for i, df in enumerate(filtered_data):
          unique_values = df[f'Name_{num}'].unique()
          #print("Unique Values in 'Name_1' Column:", unique_values)
          #print(f"CSV {i + 1}:")
          #print(df)
          bars = ax.barh(  
              indices + i * bar_width,
              df[f'Event_{num}'],  # Use values from "Event_1" column
              height=bar_width,  
              label=f'{scenario[i]}',
              alpha=0.8,  # Add transparency
              linewidth=1.5,
              capstyle='round'
          )
          max_value = max(df[f'Event_{num}'].max() for df in filtered_data)
          for j, bar in enumerate(bars):
            value = df[f'Event_{num}'].iloc[j]  # Get the value of the metric
            '''
            if value < 0:
                x_pos = value + 0.8  # Move the text inside the bar
            else:
                x_pos = max(value,0.01)  # Keep the text at the end of the bar'''
            # Set the maximum position for text to avoid going out of the grid
            if bar.get_width() > max_value * 0.95:  # If the bar width is close to the right edge
                x_pos = bar.get_width() - 0.05 * max_value  # Move text slightly inside the bar
                ha_value = 'right'  # Align text to the right
            else:
                x_pos = bar.get_width()  # Default x position (bar width)
                ha_value = 'left'  # Align text to the left for shorter bars
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, 
                    f'{value:.2f}', 
                    #f'{round(value, 2)}',
                    #ha='left', va='center', fontsize=9, color='black')
                    ha='left', va='center', fontsize=12, color='black', fontweight='bold')

      # Create lists to store max and min values for each metric
      max_values = [0] * len(filtered_data[0])
      min_values = [float('inf')] * len(filtered_data[0])
      metric_names = []
      for i, df in enumerate(filtered_data):
          metric_name = df[f'Name_{num}'].iloc[0]
          values = df[f'Event_{num}'] #Event_1/Event_2 values are wrong. 

          for j, value in enumerate(values):
              if value > max_values[j]:
                  max_values[j] = value
              if value < min_values[j]:
                  min_values[j] = value

          metric_names.append(metric_name)
      colors = sns.color_palette("muted", len(filtered_data)) 
      #formatted_title = f'Bar Plot for Metrics with "{key}"'
      ax.set_ylabel(f'Event_{num}', fontsize=14, weight = 'bold')
      ax.set_xlabel('Values', fontsize=14, weight = 'bold')  # Adjust x-axis label to match "Event_1" column
      ax.set_title(formatted_title, fontsize=16, weight = 'bold')
      
      ax.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)  # Increase font size and width of ticks
      ax.tick_params(axis='x', labelsize=14, labelcolor='black', width=2, length=5)  # Customize x-axis numbers' size and color
      ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set major ticks to integers if needed
      ax.set_facecolor('#F0F0F0')   
      ax.set_yticks(indices + (bar_width + 0.0) * (len(filtered_data) - 1) / 2)
      ax.set_yticklabels(filtered_data[0][f'Name_{num}'], rotation=0, ha='right', fontsize=14)  # Adjust labels for y-axis
      ax.grid(axis='x', linestyle='--', linewidth=2, alpha=0.7, color='white')
    
      percentage_differences = calculate_percentage_difference(max_values, min_values)
      '''for i, diff in enumerate(percentage_differences):
          annotate_x = max(max_values[i], min_values[i]) + 5
          #annotate_x = max(max_values[i], min_values[i]) + 0.5 * (max(max_values[i], min_values[i]) - min(max_values[i], min_values[i]))
          #annotate_x = min(max_values[i], min_values[i]) + cluster_width / 2
          ax.annotate(f'{diff:.2f}%', (annotate_x, indices[i]), va='center', fontsize=14)'''
          
          
          
      ax.legend()
      custom_legend = ax.legend(loc='upper right', title="Cores",fontsize=12, title_fontsize=14, frameon=True)   
      #custom_legend = ax.legend(loc='upper right', title="3 CELL DIFFERENT UE CASES")
      ax.add_artist(custom_legend)
      plt.tight_layout()
      output_path = os.path.join(output_dir, f'{plot_num}.png')
      plt.savefig(output_path, bbox_inches='tight') 
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
      
    filter_and_plot_Name_n(11, data, args.op_dir, args.scenario, args.context, 1, ['TLB_REFILL'])
    filter_and_plot_Name_n(12, data, args.op_dir, args.scenario, args.context, 1, ['CACHE_REFILL'])
    filter_and_plot_Name_n(13, data, args.op_dir, args.scenario, args.context, 1, ['RETIRED'])
    filter_and_plot_Name_n(14, data, args.op_dir, args.scenario, args.context, 1, ['CACHE_RD', 'CACHE_WR'])
    filter_and_plot_Name_n(15, data, args.op_dir, args.scenario, args.context, 1, ['SPEC'])
    filter_and_plot_Name_n(16, data, args.op_dir, args.scenario, args.context, 1, ['CACHE_WB_VICTIM'])
    filter_and_plot_Name_n(17, data, args.op_dir, args.scenario, args.context, 1, ['WALK'])
    filter_and_plot_Name_n(18, data, args.op_dir, args.scenario, args.context, 1, ['MEM_ACCESS'])
    #filter_and_plot_Name_n(['STALL'], 19, data, args.op_dir, args.scenario, args.context)
    filter_and_plot_Name_n(19, data, args.op_dir, args.scenario, args.context, 1, ['STALL', 'END'])
    filter_and_plot_Name_n(20, data, args.op_dir, args.scenario, args.context, 1, ['EXC_TAKEN','CACHE_MISS'])
    filter_and_plot_Name_n(21, data, args.op_dir, args.scenario, args.context, 3, ['SPEC', 'PRED'])
    filter_and_plot_Name_n(22, data, args.op_dir, args.scenario, args.context, 4)
      #plot_number = plot_number + 1
    filter_and_plot('retiring', 23, data, args.op_dir, args.scenario, args.context)
    filter_and_plot('bound', 24, data, args.op_dir, args.scenario, args.context)
      #data_per_core = []'''
    output_pdf_path = os.path.join(args.op_dir, 'merged_output.pdf')
    merge_pngs_to_pdf(args.op_dir, output_pdf_path)
if __name__ == '__main__':
    main()
