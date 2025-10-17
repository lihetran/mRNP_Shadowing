'''
September 2, 2025 Liam Tran

This script will plot a bar chart using matplotlib from a txt file. I'm not going to label anything on the chart so I can add that manually in inkscape.

input: tab-separated values in a txt file
output: bar chart saved as a SVG file
'''

import sys

def parse_txt(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            print(line)
            key, value = line.strip().split()
            data[key] = float(value)
    return data

def plot_bar_chart(data, output_file):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 7))
    plt.ylim(0, 1)
    plt.bar(data.keys(), data.values(), color='grey')
    plt.tight_layout()
    plt.savefig(output_file, format='svg')
    plt.close()

def main(args):
    if len(args) != 3:
        print("Usage: python script.py <input_file.txt> <output_file.svg>")
        sys.exit(1)

    input_file = args[1]
    output_file = args[2]

    data = parse_txt(input_file)
    plot_bar_chart(data, output_file)

if __name__ == "__main__":
    main(sys.argv)
