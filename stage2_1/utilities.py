import random
import numpy as np
import math
import torch


# global variables
device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

# helper functions

# 计算欧拉距离
def euclidean_dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def custom_split(input_list, chiplet_amount, chunk_size=6):
    """
        Split the list into the specified number of chunks with the given chunk size
    """ 
    chunks = [input_list[i:i + chunk_size] for i in range(0, chunk_size * chiplet_amount, chunk_size)]
    
    # Add the remaining elements as the last chunk
    remaining_start = chunk_size * chiplet_amount
    if remaining_start < len(input_list):
        chunks.append(input_list[remaining_start:])
    
    return chunks

def avg(l: list): 
    return sum(l)//len(l)

def edge_distance(anch1, anch2, len_wid1, len_wid2):
    """
    Calculate the distance between the closest edges of two chiplets.
    """
    x1, y1 = anch1 
    w1, h1 = len_wid1
    x2, y2 = anch2
    w2, h2 = len_wid2

    # Horizontal and vertical distances between chiplets
    dx = max(0, max(x2 - (x1 + w1), x1 - (x2 + w2)))
    dy = max(0, max(y2 - (y1 + h1), y1 - (y2 + h2)))

    # If dx or dy is zero, it means they are adjacent or overlapping along that axis
    return max(dx, dy)

def get_chip_board_size(layout, margin_width=1.0): 

    """
        return the length and width of the whole input chip board
    """

    min_x = min(chiplet[0] for chiplet in layout)
    min_y = min(chiplet[1] for chiplet in layout)
    max_x = max(chiplet[0] + chiplet[2] for chiplet in layout)
    max_y = max(chiplet[1] + chiplet[3] for chiplet in layout)
    return max_x - min_x + 2 * margin_width, max_y - min_y + 2 * margin_width

# visualization

import matplotlib.pyplot as plt
def show_chip_design(layout): 
# Updated layout data
    # Extract chiplet information (ignore the last two values)
    chiplets = layout
    size = get_chip_board_size(layout, margin_width=1.0)
    # Visualization ignoring board dimensions
    fig, ax = plt.subplots(figsize=size)

    # Calculate dynamic bounds based on chiplet positions and sizes
    min_x = min(chiplet[0] for chiplet in chiplets)
    min_y = min(chiplet[1] for chiplet in chiplets)
    max_x = max(chiplet[0] + chiplet[2] for chiplet in chiplets)
    max_y = max(chiplet[1] + chiplet[3] for chiplet in chiplets)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_aspect('equal', adjustable='box')

    # Plot chiplets
    for chiplet in chiplets:
        x, y, length, width, flag, index = chiplet
        rect = plt.Rectangle((x, y), length, width, edgecolor='blue', facecolor='lightblue', alpha=0.6, label=f"Chiplet {index}")
        ax.add_patch(rect)
        ax.text(x + length / 2, y + width / 2, str(index), ha='center', va='center', fontsize=10, color='black')

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Chiplet Layout Visualization (Dynamic Bounds)")
    plt.grid(True)
    plt.show()


def euclidean_dist_cuda(p1, p2):
    return torch.sqrt(torch.sum((p1 - p2) ** 2))

def get_chip_board_size_cuda(layout, margin_width=1.0):
    if not torch.is_tensor(layout):
        layout = torch.tensor(layout, device=device)
    
    max_x = torch.max(layout[:, 0] + layout[:, 2])
    max_y = torch.max(layout[:, 1] + layout[:, 3])
    
    return max_x + margin_width, max_y + margin_width


def show_chip_design_cuda(layout): 
# Updated layout data
    # Extract chiplet information (ignore the last two values)
    chiplets = layout
    size = get_chip_board_size(layout, margin_width=1.0)
    # Visualization ignoring board dimensions
    fig, ax = plt.subplots(figsize=size)

    # Calculate dynamic bounds based on chiplet positions and sizes
    min_x = min(chiplet[0] for chiplet in chiplets)
    min_y = min(chiplet[1] for chiplet in chiplets)
    max_x = max(chiplet[0] + chiplet[2] for chiplet in chiplets)
    max_y = max(chiplet[1] + chiplet[3] for chiplet in chiplets)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_aspect('equal', adjustable='box')

    # Plot chiplets
    for chiplet in chiplets:
        x, y, length, width, index = chiplet
        rect = plt.Rectangle((x, y), length, width, edgecolor='blue', facecolor='lightblue', alpha=0.6, label=f"Chiplet {index}")
        ax.add_patch(rect)
        ax.text(x + length / 2, y + width / 2, str(index), ha='center', va='center', fontsize=10, color='black')

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Chiplet Layout Visualization (Dynamic Bounds)")
    plt.grid(True)
    plt.show()

