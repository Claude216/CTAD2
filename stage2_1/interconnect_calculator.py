import utilities
import torch



# interconnect calculation related

def middle_of_edge(x_min, x_max, y_min, y_max, is_vertical):
    """Calculate the middle point of an edge."""
    if is_vertical:
        return (x_min, (y_min + y_max) / 2)
    else:
        return ((x_min + x_max) / 2, y_min)

def closest_edges(anchor1, anchor2, len_wid1, len_wid2): 
    x1, y1 = anchor1
    x2, y2 = anchor2
    len1, wid1 = len_wid1
    len2, wid2 = len_wid2

    x1_min, x1_max = x1, x1 + len1
    y1_min, y1_max = y1, y1 + wid1
    x2_min, x2_max = x2, x2 + len2
    y2_min, y2_max = y2, y2 + wid2
    
    # Determine relative positions and select closest edges
    if x1_max <= x2_min:  # chiplet1 is to the left of chiplet2
        # Closest edges: right edge of chiplet1, left edge of chiplet2
        middle1 = (x1_max, (y1_min + y1_max) / 2)
        middle2 = (x2_min, (y2_min + y2_max) / 2)
    elif x2_max <= x1_min:  # chiplet2 is to the left of chiplet1
        # Closest edges: left edge of chiplet1, right edge of chiplet2
        middle1 = (x1_min, (y1_min + y1_max) / 2)
        middle2 = (x2_max, (y2_min + y2_max) / 2)
    elif y1_max <= y2_min:  # chiplet1 is below chiplet2
        # Closest edges: top edge of chiplet1, bottom edge of chiplet2
        middle1 = (((x1_min + x1_max) / 2), y1_max)
        middle2 = (((x2_min + x2_max) / 2), y2_min)
    elif y2_max <= y1_min:  # chiplet2 is below chiplet1
        # Closest edges: bottom edge of chiplet1, top edge of chiplet2
        middle1 = (((x1_min + x1_max) / 2), y1_min)
        middle2 = (((x2_min + x2_max) / 2), y2_max)
    else:
        # Overlap or adjacency detected; distance is zero
        return 0
    
    return utilities.euclidean_dist(middle1, middle2)

def get_total_interconnect_length(layout, connectivity_pairs): 

    # process the connectivity pairs    
    connectivity_pairs = [[(i, j), num_wires] for (i, j), num_wires in connectivity_pairs if i != j]
    connectivity_pairs = [[(min(i, j), max(i, j)), num_wires] for (i, j), num_wires in connectivity_pairs]

    
    layout_chips = layout# [:-2]
    # layout_chunks = utilities.custom_split(input_list=layout, chiplet_amount=len(chiplets.keys()))

    total_length = 0

    for (i, j), numer_of_wires in connectivity_pairs: 
        total_length += closest_edges(
            anchor1=(layout_chips[i][0], layout_chips[i][1]),
            anchor2=(layout_chips[j][0], layout_chips[j][1]),
            len_wid1=(layout_chips[i][2], layout_chips[i][3]), 
            len_wid2=(layout_chips[j][2], layout_chips[j][3])
        ) * numer_of_wires

    return total_length


#=============================================== cuda ver below ===========================================

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def middle_of_edge_cuda(x_min, x_max, y_min, y_max, is_vertical):
    if is_vertical:
        return torch.tensor([x_min, (y_min + y_max) / 2], device=device)
    else:
        return torch.tensor([(x_min + x_max) / 2, y_min], device=device)

def closest_edges_cuda(anchor1, anchor2, len_wid1, len_wid2):
    # Convert inputs to tensors
    x1, y1 = torch.tensor(anchor1, device=device)
    x2, y2 = torch.tensor(anchor2, device=device)
    len1, wid1 = torch.tensor(len_wid1, device=device)
    len2, wid2 = torch.tensor(len_wid2, device=device)
    
    # Calculate boundaries
    x1_min, x1_max = x1, x1 + len1
    y1_min, y1_max = y1, y1 + wid1
    x2_min, x2_max = x2, x2 + len2
    y2_min, y2_max = y2, y2 + wid2
    
    # Determine relative positions using tensor operations
    is_left = x1_max <= x2_min
    is_right = x2_max <= x1_min
    is_below = y1_max <= y2_min
    is_above = y2_max <= y1_min
    
    # Calculate middle points for each case
    middle1 = torch.zeros(2, device=device)
    middle2 = torch.zeros(2, device=device)
    
    # Use torch.where for conditional assignments
    if is_left:
        middle1 = torch.tensor([x1_max, (y1_min + y1_max) / 2], device=device)
        middle2 = torch.tensor([x2_min, (y2_min + y2_max) / 2], device=device)
    elif is_right:
        middle1 = torch.tensor([x1_min, (y1_min + y1_max) / 2], device=device)
        middle2 = torch.tensor([x2_max, (y2_min + y2_max) / 2], device=device)
    elif is_below:
        middle1 = torch.tensor([(x1_min + x1_max) / 2, y1_max], device=device)
        middle2 = torch.tensor([(x2_min + x2_max) / 2, y2_min], device=device)
    elif is_above:
        middle1 = torch.tensor([(x1_min + x1_max) / 2, y1_min], device=device)
        middle2 = torch.tensor([(x2_min + x2_max) / 2, y2_max], device=device)
    else:
        return torch.tensor(0.0, device=device)
    
    return utilities.euclidean_dist_cuda(middle1, middle2)

def get_total_interconnect_length_cuda(layout, connectivity_pairs):
    # Convert layout to tensor if not already
    # print(layout)
    if not torch.is_tensor(layout):
        layout = torch.tensor(layout, device=device)
    
    # Process connectivity pairs
    connectivity_pairs = [[(min(i, j), max(i, j)), num_wires] for (i, j), num_wires in connectivity_pairs if i != j]
    
    # Calculate total length
    total_length = torch.tensor(0.0, device=device)
    
    for (i, j), num_wires in connectivity_pairs:
        index1 = torch.where(layout[:, -1] == i)[0].item()
        index2 = torch.where(layout[:, -1] == j)[0].item()
        # print(index1, index2)
        length = closest_edges_cuda(
            anchor1=(layout[index1][0].item(), layout[index1][1].item()),
            anchor2=(layout[index2 ][0].item(), layout[index2 ][1].item()),
            len_wid1=(layout[index1][2].item(), layout[index1][3].item()),
            len_wid2=(layout[index2 ][2].item(), layout[index2 ][3].item())
        )
        total_length += length*num_wires
    
    return total_length

