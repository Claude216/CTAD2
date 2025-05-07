import utilities
import math
import numpy as np
import chip_temp_pred_models as ctpm
import torch

# temperature calculation related


def get_part_temp(x, y, archor_point, chip, T0: float= 35.0):

    """
        计算芯片对某点温度的单独影响/ calculate the impact of a single chiplet on the temperature at a point 
    """

    x_c, y_c = archor_point
    in_chip_x, in_chip_y = False, False
    x_closest, y_closest = 0, 0
    if x < x_c: 
        x_closest = x_c
    elif x > x_c + chip['len']: 
        x_closest = x_c + chip['len']
    else: 
        x_closest = x
        in_chip_x = True

    if y < y_c: 
        y_closest = y_c
    elif y > y_c + chip['wid']: 
        y_closest = y_c + chip['wid']
    else: 
        y_closest = y
        in_chip_y = True

    distance = math.sqrt((x - x_closest) ** 2 + (y - y_closest) ** 2)
    # result = 0
    if in_chip_x & in_chip_y: # 如果点在芯片范围之中/if the point is within the chiplet
        distance_to_center = utilities.euclidean_dist((x, y), (x_c + chip['len']/2, y_c + chip['wid']/2))
        result = ctpm.get_in_chip_temp(chip_len=chip['len'],
                                        chip_wid=chip['wid'],
                                        Convection_Film_Coefficient=chip['CFC'],
                                        Internal_Heat_Generation_Magnitude=chip['IHGM'],
                                        distance_to_center=distance_to_center)

        result -= T0  # get the temperature increment rather than the whole temperature 
        
    else: 
        if distance >= 20: # 距离大于20，温度增量按20计算/ if the distance is longer than 20mm, treat as 20mm
            result = chip['A'] * np.exp(-1 * chip['k'] * 20)
        else:
            result = chip['A'] * np.exp(-1 * chip['k'] * distance)


    return result
    



def get_point_temp(x, y, chiplets, layout, T0=35): 

    """
        预测某点的温度/ predict the temperature at a point
    """

    layout_chip = layout
    point_temp = T0
    for chunk in layout_chip: 
        point_temp += get_part_temp(x, y, (chunk[0], chunk[1]), chiplets[chunk[-1]])
       
    return point_temp

# 预测并计算芯片版布局上的最大温度以及温度均匀度/ predict and calculate the max temperature and the temperature uniformity on the chiplet board.
def get_max_temp_and_temp_uniformity(layout: list, chiplets: dict, unit_len: int=1):
    layout_len, layout_wid = utilities.get_chip_board_size(layout, margin_width=1.0) # the last two elements represent the length and width of the layout
    # pred_curves = [[chip['A'], chip['k']] for chip in chiplets.values()]
    max_temp = 0
    temps = []
    for i in range(0, int(layout_len), unit_len): 
        for j in range(0, int(layout_wid), unit_len):  
            curr_temp = get_point_temp(i, j, chiplets, layout)
            temps.append(curr_temp)
            max_temp = max(curr_temp, max_temp)
    
    # mean_temp = np.mean(temps)

    temp_uniformity = np.std(temps)
    return max_temp, temp_uniformity














device = utilities.device  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_part_temp_cuda(x, y, anchor_point, chip, T0: float = 35.0):
    # print(3)
    x_c, y_c = anchor_point
    
    # Vectorized closest point calculation

    
    
    
    in_chip_x = (x >= x_c) & (x <= x_c + chip['len'])
    in_chip_y = (y >= y_c) & (y <= y_c + chip['wid'])
    in_chip = in_chip_x & in_chip_y

        # Calculate closest x coordinate
    x_closest = torch.zeros_like(x)
    x_closest[x < x_c] = x_c
    x_closest[x > x_c + chip['len']] = x_c + chip['len']
    x_closest[(x >= x_c) & (x <= x_c + chip['len'])] = x[(x >= x_c) & (x <= x_c + chip['len'])]
    
    # Calculate closest y coordinate
    y_closest = torch.zeros_like(y)
    y_closest[y < y_c] = y_c
    y_closest[y > y_c + chip['wid']] = y_c + chip['wid']
    y_closest[(y >= y_c) & (y <= y_c + chip['wid'])] = y[(y >= y_c) & (y <= y_c + chip['wid'])]
    
    
    distance = torch.sqrt((x - x_closest)**2 + (y - y_closest)**2)
    # Vectorized temperature calculation
    chip_center_x = x_c + chip['len']/2
    chip_center_y = y_c + chip['wid']/2
    distance_to_center = torch.sqrt((x - chip_center_x)**2 + (y - chip_center_y)**2)

    if in_chip: # if in the chip area
        # chip_center_x = x_c + chip['len']/2
        # chip_center_y = y_c + chip['wid']/2
        # distance_to_center = torch.sqrt((x - chip_center_x)**2 + (y - chip_center_y)**2)
        temp_incre = ctpm.get_in_chip_temp_increment_RF_IC(
            chip_len=chip['len'],
            chip_wid=chip['wid'],
            Convection_Film_Coefficient=chip['CFC'],
            Internal_Heat_Generation_Magnitude=chip['IHGM'],
            distance_to_center=distance_to_center
        ) # only need the temperature increment for now
    
        result = torch.tensor(temp_incre[0], device=device, dtype=torch.float32)
    else: 
        distance = torch.clamp(distance, max=20, min=0)
        result = chip['A'] * torch.exp(-chip['k'] * distance)
    
    return result

def get_point_temp_cuda(x, y, chiplets, layout, T0=35):
    
    point_temp = torch.full_like(x, T0, device=device, dtype=torch.float32)
    
    # Pre-convert layout to float tensors once
    layout_chips = [(chunk[0].item(), chunk[1].item(), int(chunk[-1].item())) 
                     for chunk in layout]
    
    point_temp_effect = torch.stack([
        torch.tensor(get_part_temp_cuda(x, y, (chip[0], chip[1]), chiplets[chip[-1]]), device=device)
        for chip in layout_chips
    ])

    return (point_temp + sum(point_temp_effect)).item()

def get_max_temp_and_temp_uniformity_cuda(layout, chiplets, unit_len=1):
    
    layout_len, layout_wid = utilities.get_chip_board_size_cuda(layout, margin_width=1.0)
    
    # print(f"Board size: {layout_len} x {layout_wid}")  # Debug print
    
    # Reduce the grid size if it's too large
    while (layout_len * layout_wid) / unit_len**2 > 49: 
        unit_len += 1
    #unit_len = max(unit_len, int(min(layout_len, layout_wid) / 50))  # Adjust sampling density
    # print(f"Using unit length: {unit_len}")  # Debug print
    
    # Create grid of points
    x = torch.arange(0, int(layout_len), unit_len, device=device, dtype=torch.float32)
    y = torch.arange(0, int(layout_wid), unit_len, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # print(f"Grid size: {X.shape[0]} x {X.shape[1]}")  # Debug print
    
    # Vectorize the calculation if possible
    temps = torch.zeros_like(X, device=device, dtype=torch.float32)
    
    # Process in batches to avoid memory issues
    # batch_size = 100
    # for i in range(0, X.shape[0], batch_size):
    #     end_i = min(i + batch_size, X.shape[0])
    #     for j in range(0, X.shape[1], batch_size):
    #         end_j = min(j + batch_size, X.shape[1])
            
    #         # Process this batch
    #         for ii in range(i, end_i):
    #             for jj in range(j, end_j):
    #                 temps[ii,jj] = get_point_temp_cuda(X[ii,jj], Y[ii,jj], chiplets, layout)
        
    #     # Print progress
    #     # print(f"Processed {end_i}/{X.shape[0]} rows")
    
    # max_temp = torch.max(temps)
    # temp_uniformity = torch.std(temps.float())


    # Flatten the grid for parallel processing
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Use torch.vmap for parallel processing (requires PyTorch 2.0+)
    
    # Fallback to simple parallel processing
    # print(len(X_flat))
    temps = torch.stack([
        torch.tensor(get_point_temp_cuda(x, y, chiplets, layout), device=device)
        for x, y in zip(X_flat, Y_flat)
    ])
    
    # Reshape back to grid
    temps = temps.reshape(X.shape)
    
    max_temp = torch.max(temps)
    temp_uniformity = torch.std(temps.float())
    
    return max_temp.item(), temp_uniformity.item()


