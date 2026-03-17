import utilities
import math
import numpy as np
import chip_temp_pred_models as ctpm
import torch

# ==============================================================================
# CPU Versions (Keep for legacy support or CPU fallback)
# ==============================================================================

def get_part_temp(x, y, archor_point, chip, T0: float= 35.0):
    """
    计算芯片对某点温度的单独影响 (CPU Version)
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
    
    if in_chip_x & in_chip_y: # 如果点在芯片范围之中
        distance_to_center = utilities.euclidean_dist((x, y), (x_c + chip['len']/2, y_c + chip['wid']/2))
        result = ctpm.get_in_chip_temp(chip_len=chip['len'],
                                        chip_wid=chip['wid'],
                                        Convection_Film_Coefficient=chip['CFC'],
                                        Internal_Heat_Generation_Magnitude=chip['IHGM'],
                                        distance_to_center=distance_to_center)

        result -= T0  # get the temperature increment rather than the whole temperature 
        
    else: 
        if distance >= 20: # 距离大于20，温度增量按20计算
            result = chip['A'] * np.exp(-1 * chip['k'] * 20)
        else:
            result = chip['A'] * np.exp(-1 * chip['k'] * distance)

    return result

def get_point_temp(x, y, chiplets, layout, T0=35): 
    """
    预测某点的温度 (CPU Version)
    """
    layout_chip = layout
    point_temp = T0
    for chunk in layout_chip: 
        point_temp += get_part_temp(x, y, (chunk[0], chunk[1]), chiplets[chunk[-1]])
       
    return point_temp

def get_max_temp_and_temp_uniformity(layout: list, chiplets: dict, unit_len: int=1):
    """
    预测并计算芯片版布局上的最大温度以及温度均匀度 (CPU Version)
    """
    layout_len, layout_wid = utilities.get_chip_board_size(layout, margin_width=1.0) 
    max_temp = 0
    temps = []
    for i in range(0, int(layout_len), unit_len): 
        for j in range(0, int(layout_wid), unit_len):  
            curr_temp = get_point_temp(i, j, chiplets, layout)
            temps.append(curr_temp)
            max_temp = max(curr_temp, max_temp)

    temp_uniformity = np.std(temps)
    return max_temp, temp_uniformity


# ==============================================================================
# GPU Vectorized Versions (Optimized for Batch Processing)
# ==============================================================================

# 独立检测设备，不依赖 utilities.py 的设置
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper: Extract scaler params to GPU/CPU Tensor
def get_scaler_params_as_tensor(scaler, target_device):
    """
    提取 StandardScaler/MinMaxScaler 的参数到指定设备的 Tensor
    """
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        mean = torch.tensor(scaler.mean_, device=target_device, dtype=torch.float32)
        scale = torch.tensor(scaler.scale_, device=target_device, dtype=torch.float32)
        return mean, scale
    return None, None

# 预加载 Scaler 参数到默认设备 (作为缓存)
global_in_chip_mean, global_in_chip_scale = get_scaler_params_as_tensor(ctpm.inchip_pred_scaler_x, default_device)
global_out_chip_mean, global_out_chip_scale = get_scaler_params_as_tensor(ctpm.inchip_pred_scaler_y, default_device)

def batch_predict_in_chip_temp_increment(inputs_tensor):
    """
    GPU 批量预测函数
    自动处理 inputs_tensor 所在的设备，并强制模型移动到该设备
    """
    target_device = inputs_tensor.device
    
    # -----------------------------------------------------------
    # FIX: 检查并强制模型移动到正确的设备 (CPU/GPU)
    # -----------------------------------------------------------
    try:
        # 获取模型第一个参数的设备
        model_device = next(ctpm.inchip_pred_model.parameters()).device
    except StopIteration:
        # 如果模型没有参数(不太可能)，默认它在 cpu
        model_device = torch.device('cpu')
        
    if model_device != target_device:
        # 强制将外部模型移动到当前数据所在的设备 (通常是 cuda:0)
        ctpm.inchip_pred_model.to(target_device)
    
    # 1. 确保归一化参数在正确的设备上
    if global_in_chip_mean is not None:
        if global_in_chip_mean.device != target_device:
            mean_x = global_in_chip_mean.to(target_device)
            scale_x = global_in_chip_scale.to(target_device)
        else:
            mean_x = global_in_chip_mean
            scale_x = global_in_chip_scale
            
        inputs_scaled = (inputs_tensor - mean_x) / scale_x
    else:
        inputs_scaled = inputs_tensor 

    # 2. MLP 推理
    with torch.no_grad():
        preds_scaled = ctpm.inchip_pred_model(inputs_scaled)
    
    # 3. 反归一化
    if global_out_chip_mean is not None:
        if global_out_chip_mean.device != target_device:
            mean_y = global_out_chip_mean.to(target_device)
            scale_y = global_out_chip_scale.to(target_device)
        else:
            mean_y = global_out_chip_mean
            scale_y = global_out_chip_scale
            
        preds = preds_scaled * scale_y + mean_y
    else:
        preds = preds_scaled
        
    return preds.squeeze()

def get_max_temp_and_temp_uniformity_cuda(layout, chiplets, unit_len=1, T0=35.0):
    """
    完全向量化的温度场计算函数。
    逻辑：动态检测 layout 所在的设备，确保所有生成的张量都在同一设备上。
    """
    
    # ----------------------- 1. 准备布局数据 & 确定设备 -----------------------
    if not torch.is_tensor(layout):
        layout = torch.tensor(layout, device=default_device)
    
    # 获取 layout 实际所在的设备 (例如 cuda:0)
    current_device = layout.device
    
    # 提取芯片板尺寸
    max_x = torch.max(layout[:, 0] + layout[:, 2])
    max_y = torch.max(layout[:, 1] + layout[:, 3])
    layout_len = max_x + 1.0 # margin
    layout_wid = max_y + 1.0 # margin
    
    # 自动调整网格密度
    target_points = 250000 
    current_points = (layout_len / unit_len) * (layout_wid / unit_len)
    if current_points > target_points:
        unit_len = int(math.sqrt((layout_len * layout_wid) / target_points))
        unit_len = max(1, unit_len)

    # 生成网格时指定 device=current_device
    x = torch.arange(0, int(layout_len.item()), unit_len, device=current_device, dtype=torch.float32)
    y = torch.arange(0, int(layout_wid.item()), unit_len, device=current_device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    flat_grid_x = grid_x.flatten().unsqueeze(1) 
    flat_grid_y = grid_y.flatten().unsqueeze(1) 
    
    # ----------------------- 2. 准备芯片数据 (Tensor) -----------------------
    num_chips = layout.shape[0]
    indices = layout[:, 4].long().cpu().numpy()
    
    props_list = []
    for idx in indices:
        c = chiplets[idx]
        props_list.append([c['CFC'], c['IHGM'], c['A'], c['k']])
    
    chips_props = torch.tensor(props_list, device=current_device, dtype=torch.float32).unsqueeze(0)
    
    chip_x = layout[:, 0].unsqueeze(0)
    chip_y = layout[:, 1].unsqueeze(0)
    chip_len = layout[:, 2].unsqueeze(0)
    chip_wid = layout[:, 3].unsqueeze(0)
    
    # ----------------------- 3. 向量化计算几何距离 -----------------------
    closest_x = torch.max(chip_x, torch.min(flat_grid_x, chip_x + chip_len))
    closest_y = torch.max(chip_y, torch.min(flat_grid_y, chip_y + chip_wid))
    
    dists = torch.sqrt((flat_grid_x - closest_x)**2 + (flat_grid_y - closest_y)**2)
    is_inside = dists < 1e-4
    
    # ----------------------- 4. 计算温度场 -----------------------
    
    # === A. 外部温度 ===
    param_A = chips_props[:, :, 2]
    param_k = chips_props[:, :, 3]
    
    # [Restored Logic] 恢复截断逻辑：距离超过20的按20算
    clamped_dists = torch.clamp(dists, max=20.0)
    out_temps = param_A * torch.exp(-param_k * clamped_dists)
    
    # === B. 内部温度 (MLP) ===
    in_temps_map = torch.zeros_like(dists)
    
    if is_inside.any():
        center_x = chip_x + chip_len / 2
        center_y = chip_y + chip_wid / 2
        dist_to_center = torch.sqrt((flat_grid_x - center_x)**2 + (flat_grid_y - center_y)**2)
        
        N = flat_grid_x.shape[0]
        M = num_chips
        
        f_len = chip_len.expand(N, M).unsqueeze(-1)
        f_wid = chip_wid.expand(N, M).unsqueeze(-1)
        f_cfc = chips_props[:, :, 0].expand(N, M).unsqueeze(-1)
        f_ihgm = chips_props[:, :, 1].expand(N, M).unsqueeze(-1)
        f_dist = dist_to_center.unsqueeze(-1)
        
        mlp_inputs = torch.cat([f_len, f_wid, f_cfc, f_ihgm, f_dist], dim=-1)
        masked_inputs = mlp_inputs[is_inside]
        
        if masked_inputs.shape[0] > 0:
            # 调用批量预测，它会自动适配设备并移动模型
            pred_temps = batch_predict_in_chip_temp_increment(masked_inputs)
            pred_increments = pred_temps - T0
            in_temps_map[is_inside] = pred_increments
    
    # ----------------------- 5. 组合结果 -----------------------
    final_increments = torch.where(is_inside, in_temps_map, out_temps)
    total_increment = torch.sum(final_increments, dim=1)
    grid_temps = total_increment + T0
    
    # ----------------------- 6. 统计指标 -----------------------
    max_temp = torch.max(grid_temps)
    temp_uniformity = torch.std(grid_temps)
    
    return max_temp.item(), temp_uniformity.item()