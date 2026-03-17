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

def middle_of_edge_cuda(x_min, x_max, y_min, y_max, is_vertical):
    """
    计算边中点（输入是标量张量，输出保持在同一设备上）。
    """
    if is_vertical:
        return torch.stack((x_min, (y_min + y_max) * 0.5))
    return torch.stack(((x_min + x_max) * 0.5, y_min))


def _extract_chip_geometry_cuda(chip):
    """
    从芯片张量中提取几何信息。
    芯片格式: [x, y, len, wid, index]，这里只用前4个值。
    """
    return chip[0], chip[1], chip[2], chip[3]


def closest_edges_cuda(anchor1, anchor2=None, len_wid1=None, len_wid2=None):
    """
    CUDA 版本最近边距离，支持两种调用方式：
    1) closest_edges_cuda(chip1, chip2) ，chip=[x, y, len, wid, ...]
    2) closest_edges_cuda(anchor1, anchor2, len_wid1, len_wid2)
    """
    # 新接口：直接传两个芯片张量，避免 .item() 导致 GPU 同步
    if anchor2 is not None and len_wid1 is None and len_wid2 is None:
        x1, y1, len1, wid1 = _extract_chip_geometry_cuda(anchor1)
        x2, y2, len2, wid2 = _extract_chip_geometry_cuda(anchor2)
    else:
        # 兼容旧接口：把输入转到同设备张量
        dev = anchor1.device if torch.is_tensor(anchor1) else "cpu"
        x1, y1 = torch.as_tensor(anchor1, device=dev, dtype=torch.float32)
        x2, y2 = torch.as_tensor(anchor2, device=dev, dtype=torch.float32)
        len1, wid1 = torch.as_tensor(len_wid1, device=dev, dtype=torch.float32)
        len2, wid2 = torch.as_tensor(len_wid2, device=dev, dtype=torch.float32)

    # 计算边界框
    x1_min, x1_max = x1, x1 + len1
    y1_min, y1_max = y1, y1 + wid1
    x2_min, x2_max = x2, x2 + len2
    y2_min, y2_max = y2, y2 + wid2

    # 判断相对位置并选择最近的一组边中点
    if x1_max <= x2_min:  # 1 在 2 左边
        middle1 = middle_of_edge_cuda(x1_max, x1_max, y1_min, y1_max, is_vertical=True)
        middle2 = middle_of_edge_cuda(x2_min, x2_min, y2_min, y2_max, is_vertical=True)
    elif x2_max <= x1_min:  # 2 在 1 左边
        middle1 = middle_of_edge_cuda(x1_min, x1_min, y1_min, y1_max, is_vertical=True)
        middle2 = middle_of_edge_cuda(x2_max, x2_max, y2_min, y2_max, is_vertical=True)
    elif y1_max <= y2_min:  # 1 在 2 下方
        middle1 = middle_of_edge_cuda(x1_min, x1_max, y1_max, y1_max, is_vertical=False)
        middle2 = middle_of_edge_cuda(x2_min, x2_max, y2_min, y2_min, is_vertical=False)
    elif y2_max <= y1_min:  # 2 在 1 下方
        middle1 = middle_of_edge_cuda(x1_min, x1_max, y1_min, y1_min, is_vertical=False)
        middle2 = middle_of_edge_cuda(x2_min, x2_max, y2_max, y2_max, is_vertical=False)
    else:
        # 有重叠或接触，边距记为 0
        return torch.zeros((), device=x1.device, dtype=x1.dtype)

    return utilities.euclidean_dist_cuda(middle1, middle2)

def get_total_interconnect_length_cuda(layout, connectivity_pairs):
    """
    计算总互连长度（CUDA）。
    关键点：全程使用 layout 所在设备，减少 CPU/GPU 同步开销。
    """
    # 如果输入不是张量，默认转成 float32
    if not torch.is_tensor(layout):
        layout = torch.tensor(layout, dtype=torch.float32)
    else:
        layout = layout.float()

    target_device = layout.device

    # 预处理连线对：去掉自连接，并规范成 (小索引, 大索引)
    connectivity_pairs = [[(min(i, j), max(i, j)), num_wires] for (i, j), num_wires in connectivity_pairs if i != j]

    # 用标量张量累计长度，保持在同一设备上
    total_length = torch.zeros((), device=target_device, dtype=layout.dtype)

    for (i, j), num_wires in connectivity_pairs:
        # 直接在设备侧定位索引，避免 .item() 触发同步
        index1 = torch.where(layout[:, -1].long() == i)[0][0]
        index2 = torch.where(layout[:, -1].long() == j)[0][0]

        length = closest_edges_cuda(layout[index1], layout[index2])
        total_length = total_length + length * num_wires

    return total_length

