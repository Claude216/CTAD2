# chiplets format
"""
{
    index1: chip{},
    index2: chip{},
    ...
    indexn: chip{}, 
}
"""

# chip format: 
"""
{
    'len': ...,
    'wid': ...,
    'CFC': ...,
    'IHGM': ...,
    'A': ...,
    'k': ...
}
"""

# layout format: 
"""
[
    [x1, y1, len1, wid2, index1], 
    [x2, y2, len2, wid2, index2], 
    ...
    [xn, yn, lenn, widn, indexn],
]

"""

# layouts (gene) format: 
"""
[
    layout1,
    layout2,
    layout3,
    ...,
    layoutn
]

"""

# connectivity format: 
"""
[
    [(chip_index_1, chip_index2), number_of_wires], 
    ...

]

"""













