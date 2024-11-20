import numpy as np

def slope_terrain(terrain, angle=-np.pi/6):
    center_x = terrain.length // 2
    slope_length = terrain.length // 4

    slope_heights = np.arange(0, slope_length, 1) * terrain.horizontal_scale * np.tan(angle)
    slope_heights = (slope_heights / terrain.vertical_scale).astype(np.int)

    terrain.height_field_raw[center_x: center_x + slope_length] += slope_heights[:, None]
    terrain.height_field_raw[center_x + slope_length:] = terrain.height_field_raw[center_x + slope_length - 1]

def drop_off_terrain(terrain, height=1.0):
    center_x = terrain.length // 2
    drop_off_length = terrain.length // 2
    
    terrain.height_field_raw[center_x: center_x + drop_off_length] -= int(height / terrain.vertical_scale)

def infant_gap_terrain(terrain, gap_size=0.5):
    center_x = terrain.length // 2
    half_gap_size = int(gap_size / terrain.horizontal_scale) // 2

    terrain.height_field_raw[center_x - half_gap_size: center_x + half_gap_size] = -400

def bridge_terrain(terrain, bridge_length=0.4, bridge_width=0.8):
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    half_bridge_length = int(bridge_length / terrain.horizontal_scale) // 2
    half_bridge_width = int(bridge_width / terrain.horizontal_scale) // 2

    terrain.height_field_raw[\
        center_x - half_bridge_length: center_x + half_bridge_length,\
        center_y + half_bridge_width:] = -400
    terrain.height_field_raw[\
        center_x - half_bridge_length: center_x + half_bridge_length,\
        :center_y - half_bridge_width] = -400