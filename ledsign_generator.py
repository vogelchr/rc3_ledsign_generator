#!/usr/bin/python
import cv2
import numpy as np
import scipy.signal
from pathlib import Path
import json

def arr(*v, **kwargs):
    """ Convenience function to create numpy arrays. """
    return np.array(v, **kwargs)

# generate the individual pixel used later on for convolution


def pixel_onoff_map(onoff_map, pixel):
    anchors = np.zeros((
        onoff_map.shape[0]*pixel.shape[0],
        onoff_map.shape[1]*pixel.shape[1]
    ), 'i')
    anchors[::pixel.shape[0], ::pixel.shape[1]] = onoff_map

    ret = np.zeros(anchors.shape + pixel.shape[2:3], dtype=pixel.dtype)
    for plane in range(pixel.shape[2]):
        ret[:, :, plane] = scipy.signal.convolve2d(anchors, pixel[:, :, plane])[
            :ret.shape[0], :ret.shape[1]]
    return ret


def gen_pixel(pitch, size, color_pix, color_bkg):
    """ Generate one pixel (later on used for convolutions) """
    ret = np.zeros((pitch, pitch, 4), np.uint8)
    ret[...] = color_bkg.reshape(1, 1, 4)
    offs = pitch-size
    ret[offs:, offs:] = color_pix.reshape(1, 1, 4)
    return ret


# colors, BGR!
col_bkg = arr(0x55, 0x55, 0x55, 0xff, dtype=np.uint8)
col_pix_on = arr(0x0f, 0x9b, 0xff, 0xff, dtype=np.uint8)
col_pix_off = arr(0x04, 0x2c, 0x48, 0xff, dtype=np.uint8)

led_rows = 7
led_cols = 28

led_arr = cv2.imread('scroll_sign_pixels.png')  # imread -> RGB
if led_arr.ndim == 3:
    led_arr = np.average(led_arr, axis=2)
led_arr = (led_arr > 127).astype('i')  # 0/1 valued array

# generate individual pixels
# ..    ## lighting up
# ..    .. grey background
# .. .. ..

pixel_size = 2
pixel_pitch = 3
pixel_on = gen_pixel(pixel_pitch, pixel_size, col_pix_on, col_bkg)
pixel_off = gen_pixel(pixel_pitch, pixel_size, col_pix_off, col_bkg)

pixels_on = pixel_onoff_map(led_arr, pixel_on)
pixels_off = pixel_onoff_map(1-led_arr, pixel_off)
pixels_on_and_off = pixels_on + pixels_off


tile_width = 96
tile_height = 32
pixel_x0 = 5
pixel_y0 = 5
pixel_xend = pixel_x0 + pixel_pitch * (led_cols+1) - pixel_size
pixel_yend = pixel_y0 + pixel_pitch * (led_rows+1) - pixel_size


##
# calculate skewed pixel for putting into the tilesets
##

side_indent_top = 4
side_indent_bottom = 8
top_row = 7
bottom_row = 28

pts1 = arr(  # X, Y
    [pixel_x0, pixel_y0],  # Upper Left
    [pixel_x0, pixel_yend-1],  # Lower Left
    [pixel_xend-1, pixel_yend-1],  # Lower Right
    [pixel_xend-1, pixel_y0],  # Upper Right
    dtype='f')

pts2 = arr(  # X,Y
    [side_indent_top, top_row],  # Upper Left
    [side_indent_bottom, bottom_row],  # Lower Left
    [tile_width-1-side_indent_bottom, bottom_row],  # Lower Right
    [tile_width-1-side_indent_top, top_row],  # Upper Right
    dtype='f')

M = cv2.getPerspectiveTransform(pts1, pts2)


##
# scroll in n_phases phases
##

n_phases = led_arr.shape[1] - led_cols

tileset_cols = 4
tileset_rows = (n_phases + tileset_cols - 1) // tileset_cols
tileset = np.zeros((tile_height * tileset_rows,
                    tile_width * tileset_cols, 4), dtype=np.uint8)

for pixel_phase in range(n_phases):
    world_tile = np.zeros((tile_height, tile_width, 4), np.uint8)
    # grey box
    world_tile[pixel_y0:pixel_yend,
               pixel_x0:pixel_xend] = col_bkg.reshape(1, 1, 4)

    world_tile[pixel_y0:pixel_y0 + led_rows * pixel_pitch,
               pixel_x0:pixel_x0 + led_cols * pixel_pitch] = \
        pixels_on_and_off[:, pixel_phase *
                          pixel_pitch: (pixel_phase + led_cols) * pixel_pitch]

    dst_x = tile_width * (pixel_phase % tileset_cols)
    dst_y = tile_height * (pixel_phase // tileset_cols)
    dst = tileset[dst_y:dst_y+tile_height, dst_x:dst_x+tile_width]

    cv2.warpPerspective(world_tile, M,
                        (tile_width, tile_height), dst=dst)

fn = Path('ledsign_animation.png')

cv2.imwrite(fn.as_posix(), tileset)

metadata = {
    'image': fn.name,
    'imageheight': tileset.shape[0],
    'imagewidth': tileset.shape[1],
    'columns': tileset.shape[1] // 32,
    'margin': 0,
    'name': fn.stem,
    'spacing': 0,
    'tilecount': n_phases * 3,
    'tiledversion': '1.7.2',
    'tileheight': 32,
    'tilewidth': 32,
    'type': 'tileset',
    'version': '1.6'
}

metadata['tiles'] = list()
anim_baseid = tileset_rows * tileset_cols
for k in range(3):
    anim_frames = list()

    for j in range(n_phases):
        anim_frames.append({
            'duration': 100,
            'tileid': 3*j+k
        })

    metadata['tiles'].append({
        'animation': anim_frames,
        'id': k
    })

json.dump(metadata, fn.with_suffix('.json').open('w'), indent=4)