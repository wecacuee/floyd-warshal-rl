from __future__ import print_function
import subprocess
import os
import sys
curr_dir = os.path.dirname(__file__ or '.')
sys.path.insert(0, os.path.dirname(curr_dir))
import conf
import glob
import numpy as np
import cv2
import shutil

def get_start_number(patt):
    print("Looking for {}".format(patt))
    for i in xrange(100000):
        if os.path.exists(patt % i):
            return i

def get_end_number(patt):
    print("Looking for {}".format(patt))
    for i in reversed(xrange(100000)):
        if os.path.exists(patt % i):
            return i

def imgs_patt_to_video_cmd(left_patt, right_patt, output_file):
    cmd = """ffmpeg 
    -start_number {left_start_number}
    -r 30
    -i {left_patt}
    -start_number {right_start_number}
    -r 30
    -i {right_patt}
    -filter_complex [0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]
    -map [vid]
    -pix_fmt yuv420p
    -framerate 30
    {output_file}""".format(
        left_start_number = get_start_number(left_patt),
        right_start_number = get_start_number(right_patt),
        left_patt = left_patt,
        right_patt = right_patt,
        output_file = output_file,
    ).split()
    return cmd


def ffmpeg_cmd(**kwargs):
    cmd = """ffmpeg 
    -start_number {start_number}
    -r 30
    -i {front_view_patt}
    -pix_fmt yuv420p
    -framerate 30
    {output_file}""".format(**kwargs).split()
    return cmd


def ffmpeg_cmd_gen_small_video(**kwargs):
    cmd = """ffmpeg 
    -i {output_file}
    -r 2
    -filter:v setpts=0.1*PTS
    -pix_fmt yuv420p
    {small_output_file}""".format(**kwargs).split()
    return cmd


def verbose_rename(x, y):
    print("Backing up {} to {}".format(x, y))
    os.rename(x, y)


def rotate_output_file(output_file):
    fileroot, ext = os.path.splitext(output_file)
    backup_ext = ""
        
    for i in range(100):
        new_backup_ext = ".%d" % i
        if os.path.exists(fileroot + backup_ext + ext):
            verbose_rename(fileroot + backup_ext + ext,
                           fileroot + new_backup_ext + ext)


def combine_imgs_to_video(action_value_img_fmt,
                          agent_vis_img_fmt,
                          out_vid_file):
    cmd = imgs_patt_to_video_cmd(
        left_patt = agent_vis_img_fmt,
        right_patt = action_value_img_fmt,
        out_vid_file = out_vid_file)
    print("running: '{}'".format(" ".join(cmd)))
    subprocess.call(cmd)
                                 
    
if __name__ == '__main__':
    try:
        action_value_img_fmt = sys.argv[1]
        agent_vis_img_fmt = sys.argv[2]
        out_vid_file = sys.argv[3]
    except KeyError:
        raise ValueError("Need conf_name as first argument")
    combine_imgs_to_video(action_value_img_fmt, agent_vis_img_fmt, out_vid_file)
