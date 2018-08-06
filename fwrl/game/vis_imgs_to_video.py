import subprocess
import os
import sys
from pathlib import Path

from umcog.confutils import (extended_kwprop, KWProp as prop, xargspartial)
from .play import (NoOPObserver)

def get_start_number(patt):
    print("Looking for {}".format(patt))
    for i in range(100000):
        if os.path.exists(patt % i):
            return i

def get_end_number(patt):
    print("Looking for {}".format(patt))
    for i in reversed(range(100000)):
        if os.path.exists(patt % i):
            return i


def repl_ext(f, ext):
    return os.path.splitext(f)[0] + '.' + ext
    

def img_patt_to_video_cmd(left_patt, out_vid_file):
    cmd = """ffmpeg 
    -y
    -start_number {left_start_number}
    -r 30
    -i {left_patt}
    -framerate 30
    {out_vid_file}""".format(
        left_start_number = get_start_number(left_patt),
        left_patt = left_patt,
        out_vid_file = out_vid_file,
    ).split()
    return cmd


def run_cmd(cmd):
    print("running: '{}'".format(" ".join(cmd)))
    return subprocess.call(cmd)
    

def combine_left_right_vid(left_inp, right_inp, out_vid_file):
    cmd = """ffmpeg 
    -y
    -i {left_inp}
    -i {right_inp}
    -filter_complex [0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]
    -map [vid]
    -framerate 30
    {out_vid_file}""".format(
        left_inp = left_inp,
        right_inp = right_inp,
        out_vid_file = out_vid_file,
    ).split()
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

def rm_images(patt, pattern_iter):
    for t in pattern_iter:
        f = patt % t
        if os.path.exists(f):
            os.remove(f)


@extended_kwprop
def combine_imgs_to_video(patt,
                          vidfile = prop(lambda s: repl_ext(s.patt % 0, 'webm'))):
    run_cmd(img_patt_to_video_cmd(patt, vidfile))
    rm_images(patt, range(1000))
    return vidfile


def combine_imgs_to_videos(action_value_img_fmt,
                           agent_vis_img_fmt,
                           out_vid_file,
                           nepisodes):
    for e in range(nepisodes):
        acv, agv, ovf = (action_value_img_fmt % e,
                         agent_vis_img_fmt % e,
                         out_vid_file % e)
        right_vid = combine_imgs_to_video(agv)
        run_cmd(combine_left_right_vid(
            left_inp = combine_imgs_to_video(acv),
            right_inp = combine_imgs_to_video(agv),
            out_vid_file = ovf))


class ImgsToVideoObs(NoOPObserver):
    @extended_kwprop
    def __init__(self,
                 post_process = xargspartial(
                     combine_imgs_to_videos,
                     ["action_value_img_fmt", "agent_vis_img_fmt",
                      "out_vid_file", "nepisodes"]),
                 agent_vis_img_fmt = prop(lambda s : str(Path(s.log_file_dir)
                                                         / "agent_grid_world_%d_%%d.png")),
                 action_value_img_fmt = prop(lambda s : str(Path(s.log_file_dir)
                                                            / "action_value_%d_%%d.png")),

                 out_vid_file = prop(lambda s : str(Path(s.log_file_dir) / "out_%d.webm")),
                 # Needs: log_file_dir, nepisodes
                 ):
        self.post_process = post_process

    def on_play_end(self):
        self.post_process()

    
                                 
    
if __name__ == '__main__':
    try:
        action_value_img_fmt = sys.argv[1]
        agent_vis_img_fmt = sys.argv[2]
        out_vid_file = sys.argv[3]
    except KeyError:
        raise ValueError("Need conf_name as first argument")
    combine_imgs_to_video(action_value_img_fmt, agent_vis_img_fmt, out_vid_file)
