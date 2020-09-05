#!/usr/bin/env python

from os.path import join as p_join, dirname, basename, splitext
from sys import path as s_path, argv
from pathlib import Path, PurePath
PROJDIR = str(Path(__file__).resolve().parents[1])
SRCDIR = p_join(PROJDIR, 'src')
s_path.append(SRCDIR)
GENDIR = p_join(PROJDIR, 'generatedData')
DATASETDIR = p_join(GENDIR, 'datasets')
DS_SUB_DIR = 'data'
EVAL_SUB_DIR = 'eval'
IMG_SUB_DIR = 'img'
PDF_FIG_STEM='figure'
AE2D_EMB_PICKLE_STEM = 'ae2d_plot_data'
POT2D_EMB_PICKLE_STEM = 'pot2d_plot_data'
LOGDIR = p_join(PROJDIR, 'logs')
CKPDIR = p_join(PROJDIR, 'checkpoints')

DATE_TIME_FORMAT = "%Y-%m-%d_%H_%M_%S"

KNND_COL = 'knnd'
TYPE_COL = 'type'
NAME_COL = 'name'
K_COL = 'k'
TRIAL_COL = 'trial'
BGN = 'benign'
MGN = 'malign'

silent = False

###############################################################################
if __name__ == '__main__':
    pass
