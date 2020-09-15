#!/usr/bin/env python

# best vim colorscheme: koehler, next best: ron

from os.path import join as p_join, dirname, basename, splitext
from sys import path as s_path, maxsize, argv, exit
from pathlib import Path
s_path.append(p_join(str(Path(__file__).resolve().parents[0]), 'src'))

import os
import re

from od_conf import GENDIR, DATASETDIR, DS_SUB_DIR, EVAL_SUB_DIR, DATE_TIME_FORMAT

import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2, l1_l2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from datetime import datetime

from od_models import  ConvAe2, Aee, RocAuc, OF1, Ode, KPr, Ap, Potatoes, PotEns, IfModel, OcsvmModel, EvalConf
from od_tools import Mnist, FMnist, Cifar10, Sine, Sine50, SineRnd, BinData, CircleRnd
from tools import ewedge

def conf_cmp_potatoes_f():
    #test = True
    test = False

    ###############################################################################
    # configuring the datasets
    if test:
        ds_col = "small_ova_mnist_bc0_rm0.01_s2_0"
    else:
        ds_col = "ova_mnist_bc0_rm0.005_s10_0"
        #ds_col = "ova_mnist_bc0_rm0.005_s10_1"
        #ds_col = "ova_mnist_bc0_rm0.005_s10_2"
        #ds_col = "ova_mnist_bc0_rm0.005_s10_3"
        #ds_col = "ova_mnist_bc0_rm0.005_s10_4"

        #ds_col = "ova_mnist_bc1_rm0.005_s10_0"
        #ds_col = "ova_mnist_bc1_rm0.005_s10_1"
        #ds_col = "ova_mnist_bc1_rm0.005_s10_2"
        #ds_col = "ova_mnist_bc1_rm0.005_s10_3"
        #ds_col = "ova_mnist_bc1_rm0.005_s10_4"

        #ds_col = "ova_fmnist_bc0_rm0.005_s10_0"
        #ds_col = "ova_fmnist_bc0_rm0.005_s10_1"
        #ds_col = "ova_fmnist_bc0_rm0.005_s10_2"
        #ds_col = "ova_fmnist_bc0_rm0.005_s10_3"
        #ds_col = "ova_fmnist_bc0_rm0.005_s10_4"

        #ds_col = "ova_fmnist_bc1_rm0.005_s10_0"
        #ds_col = "ova_fmnist_bc1_rm0.005_s10_1"
        #ds_col = "ova_fmnist_bc1_rm0.005_s10_2"
        #ds_col = "ova_fmnist_bc1_rm0.005_s10_3"
        #ds_col = "ova_fmnist_bc1_rm0.005_s10_4"

    dn = p_join(DATASETDIR, ds_col, DS_SUB_DIR)
    bds = [BinData(ol_lab = 1, fn = p_join(dn, fn),
                   name = ds_col + "_" + re.split("\s", splitext(fn)[0])[-1])
            for fn in sorted(os.listdir(dn))]
    mo = re.match(r'^.*ova_([^_]+)_bc(\d+)', ds_col)
    bdgids = [f"{mo.group(1)}_{mo.group(2)}"]*len(bds)
    
    print(f"\nused dataset collection: {ds_col}\n\n")
    print("\n---- len(bds) ----")
    print(len(bds))
    print("\n---- [bd.X.shape for bd in bds] ----")
    print([bd.X.shape for bd in bds])
    bm_strs = [f"b[{i}] {bd.get_b_count()} m[{i}] {bd.get_m_count()}"
                   for i,bd in enumerate(bds)]
    print(*bm_strs, sep = "\n")

    ###############################################################################
    # configuring the models
    vb = 0
    dot_iv = 1000
    block_iv = 10*dot_iv

    def emax(ens, l_ols): return np.amax(l_ols, axis = 0)
    def emin(ens, l_ols): return np.amin(l_ols, axis = 0)
    def emed(ens, l_ols): return np.median(l_ols, axis = 0)

    ############# reg ConvAe2 ##################
    ae_reg_c = ConvAe2
    ae_reg_kregf = 1.e-4
    ae_reg_kreg = l2(ae_reg_kregf)
    ae_reg_breg = None
    ae_reg_areg = None
    ae_reg_ld = 32
    ae_reg_dot_iv = 50
    ae_reg_block_iv = 10*ae_reg_dot_iv
    if test:
        ae_reg_epochs = 3
        ae_reg_loss_th = 1
        ae_reg_ep_th = {1: ae_reg_loss_th}
    else:
        ae_reg_epochs = 750
        ae_reg_loss_th = 0.015
        ae_reg_ep_th = {10: .1, 50: 0.08}

    ############# of ConvAe2 ##################
    ae_of_c = ConvAe2
    ae_of_ld = 32
    ae_of_dot_iv = 100
    ae_of_block_iv = 10*ae_of_dot_iv
    if test:
        ae_of_epochs = 1
        ae_of_loss_th = 1
        ae_of_ep_th = {}
    else:
        ae_of_epochs = 3000
        ae_of_loss_th = 0.005
        ae_of_ep_th = {50: 0.01, 100: 0.008}

    ############# aee ConvAe2 ##################
    aee_aec = ConvAe2
    aee_kregf = 1.e-4
    aee_kreg = l2(aee_kregf)
    aee_breg = None
    aee_areg = None
    aee_ld = 32
    aee_k = 5
    aee_efun = lambda l_ols: np.median(l_ols, axis = 0)
    aee_efun.__name__ = "median"
    aee_dot_iv = 100
    aee_block_iv = 500
    if test:
        aee_epochs = 3
        aee_loss_th = 1
        aee_ep_th = {1: aee_loss_th}
        aee_mr = 2
    else:
        aee_epochs = 750
        aee_loss_th = 0.015
        aee_ep_th = {10: .1, 50: 0.08}
        aee_mr = 10

    aee_ae_kwargs = {'ld': aee_ld, 'epochs': aee_epochs, 'kreg': aee_kreg, 'loss_th': aee_loss_th, 'vb': vb, 'ep_th': aee_ep_th, 'dot_iv': aee_dot_iv, 'block_iv': aee_block_iv}

    ############# aee_of ConvAe2 ##################
    aee_of_aec = ConvAe2
    aee_of_kreg = None
    aee_of_breg = None
    aee_of_areg = None
    aee_of_ld = 32
    aee_of_k = 5
    aee_of_efun = lambda l_ols: np.amax(l_ols, axis = 0)
    aee_of_efun.__name__ = "amax"
    aee_of_dot_iv = 100
    aee_of_block_iv = 100
    if test:
        aee_of_epochs = 3
        aee_of_loss_th = 1
        aee_of_ep_th = {1: aee_of_loss_th}
        aee_of_mr = 2
    else:
        aee_of_epochs = 2000
        aee_of_loss_th = 0.015
        aee_of_ep_th = {10: .1, 50: 0.08}
        aee_of_mr = 10

    aee_of_ae_kwargs = {'ld': aee_of_ld, 'epochs': aee_of_epochs, 'kreg': aee_of_kreg, 'loss_th': aee_of_loss_th, 'vb': vb, 'ep_th': aee_of_ep_th, 'dot_iv': aee_of_dot_iv, 'block_iv': aee_of_block_iv}

    ############# pot ConvAe2 ##################
    pot_aec = ConvAe2
    pot_ld = 32
    pot_k = 5
    pot_dot_iv = dot_iv
    pot_block_iv = 100
    if test:
        pot_epochs = 3
        pot_loss_th = 1
        pot_ep_th = {1: pot_loss_th}
        pot_mr = 2
        pot_rp = 2
    else:
        pot_epochs = 3000
        pot_loss_th = 0.005
        pot_ep_th = {50: 0.015, 100: 0.008, 740: 0.008, 1490: 0.008}
        pot_mr = 10
        pot_rp = 3

    ae_kwargs = {'ld': pot_ld, 'epochs': pot_epochs, 'loss_th': pot_loss_th, 'vb': vb, 'ep_th': pot_ep_th, 'dot_iv': pot_dot_iv, 'block_iv': pot_block_iv}

    ############# potatoes ensemble ##################
    pens_aec = ConvAe2
    pens_ld = 32
    pens_efun = emed
    pens_mr = 1
    pens_dot_iv = dot_iv
    pens_block_iv = block_iv
    if test:
        pens_k = 2
        pens_s = 2
        pens_epochs = 2
        pens_loss_th = 1
        pens_ep_th = {}
        pens_pmr = 2
        pens_prp = 2
    else:
        pens_k = 5
        pens_s = 5
        pens_epochs = 3000
        pens_loss_th = 0.005
        pens_ep_th = {50: 0.015, 100: 0.008}
        pens_pmr = 10
        pens_prp = 3

    pens_kwargs = {'ld': pens_ld, 'epochs': pens_epochs, 'loss_th': pens_loss_th, 'vb': vb, 'ep_th': pens_ep_th, 'dot_iv': pens_dot_iv, 'block_iv': pens_block_iv}

    ############# if ##################
    if_ne = 500
    if_ms = 5000

    ############# ocsvm ###############
    oc_g = 'scale'
    oc_nu = .1

    ###########################################################################
    # apply above configuration parameters
    ae_reg = ae_reg_c(ae_reg_ld, ae_reg_epochs, None, ae_reg_kreg, ae_reg_breg, ae_reg_areg,
            loss_th = ae_reg_loss_th, vb = vb, ep_th = ae_reg_ep_th,
            dot_iv = ae_reg_dot_iv, block_iv = ae_reg_block_iv)
    ae_of = ae_of_c(ae_of_ld, ae_of_epochs, None, loss_th = ae_of_loss_th,
            vb=vb, ep_th=ae_of_ep_th, dot_iv = ae_of_dot_iv, block_iv = ae_of_block_iv)
    pot = Potatoes(pot_aec, ae_kwargs, pot_k, mr = pot_mr, rp = pot_rp)

    aee = Aee(aee_aec, aee_ae_kwargs, aee_k, mr = aee_mr, efun = aee_efun)
    aee_of = Aee(aee_of_aec, aee_of_ae_kwargs, aee_of_k, mr = aee_of_mr, efun = aee_of_efun)
    pens = PotEns(pens_aec, pens_kwargs, pens_k, pens_s, pens_efun, None, pens_pmr, pens_prp, pens_mr)


    ifor = IfModel(n_estimators = if_ne, max_samples = if_ms)
    ocsvm = OcsvmModel(gamma = oc_g, nu = oc_nu)

    odms = [ifor, ocsvm, ae_reg, pot]
    #odms = [aee_of, aee, pens]
    odmss = "_".join([str(m) for m in odms])

    ###############################################################################
    # configuring the metrics
    mets = [Ap(), RocAuc(), KPr(k=20), KPr(k=40), OF1()]
    metss = "_".join([str(m) for m in mets])

    ###############################################################################
    # collecting all the configuration into EvalConf instances
    rf = lambda bd, odm, met: 1 if odm is pot else 10
    ecs = EvalConf.cross_comb(bds, odms, mets, bdgids, rf)
    
    dt = datetime.now().strftime(DATE_TIME_FORMAT)
    csv_fn = f"{ds_col}_{odmss}_{metss}_{dt}.csv"
    nl =  len(csv_fn)
    if nl > 256:
        raise ValueError(f"error: file name too long ({nl}):\n'{csv_fn}'")

    return ecs, ds_col, csv_fn


def cmp_save_potatoes_f():
    start = datetime.now()
    ecs, ds_col, csv_fn = conf_cmp_potatoes_f()

    dn = p_join(DATASETDIR, ds_col, EVAL_SUB_DIR)
    os.makedirs(dn, exist_ok = True)
    fn_fdf = p_join(dn, csv_fn)

    # test whether filesystem-wise everything is OK
    fn_fdf_p = Path(fn_fdf)
    fn_fdf_p.touch()
    fn_fdf_p.unlink()

    print(f"\nevaluations will be written to:\n{fn_fdf}")

    df = Ode(ecs).eval_save(fn_fdf)
    print(f"wrote file {fn_fdf}")
    Ode.print_df(df)

    end = datetime.now()
    print("runtime:", f"start: {start}", f"end: {end}",
            f"duration: {end-start}", sep = "\n")
    return fn_fdf, df


def plot_file(fn, title):
    fdf, f = Ode.facets_f(fn,
                 kind = "box",
                 col_wrap = 2,
                 height = 5,
                 aspect = 3,
                 title = title)

    print(fdf.head())

    plt.tight_layout()
    plt.subplots_adjust(top = .92, left = .04, right = .9)
    fn_plot = p_join(GENDIR, f"plot_{Path(fn).stem}.pdf")
    f.savefig(fn_plot)
    print(f"saved plot to {fn_plot}")
    plt.show()

###############################################################################

def gen_mnist_files(n_col = 5, bc = 0, rm = 0.005, n_s = 10):
    """This generates npz files with Mnist OD datasets.
    """
    for i in range(n_col):
        dn = p_join(DATASETDIR,
                    f"ova_mnist_bc{bc}_rm{rm}_s{n_s}_{i}",
                    DS_SUB_DIR)
        Mnist(flatten = False).npz_ova_bds(bc, rm, n_s, dn)


def gen_small_mnist_files(n_col = 1, bc = 0, rm = 0.01, n_s = 2):
    """This generates npz files with Mnist OD datasets.
    """
    tc = 0
    ec = 2000
    for i in range(n_col):
        dn = p_join(DATASETDIR,
                    f"small_ova_mnist_bc{bc}_rm{rm}_s{n_s}_{i}",
                    DS_SUB_DIR)
        Mnist(t_count = tc, e_count = ec, flatten = False).\
                npz_ova_bds(bc, rm, n_s, dn)


def gen_fmnist_files(n_col = 5, bc = 0, rm = 0.005, n_s = 10):
    """This generates npz files with FMnist OD datasets.
    """
    for i in range(n_col):
        dn = p_join(DATASETDIR,
                    f"ova_fmnist_bc{bc}_rm{rm}_s{n_s}_{i}",
                    DS_SUB_DIR)
        FMnist(flatten = False).npz_ova_bds(bc, rm, n_s, dn)


def gen_ds(n_col = 5, bcs = [0, 1], rms = [0.005], n_s = 10):
    for bc in bcs:
        for rm in rms:
            gen_mnist_files(n_col, bc, rm, n_s)
            gen_fmnist_files(n_col, bc, rm, n_s)
