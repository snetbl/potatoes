#!/usr/bin/env python

from os.path import join as p_join, splitext
from sys import path as s_path, exit
from pathlib import Path
PROJDIR = str(Path(__file__).resolve().parents[1])
SRCDIR = p_join(PROJDIR, 'src')
s_path.append(SRCDIR)

import os
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import pi as pi
from matplotlib import pyplot as plt

plt.style.use("ggplot")

# noinspection PyUnresolvedReferences
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

from tools import flatten

import od_conf
from od_conf import DATE_TIME_FORMAT, DATASETDIR, DS_SUB_DIR


def p(s, sep = '', end = '\n', flush = False):
    if not od_conf.silent:
        print(s, sep = sep, end = end, flush = flush)


def precs(df, lab_col, pred_col, true_lab):
    ndf = df.assign(lab_t_f=lambda x: x[lab_col] == true_lab)
    return precs_t_f(ndf, "lab_t_f", pred_col)


def precs_t_f(df, lab_col, pred_col):
    sdf = df.sort_values(by=pred_col, ascending=False)
    cs = sdf[lab_col].cumsum()
    return cs / np.arange(1, len(df.index) + 1)


def rcs(df, lab_col, pred_col, true_lab):
    ndf = df.assign(lab_t_f=lambda x: x[lab_col] == true_lab)
    return rcs_t_f(ndf, "lab_t_f", pred_col)


def rcs_t_f(df, lab_col, pred_col):
    sdf = df.sort_values(by=pred_col, ascending=False)
    t_t = df[lab_col].sum()
    cs = sdf[lab_col].cumsum()
    return cs / t_t


def f1s(df, lab_col, pred_col, true_lab):
    ndf = df.assign(lab_t_f=lambda x: x[lab_col] == true_lab)
    return f1s_t_f(ndf, "lab_t_f", pred_col)


def f1s_t_f(df, lab_col, pred_col):
    recalls = rcs_t_f(df, lab_col, pred_col)
    prs = precs_t_f(df, lab_col, pred_col)
    return 2 * prs * recalls / (prs + recalls)


def opt_f1(df, lab_col, pred_col, true_lab):
    ndf = df.assign(lab_t_f=lambda x: x[lab_col] == true_lab)
    return opt_f1_t_f(ndf, "lab_t_f", pred_col)


def opt_f1_t_f(df, lab_col, pred_col):
    return np.nanmax(f1s_t_f(df, lab_col, pred_col))


class LabData:
    DATADIR = "reference_datasets"
    COL_PRE = "f"
    LABEL_COLUMN = "label"
    NPZ_DATA_NAME = 'data'
    NPZ_LABELS_NAME = 'labels'

    def __init__(self, *args, **kwargs):
        success = False
        for fun in [self.init1,self.init2, self.init3, self.init4, self.init5]:
            try:
                fun(*args, **kwargs)
            except (TypeError, AssertionError):
                # p(e)
                continue
            success = True
            break
        if not success:
            raise TypeError("in LabData: no fitting ctor found")

    def init1(self, X, ls, name):
        assert type(X) is np.ndarray
        assert hasattr(ls, "__iter__")
        assert hasattr(ls, "__getitem__")
        assert type(name) is str
        self.X = X
        self.ls = np.array(ls)
        self.name = name

    def init2(self, i_Xs, ls, name):
        assert type(i_Xs) is not str
        assert hasattr(i_Xs, "__iter__")
        assert hasattr(i_Xs, "__getitem__")
        assert type(ls) is not str
        assert hasattr(ls, "__iter__")
        assert hasattr(ls, "__getitem__")
        assert type(name) is str
        lens = [len(X) for X in i_Xs]
        lsr = np.array(ls).repeat(lens)
        self.init1(np.concatenate(i_Xs, axis=0), lsr, name)

    def init3(self, i_Xs, name):
        assert hasattr(i_Xs, "__iter__")
        assert hasattr(i_Xs, "__getitem__")
        assert type(name) is str
        self.init2(i_Xs, range(len(i_Xs)), name)

    def init4(self, fn, name, lc = None, dtype = np.float64):
        assert type(fn) is str and type(name) is str
        assert splitext(fn)[1] == ".csv"
        assert lc is None or type(lc) is str

        ldf = pd.read_csv(fn)
        if lc is None:
            lab_col = self.LABEL_COLUMN
        else:
            lab_col = lc
        ls = ldf[lab_col].to_numpy()

        cs = [c for c in ldf.columns if c != lab_col]
        X = ldf[cs].to_numpy(dtype)
        self.init1(X, ls, name)

    def init5(self, fn, name):
        assert type(fn) is str and type(name) is str
        assert splitext(fn)[1] == ".npz"
        r_ld = np.load(fn)
        self.init1(r_ld[self.NPZ_DATA_NAME], r_ld[self.NPZ_LABELS_NAME], name)

    def get_labeled(self, l):
        return self.X[self.ls == l]

    def get_labeled_c(self, l):
        return self.X[self.ls != l]

    def get_l_count(self, l):
        return self.get_labeled(l).shape[0]

    def get_lc_count(self, l):
        return self.get_labeled_c(l).shape[0]

    def make_ova(self, bc, rm, n_trials):
        b = self.X[self.ls == bc]
        m = self.X[self.ls != bc]
        n_m = int(b.shape[0] * rm / (1 - rm))
        if n_m > m.shape[0]:
            raise ValueError(
                f"The amount of malign data points {m.shape[0]} is " \
                + f"too small to obtain the required malign ratio {rm}, " \
                + f"which would require at least {n_m} data points.")


        ms = [m[np.random.choice(m.shape[0], n_m, replace=False)]
              for _ in range(n_trials)]

        return b, ms

    def make_avo(self, mc, rm, n_trials):
        b = self.X[self.ls != mc]
        m = self.X[self.ls == mc]
        n_m = int(b.shape[0] * rm / (1 - rm))
        if n_m > m.shape[0]:
            raise ValueError(
                f"The amount of malign data points {m.shape[0]} is " \
                + f"too small to obtain the required malign ratio {rm}, " \
                + f"which would require at least {n_m} data points.")

        ms = [m[np.random.choice(m.shape[0], n_m, replace=False)]
              for _ in range(n_trials)]
        return b, ms

    def make_ova_bds(self, bc, rm, n_bds):
        b, ms = self.make_ova(bc, rm, n_bds)
        name = f"ova_{self.name}_bc_{bc}_rm{rm:g}" + "_{i:03d}"
        return [BinData(1, [b, m], name.format(**vars()))
                for i, m in enumerate(ms)]

    def csv_ova_bds(self, bc, rm, n_bds, dr = None, col_pre = "f",
            lcn = "label"):
        if dr is None:
            dt = datetime.now().strftime(DATE_TIME_FORMAT + "_%f")
            dr = p_join(DATASETDIR, f"ova_{self}_bc{bc}_rm{rm}_s{n_bds}_{dt}",
                    DS_SUB_DIR)
        os.makedirs(dr, exist_ok = True)

        for bd in self.make_ova_bds(bc, rm, n_bds):
            fn = p_join(dr, str(bd))
            bd.save_csv(fn, col_pre, lcn)

    def npz_ova_bds(self, bc, rm, n_bds, dr = None):
        if dr is None:
            dt = datetime.now().strftime(DATE_TIME_FORMAT + "_%f")
            dr = p_join(DATASETDIR, f"ova_{self}_bc{bc}_rm{rm}_s{n_bds}_{dt}",
                    DS_SUB_DIR)
        os.makedirs(dr, exist_ok = True)

        for bd in self.make_ova_bds(bc, rm, n_bds):
            fn = p_join(dr, str(bd))
            bd.save_npz(fn)

        return dr

    def make_ova_bd(self, bc, rm):
        b, ms = self.make_ova(bc, rm, 1)
        name = f"ova {self.name} bc {bc} rm {rm:0.2f}"
        return BinData(1, [b, ms[0]], name)

    def make_avo_bds(self, mc, rm, n_bds):
        b, ms = self.make_avo(mc, rm, n_bds)
        name = f"avo {self.name} mc {mc} rm {rm}" + " {i:03d}"
        return [BinData(1, [b, m], name.format(**vars())) \
                for i, m in enumerate(ms)]

    def make_avo_bd(self, mc, rm):
        b, ms = self.make_avo(mc, rm, 1)
        name = f"ova {self.name} mc {mc} rm {rm:0.2f}"
        return BinData(1, [b, ms[0]], name)

    def __str__(self):
        return str(self.name)

    def save_csv(self, fn, col_pre = None, lc = None):
        if col_pre is None:
            col_pre = self.COL_PRE
        if lc is None:
            lc = self.LABEL_COLUMN
        X_fl = self.get_flt()
        cs = [col_pre+str(i) for i in range(X_fl.shape[1])]
        xdf = pd.DataFrame(X_fl, columns = cs)
        kwargs = {lc: self.ls}
        fdf = xdf.assign(**kwargs)

        fdf.to_csv(fn, index = False)

    def save_npz(self, fn):
        kwargs = {self.NPZ_DATA_NAME: self.X, self.NPZ_LABELS_NAME: self.ls}
        np.savez_compressed(fn, **kwargs)

    def get_flt(self):
        return flatten(self.X)

    @staticmethod
    def load_bds(col):
        dn = p_join(GENDIR, col, DS_SUB_DIR)
        bds = [BinData(ol_lab = 1, fn = p_join(dn, fn), name = splitext(fn)[0])
                for fn in os.listdir(dn)]
        return bds


class BinData(LabData):
    def __init__(self, ol_lab, *args, **kwargs):
        self.ol_lab = ol_lab
        super(BinData, self).__init__(*args, **kwargs)

    def get_b(self):
        return self.get_labeled_c(self.ol_lab)

    def get_m(self):
        return self.get_labeled(self.ol_lab)

    def get_b_count(self):
        return self.get_b().shape[0]

    def get_m_count(self):
        return self.get_m().shape[0]


class Mnist(LabData):
    def __init__(self, name = "mnist", norm = True, cast32 = True,
            t_count=None, e_count=None, flatten = True, add_channel = False):
        self.norm = norm
        self.cast32 = cast32
        self.t_count = t_count
        self.e_count = e_count
        self.flatten = flatten
        self.add_channel = add_channel
        (self.td, self.tl), (self.ed, self.el) = \
            self.get_data(norm, cast32, t_count, e_count, flatten, add_channel)
        super(Mnist, self).__init__(np.r_[self.td, self.ed],
                                    np.r_[self.tl, self.el],
                                    name)

    @staticmethod
    def get_data(norm = True, cast32 = True, t_count=None, e_count=None,
            flatten = True, add_channel = False):
        ((td, tl), (ed, el)) = mnist.load_data()

        if flatten:
            (td, tl), (ed, el) = ((td.reshape(-1, 28 * 28), tl),
                                  (ed.reshape(-1, 28 * 28), el))

        if add_channel:
            (td, tl), (ed, el) = ((td[..., None], tl),
                                  (ed[..., None], el))

        if norm:
            td = td/255.
            ed = ed/255.
            if cast32:
                td = td.astype(np.float32)
                ed = ed.astype(np.float32)
        if t_count is not None:
            inds = np.random.choice(td.shape[0], t_count, replace=False)
            td, tl = td[inds], tl[inds]
        if e_count is not None:
            inds = np.random.choice(ed.shape[0], e_count, replace=False)
            ed, el = ed[inds], el[inds]

        return (td, tl), (ed, el)


class FMnist(LabData):

    def __init__(self, name = "fmnist", norm = True, cast32 = True,
            t_count=None, e_count=None, flatten = True, add_channel = False):
        self.norm = norm
        self.cast32 = cast32
        self.t_count = t_count
        self.e_count = e_count
        self.flatten = flatten
        self.add_channel = add_channel
        (self.td, self.tl), (self.ed, self.el) = \
            self.get_data(norm, cast32, t_count, e_count, flatten, add_channel)
        super(FMnist, self).__init__(np.r_[self.td, self.ed],
                                     np.r_[self.tl, self.el], name)

    @staticmethod
    def get_data(norm = True, cast32 = True, t_count=None, e_count=None,
            flatten = True, add_channel = False):
        ((td, tl), (ed, el)) = fashion_mnist.load_data()

        if flatten:
            (td, tl), (ed, el) = ((td.reshape(-1, 28 * 28), tl),
                                  (ed.reshape(-1, 28 * 28), el))

        if add_channel:
            (td, tl), (ed, el) = ((td[..., None], tl),
                                  (ed[..., None], el))

        if norm:
            td = td/255.
            ed = ed/255.
            if cast32:
                td = td.astype(np.float32)
                ed = ed.astype(np.float32)
        if t_count is not None:
            inds = np.random.choice(td.shape[0], t_count, replace=False)
            td, tl = td[inds], tl[inds]
        if e_count is not None:
            inds = np.random.choice(ed.shape[0], e_count, replace=False)
            ed, el = ed[inds], el[inds]

        return (td, tl), (ed, el)


class Cifar10(LabData):

    def __init__(self, name = "cifar10", norm = True, cast32 = True,
            t_count=None, e_count=None, flatten = True):
        self.norm = norm
        self.cast32 = cast32
        self.t_count = t_count
        self.e_count = e_count
        self.flatten = flatten
        (self.td, self.tl), (self.ed, self.el) = \
            self.get_data(norm, cast32, t_count, e_count)
        super(Cifar10, self).__init__(np.r_[self.td, self.ed],
                                      np.r_[self.tl, self.el], name)
        #self.shape = self.td.shape

    @staticmethod
    def get_data(norm = True, cast32 = True, t_count=None, e_count=None,
            flatten = True):
        (td, tl), (ed, el) = cifar10.load_data()

        if flatten:
            (td, tl), (ed, el) = ((td.reshape(-1, 32*32*3), tl.reshape(-1)),
                                  (ed.reshape(-1, 32*32*3), el.reshape(-1)))

        if norm:
            td = td/255.
            ed = ed/255.
            if cast32:
                td = td.astype(np.float32)
                ed = ed.astype(np.float32)
        if t_count is not None:
            inds = np.random.choice(td.shape[0], t_count, replace=False)
            td, tl = td[inds], tl[inds]
        if e_count is not None:
            inds = np.random.choice(ed.shape[0], e_count, replace=False)
            ed, el = ed[inds], el[inds]

        return (td, tl), (ed, el)


class Cifar100(LabData):
    def __init__(self, name = None, norm = True, cast32 = True,
            label_mode = "fine", t_count=None, e_count=None, flatten = True):
        self.norm = norm
        self.cast32 = cast32
        self.label_mode = label_mode
        self.t_count = t_count
        self.e_count = e_count
        self.flatten = flatten
        if name is None:
            name = f"cifar100 {label_mode}"
        (self.td, self.tl), (self.ed, self.el) = \
            self.get_data(norm, cast32, label_mode, t_count, e_count)
        super(Cifar100, self).__init__(np.r_[self.td, self.ed],
                                       np.r_[self.tl, self.el],
                                       name)

    @staticmethod
    def get_data(norm = True, cast32 = True, label_mode = "fine", t_count=None,
            e_count=None, flatten = True):
        (td, tl), (ed, el) = cifar100.load_data(label_mode = label_mode)

        if flatten:
            (td, tl), (ed, el) = ((td.reshape(-1, 32*32*3), tl.reshape(-1)),
                                  (ed.reshape(-1, 32*32*3), el.reshape(-1)))
        tl = tl.astype(np.uint8)
        el = el.astype(np.uint8)

        if norm:
            td = td/255.
            ed = ed/255.
            if cast32:
                td = td.astype(np.float32)
                ed = ed.astype(np.float32)
        if t_count is not None:
            inds = np.random.choice(td.shape[0], t_count, replace=False)
            td, tl = td[inds], tl[inds]
        if e_count is not None:
            inds = np.random.choice(ed.shape[0], e_count, replace=False)
            ed, el = ed[inds], el[inds]

        return (td, tl), (ed, el)


class Sine(BinData):
    def __init__(self, ms, name = "sine", n_ps = 2, n_b = 50, norm = (.1,.1),
            cast32 = True):
        self.d, self.l = self.get_data(ms, n_ps, n_b, norm, cast32)
        super(Sine, self).__init__(1, self.d, self.l, name)

    @staticmethod
    def get_data(ms, n_ps = 2, n_b = 50, norm = (.1,.1), cast32 = True):
        na = np.array(norm)
        x = np.linspace(-n_ps*pi, n_ps*pi, n_b)
        b = np.stack([x, np.sin(x)], axis = -1) * na
        m = ms * na

        d = np.concatenate([b, m], axis = 0)
        labs = np.r_[np.zeros(b.shape[0]),
                     np.ones (m.shape[0])]

        if cast32:
            d = d.astype(np.float32)

        return d, labs


class Sine50(Sine):
    def __init__(self):
        ms = np.array([
                [-pi/2, 0.],
                [-3/16*pi, 0.],
                [1/4*pi, .25],
                [7/8*pi, -.25],
            ])
        super(Sine50, self).__init__(ms, n_b = 50)
        
        
class SineRnd(Sine):

    def __init__(self, n_ps = 2, total = 100, n_m = 5, lb = .3, ub = .5):
        zs_x = (np.arange(0, 2*n_ps+1) - n_ps)[1:-1] * pi
        zs = np.r_["-1, 2, 0", zs_x, np.zeros(zs_x.shape[0])]

        def gen():
            print("\nin gen()")
            n_p_x = (np.random.random_sample() - .5) * n_ps*2*pi
            n_p = np.array([n_p_x, 0])
            print(n_p)
            ds = np.sqrt(np.sum((n_p - zs)**2, axis = -1))
            print(ds)
            if lb <= np.amin(ds) <= ub:
                return n_p
            else:
                return gen()

        ms = np.array([gen() for _ in range(n_m)])
        print("\n---- ms ----")
        print(ms)

        super(SineRnd, self).__init__(
                ms, f"sine_t{total:d}_m{n_m:d}", n_ps = n_ps, n_b = total-n_m)
        
        
class Circle(BinData):
    def __init__(self, ms, name = "circle", n_b = 50, r = .5):
        self.d, self.l = self.get_data(ms, n_b)
        super(Circle, self).__init__(1, self.d, self.l, name)

    @staticmethod
    def get_data(ms, n_b = 50, r = .5):
        assert ms.dtype == np.float32
        c = np.array(list(map(lambda s: [r*np.cos(2*pi/n_b * s),
                                         r*np.sin(2*pi/n_b * s)],
                              range(n_b))),
                     dtype = np.float32)
        d = np.concatenate([c, ms], axis = 0)
        labs = np.r_[[0]*n_b, [1]*ms.shape[0]].astype(np.int32)
        return d, labs


class CircleRnd(Circle):
    def __init__(self, total = 200, n_im = 10, n_om = 10, r = .5, inr = .3,
            outr = .7):
        im = np.array(list(map(lambda s: [inr*np.cos(2*pi * s),
                                          inr*np.sin(2*pi * s)],
                               np.random.default_rng().random(n_im))),
                      dtype = np.float32)
        om = np.array(list(map(lambda s: [outr*np.cos(2*pi * s),
                                          outr*np.sin(2*pi * s)],
                               np.random.default_rng().random(n_im))),
                      dtype = np.float32)
        ms = np.concatenate([im, om], axis = 0)
        n_m = ms.shape[0]

        super(CircleRnd, self).__init__(
                ms, f"circle_t{total:d}_m{n_m:d}", n_b = total-n_m, r = r)


###############################################################################
if __name__ == '__main__':
    FMnist("testing fashion mnist")
