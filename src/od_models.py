#!/usr/bin/env python

import sys
from os.path import join as p_join
from sys import path as s_path, maxsize, exit
from pathlib import Path, PurePath
PROJDIR = str(Path(__file__).resolve().parents[1])
SRCDIR = p_join(PROJDIR, 'src')
s_path.append(SRCDIR)
GENDIR = p_join(PROJDIR, 'generatedData')
LOGDIR = p_join(PROJDIR, 'logs')
LOGFILE = p_join(LOGDIR, "log")
CKPDIR = p_join(PROJDIR, 'checkpoints')

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro
from datetime import datetime
import re
import os
from pprint import pprint
from functools import reduce
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,\
        LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, \
        Reshape, Conv2DTranspose, UpSampling2D, Dense, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
#from tfdocs.modeling import EpochDots # this is only in tf2.2

from tools import time_it, bo_norms, close_pairs, ex_fi_fun, ewedge, pipe, \
        flatten
from od_tools import p, opt_f1, Mnist, FMnist, BinData, Cifar10, Cifar100

import  od_conf
from od_conf import DATE_TIME_FORMAT


AELT = 0.02
RF = lambda bd, odm, metric: 10


def log(msg):
    dt = datetime.now().strftime(DATE_TIME_FORMAT)
    os.makedirs(LOGDIR, exist_ok = True)
    p(msg)
    with open(LOGFILE, 'a') as fo:
        fo.write(f"{dt} {msg}\n")


class EvalConf:
    BD_COL = "bd"
    BDGID_COL = "data"
    ODM_COL = "OD model"
    MET_COL = "metric"
    RT_COL = "retry"

    def __init__(self, bd, odm, met, bdgid = 0, retry = 10):
        self.bd = bd
        self.odm = odm
        self.met = met
        self.bdgid = bdgid
        self.retry = retry

    def __str__(self):
        return f"{self.bd}, {self.bdgid}, {self.odm}, {self.met}, {self.retry}"

    @staticmethod
    def cross_comb(bds, odms, mets, bdgids = None, rf = RF):
        if bdgids is None:
            ret = [EvalConf(bd, odm, met, retry = rf(bd, odm, met))
                    for bd in bds
                    for odm in odms
                    for met in mets]
        else:
            ret = [EvalConf(bd, odm, met, bdgids[i], rf(bd, odm, met))
                    for i,bd in enumerate(bds)
                    for odm in odms
                    for met in mets]
        return ret

    @staticmethod
    def cross_comb_df(bds, odms, mets, bdgids = None, rf = RF):
        ecs = EvalConf.cross_comb(bds, odms, mets, bdgids, rf)
        mat = [[e.bd, e.bdgid, e.odm, e.met, e.retry] for e in ecs]
        return pd.DataFrame(mat, columns = (EvalConf.BD_COL, EvalConf.BDGID_COL,
            EvalConf.ODM_COL, EvalConf.MET_COL, EvalConf.RT_COL))

    @staticmethod
    def l2df(econfs):
        mat = [[e.bd, e.bdgid, e.odm, e.met, e.retry] for e in econfs]
        return pd.DataFrame(mat, columns = (EvalConf.BD_COL,
            EvalConf.BDGID_COL, EvalConf.ODM_COL, EvalConf.MET_COL,
            EvalConf.RT_COL))

    @staticmethod
    def l2sdf(econfs):
        mat = [[str(e.bd), str(e.bdgid), str(e.odm), str(e.met), str(e.retry)]
                for e in econfs]
        return pd.DataFrame(mat,
                columns = (EvalConf.BD_COL, EvalConf.BDGID_COL,
                           EvalConf.ODM_COL, EvalConf.MET_COL, EvalConf.RT_COL)
                ).astype(str)

    @staticmethod
    def make_bdgid(ld, bc, rm):
        return f"ova_{ld}_bc_{bc}_rm{rm:g}"

    @staticmethod
    def ovas(lds, bcs, rms, l_n_s, odms, mets, rf = RF):
        assert len(lds) == len(bcs) == len(rms) == len(l_n_s) == len(odms) \
                == len(mets)
        nbds = [lds[i].make_ova_bds(bcs[i], rms[i], l_n_s[i])
                for i in range(len(lds))]
        bdgids = [EvalConf.make_bdgid(ld, bc, rm)
                     for ld, bc, rm in zip(lds, bcs, rms)]
        ecs = [EvalConf(bd, odms[i], mets[i], bdgids[i],
                        rf(bd, odms[i], mets[i]))
               for i, bds in enumerate(nbds)
               for bd in bds
              ]
        return ecs

    @staticmethod
    def ovas_cross(lds, bcs, rms, l_n_s, odms, mets, rf = RF):
        assert len(lds) == len(bcs) == len(rms) == len(l_n_s)
        nbds = [lds[i].make_ova_bds(bcs[i], rms[i], l_n_s[i]) \
                   for i in range(len(lds))]
        bdgids = [EvalConf.make_bdgid(ld, bc, rm)
                     for ld, bc, rm in zip(lds, bcs, rms)]
        ecs = [EvalConf(bd, odm, met, bdgids[i], rf(bd, odm, met))
               for i, bds in enumerate(nbds)
               for bd in bds
               for odm in odms
               for met in mets
              ]
        return ecs

    @staticmethod
    def ovas_cross_s(ulds, ubcs, rm, n_s, odms, mets, rf = RF):
        lds = np.array(ulds).repeat(len(ubcs))
        bcs = np.tile(ubcs, len(ulds))
        rms = np.array([rm]).repeat(bcs.shape[0])
        l_n_s = [n_s]*bcs.shape[0]

        return EvalConf.ovas_cross(lds, bcs, rms, l_n_s, odms, mets, rf)


class Ode:
    LAB_COL = 'lab'
    OS_COL = 'pred'
    METVAL_COL = 'metric value'
    CC = "count"
    DIFF_COL = "diff"

    def __init__(self, e_confs):
        self.ecs = e_confs
        self.os_cache = dict()
        p(f"\n\n{'#'*80}\n{'#'*80}\nOde.__init__\n")
        p("evaluation configurations self.ecs:")
        for c in self.ecs:
            p(c)

    def eval_od(self, bd, odm, met):
        p("\n**********")
        p("in eval_od")
        p("**********")
        p("\n---- bd ----")
        p(bd)
        p(f"b: {bd.get_b_count()}, m: {bd.get_m_count()}")
        p(bd.ls.shape)
        p("\n---- odm ----")
        p(odm)
        p("\n---- met ----")
        p(met)
        p("\n")

        s_cache = len(self.os_cache)
        if (bd, odm) in self.os_cache:
            hit = True
            p(f"outlier scores cache hit ({s_cache} entries)")
            ols = self.os_cache[(bd, odm)]
        else:
            hit = False
            p(f"outlier scores cache miss ({s_cache} entries)")
            ols = odm.olds(bd.X)
            if ols is not None:
                self.os_cache[(bd, odm)] = ols
            else:
                log(f"ERROR: not converged: {bd} {odm} {met}")

        if ols is not None:
            osdf = pd.DataFrame({Ode.LAB_COL: bd.ls, Ode.OS_COL: ols})
            ret = met.met(osdf, Ode.LAB_COL, Ode.OS_COL, bd.ol_lab)
        else:
            if not hit:
                p(f"\n\n{'%'*80}\nWARNING: model {odm} did not converge "\
                        +f"on data '{bd}'\n{'%'*80}")
            ret = np.nan
        return hit, ret
    
    def c_odms(self):
        return list(set([k[1] for k in self.os_cache.keys()]))

    def eval_od_r(self, c, tc):
        p(f"\n\n{'='*40}\ntrials left: {tc}")
        if tc == 0:
            self.os_cache[(c.bd, c.odm)] = None
            return np.nan

        hit, ev = self.eval_od(c.bd, c.odm, c.met)
        if ev is np.nan:
            if hit:
                p(f"This evaluation already failed {tc} times earlier, so this "\
                        +f"is skipped.")
                return np.nan
            else:
                return self.eval_od_r(c, tc-1)
        else:
            return ev

    def eval_od_all(self):
        return [self.eval_od_r(c, c.retry) for c in self.ecs]

    def eval_od_all_df(self):
        met_vals = self.eval_od_all()
        ecdf = EvalConf.l2df(self.ecs)
        return ecdf.assign(**{Ode.METVAL_COL: met_vals})

    def eval_od_all_msd(self):
        df = self.eval_od_all_df()
        gdf = df.groupby(
            [EvalConf.BDGID_COL, EvalConf.ODM_COL, EvalConf.MET_COL],
            sort = False)
        means = gdf[Ode.METVAL_COL].mean()
        means = means.rename('mean')

        stds = gdf[Ode.METVAL_COL].std()
        stds = stds.rename('std')

        msdf = pd.concat([means, stds], axis = 1)
        return df, msdf

    def eval_od_all_desc(self):
        df = self.eval_od_all_df()
        gdf = df.groupby(
            [EvalConf.BDGID_COL, EvalConf.ODM_COL, EvalConf.MET_COL],
            sort = False)
        met_stats = gdf[Ode.METVAL_COL].describe()
        met_stats.columns =\
            ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        return df, met_stats


    @staticmethod
    def pivot_bdgid(ndf):
        mcol = "mean"
        gdf = ndf.groupby(
            [EvalConf.BDGID_COL, EvalConf.ODM_COL, EvalConf.MET_COL],
            sort = False)
        gdfm = gdf[Ode.METVAL_COL].mean().to_frame().\
                rename(columns = {Ode.METVAL_COL: mcol})
        pdf = pd.pivot_table(gdfm,
                             index = EvalConf.BDGID_COL,
                             columns = [EvalConf.ODM_COL, EvalConf.MET_COL],
                             values = mcol)
        return pdf

    @staticmethod
    def pivot_bdgid_f(f):
        ndf = pd.read_csv(f)
        return Ode.pivot_bdgid(ndf)

    @staticmethod
    def pivot_bdgid_sm_f(f, met):
        rdf = pd.read_csv(f)
        ndf = rdf[rdf[EvalConf.MET_COL] == met]
        return Ode.pivot_bdgid(ndf)

    @staticmethod
    def eval(odm, mets, X, ls, ol_lab, rf = RF):
        bd = BinData(ol_lab, X, ls, "bdRocOf1")
        ecs = [EvalConf(bd, odm, met, rf(bd, odm, met)) for met in mets]
        ode = Ode(ecs)
        met_vals = ode.eval_od_all()
        return met_vals, ode

    @staticmethod
    def eval_ova(odm, mets, ld, bc, rm, n_s, rf = RF):
        bd_col = 'bd'
        met_col = 'met'

        bds = ld.make_ova_bds(bc, rm, n_s)
        e_confs = EvalConf.cross_comb(bds, [odm], mets, rf = rf)
        ecdf = pd.DataFrame([[ec.bd.name, ec.met.name] for ec in e_confs],
                            columns = (bd_col, met_col))
        eva = Ode(e_confs)
        met_vals = eva.eval_od_all()
        tmp_df = ecdf.assign(met_val = met_vals)
        ret_df = tmp_df.pivot(index = bd_col, columns = met_col, values='met_val')
        return ret_df, eva

    @staticmethod
    def facets_f(f, kind, col_wrap, height, aspect, title):
        fdf = pd.read_csv(f)
        return fdf, Ode.facets(fdf, kind, col_wrap, height, aspect, title)

    @staticmethod
    def facets(fdf, kind, col_wrap, height, aspect, title):
        f = sns.catplot(data = fdf, kind = kind, x = EvalConf.ODM_COL,
                y = Ode.METVAL_COL, hue = EvalConf.MET_COL, col=EvalConf.BDGID_COL,
                col_wrap = col_wrap, height = height, aspect = aspect)
        #f = sns.catplot(data = fdf, kind = kind, x = EvalConf.ODM_COL,
        #        y = Ode.METVAL_COL, hue = EvalConf.MET_COL, col=EvalConf.BDGID_COL,
        #        col_wrap = col_wrap)
        f.fig.suptitle(title)
        for ax in f.axes:
            ax.set_title(ax.get_title(), fontsize = 8)

        return f

    @staticmethod
    def ova_fig(lds, bcs, rms, l_n_s, odms, mets, title, kind = "violin",
            col_wrap = 5, height = 3, aspect = 1, rf = RF):
        # do all the evaluation
        ecs = EvalConf.ovas_cross(lds, bcs, rms, l_n_s, odms, mets, rf)
        etor = Ode(ecs)
        fdf, ddf = etor.eval_od_all_desc()

        # create the seaborn FacetGrid object with catplot
        f = Ode.facets(fdf, kind, col_wrap, height, aspect, title)

        return fdf, ddf, f

    def eval_save(self, fn):
        df = self.eval_od_all_df()
        df.to_csv(fn, index = False)

        return df

    @staticmethod
    def eval_print(ecs):
        etor = Ode(ecs)
        df = etor.eval_od_all_df()
        return Ode.print_df(df)

    @staticmethod
    def print_df(df):
        df2 = df.astype({EvalConf.BD_COL: str, EvalConf.MET_COL: str, EvalConf.ODM_COL: str})
        pdf = pd.pivot_table(df2,
                index = EvalConf.BD_COL,
                columns = [EvalConf.MET_COL, EvalConf.ODM_COL],
                values = Ode.METVAL_COL)

        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 300)

        print("\n\n---- evaluation results, wide format (pivoted) ----")
        print(pdf)
        print("\n\n---- evaluation results, summaries ----")
        if len(pdf.index) == 0:
            print("no data, nothing to display")
        else:
            print(pdf.describe())

        return pdf

    @staticmethod
    def print_file(f):
        df = pd.read_csv(f)
        return Ode.print_df(df)

    @staticmethod
    def cmp_odms(odms, dss, bcs, rm, n_s, mets, fd):
        mets = [Ap(), RocAuc(), KPr(k=20), OF1()]
        dt = datetime.now().strftime(DATE_TIME_FORMAT)
        dsss = " ".join([str(ds) for ds in dss])
        bclzs = "[" + ','.join([str(n) for n in bcs]) + "]"
        odmss = " ".join([str(m) for m in odms])
        metss = " ".join([str(m) for m in mets])
        cs_ova = CsOva(dsss, bclzs, rm, n_s, odmss, metss, dt, 'tune')

        ecs = EvalConf.ovas_cross_s(dss, bcs, rm, n_s, odms, mets)
        etor = Ode(ecs)

        fn = p_join(fd, cs_ova.get_fn())
        pdf = Ode.print_df(etor.eval_save(fn))
        p(f"wrote file {fn}")

        return pdf

    @staticmethod
    def md(odedf, met, m1, m2):
        assert str == type(m1) == type(m2) == type(met)

        fm = odedf[odedf[EvalConf.MET_COL] == met]\
                [[EvalConf.BD_COL, EvalConf.ODM_COL, Ode.METVAL_COL]]
        ffm = fm[(fm[EvalConf.ODM_COL] == m1) | (fm[EvalConf.ODM_COL] == m2)]
        
        gb = ffm.groupby([EvalConf.BD_COL, EvalConf.ODM_COL]).count().\
                rename(columns = {Ode.METVAL_COL: Ode.CC})
        exc = gb[gb[Ode.CC] > 1]
        if len(exc.index) > 0:
            p(f"WARNING: there are duplicates (they will be combined "\
                    +"with np.mean()):")
            p(exc)

        pffm = ffm.pivot_table(values = Ode.METVAL_COL,
                               index = EvalConf.BD_COL,
                               columns = EvalConf.ODM_COL)
        ddf = pffm.assign(**{Ode.DIFF_COL: pffm[m1] - pffm[m2]})

        if ddf[Ode.DIFF_COL].isnull().any():
            nanr = ddf[ddf.isnull().any(axis=1)]
            p(f"WARNING: there are missing combinations of model and metric (they will be dropped):")
            p(nanr)
            ddf = ddf[ddf[Ode.DIFF_COL].notnull()]

        return ddf


    @staticmethod
    def ptt(odedf, met, m1, m2):
        ddf = Ode.md(odedf, met, m1, m2)
        ptt = ttest_rel(ddf[m1].values, ddf[m2].values)
        return ddf, ptt

    @staticmethod
    def ptt_f(f, met, m1, m2):
        df = pd.read_csv(f)
        return Ode.ptt(df, met, m1, m2)


class CsOva:
    fn_start = "ova_eval"
    order = ['dss', 'bcs',  'rm', 'n_s', 'odms', 'mets']
    types = [  str,   str, float,   int,    str,    str]
    fn_pre = ['dt', 'cont']

    def __init__(self, *args, **kwargs):
        ex_fi_fun([self.vals_init, self.fn_init], *args, **kwargs)

    def vals_init(self, dss, bcs_l, rm, n_s, odms, mets, dt, cont):
        self.dss = dss
        self.bcs_l = bcs_l
        self.bcs = str(self.bcs_l)
        self.rm = rm
        self.n_s = n_s
        self.odms = odms
        self.mets = mets
        self.dt = dt
        self.cont = cont
        self.cd = {'dss': self.dss,
                   'bcs_l': self.bcs_l,
                   'bcs': self.bcs,
                   'rm': self.rm,
                   'n_s': self.n_s,
                   'odms': self.odms,
                   'mets': self.mets,
                  }
        self.cs = CsOva.to_str(self.cd)
        f_spec = {'dt': self.dt, 'cont': self.cont}
        self.fnd = {**self.cd, **f_spec}
        self.fn = CsOva.dict_to_fn(self.fnd)

    def fn_init(self, fn):
        self.fn = fn
        self.fnd = CsOva.fn_to_dict(self.fn)
        self.dss = self.fnd['dss']
        self.bcs = self.fnd['bcs']
        self.bcs_l = eval(self.bcs)
        self.rm = self.fnd['rm']
        self.n_s = self.fnd['n_s']
        self.odms = self.fnd['odms']
        self.mets = self.fnd['mets']
        self.dt = self.fnd['dt']
        self.cont = self.fnd['cont']
        self.cd = {'dss': self.dss,
                   'bcs': self.bcs,
                   'bcs_l': self.bcs_l,
                   'rm': self.rm,
                   'n_s': self.n_s,
                   'odms': self.odms,
                   'mets': self.mets,
                  }
        self.cs = CsOva.to_str(self.cd)

    def get_cs(self):
        return self.cs

    def get_cd(self):
        return self.cd

    def get_fn(self):
        return self.fn

    def get_fnd(self):
        return self.fnd

    @staticmethod
    def to_dict(cs):
        mo = re.match(r'dss (.+) bcs (.+) rm (.+) ' +
                      r'n_s (.+) odms (.+) mets (.+)$', cs)
        vals = [t(mo.group(i+1)) for i, t in enumerate(CsOva.types)]
        ret = dict(zip(CsOva.order, vals))
        ret['bcs_l'] = eval(ret['bcs'])
        return ret

    @staticmethod
    def to_str(cd):
        assert set(cd.keys()) == set(CsOva.order + ['bcs_l'])
        return " ".join([f"{k} {cd[k]}" for k in CsOva.order])

    @staticmethod
    def fn_to_dict(fn):
        b = PurePath(fn).name
        mo = re.match(r'(\S+) (\S+) (\S+) (.*)\.csv', b)
        assert mo.group(1) == CsOva.fn_start
        dict_pre = dict([[k, mo.group(i+2)]
                            for i, k in enumerate(CsOva.fn_pre)])
        dict_c = CsOva.to_dict(mo.group(len(CsOva.fn_pre)+2))
        ret = {**dict_pre, **dict_c}
        ret['bcs_l'] = eval(ret['bcs'])
        return ret

    @staticmethod
    def dict_to_fn(cd):
        assert set(cd.keys()) == set(CsOva.fn_pre + CsOva.order + ['bcs_l'])
        return f"{CsOva.fn_start} {cd['dt']} {cd['cont']} " +\
                " ".join([f"{k} {cd[k]}" for k in CsOva.order]) + ".csv"


class OdMetric:
    def __init__(self, name, eis):
        self.name = name
        self.eis = eis

    def __str__(self):
        return self.name


    def met(self, df, lab_col, os_col, ol_lab):
        raise AttributeError("subclasses of OdMetric need to implement met")


class KPr(OdMetric):
    NAME = "prec"

    def __init__(self, k = 100, name = None, eis = None):
        self.k = int(k)
        if name is None:
            name = f"{self.NAME}@{k}"
        super(KPr, self).__init__(name, eis)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def met(self, df, lab_col, os_col, ol_lab):
        if self.eis is None:
            cdf = df
        else:
            cdf = df.loc[self.eis]
        s_cdf = len(cdf.index)
        if self.k > s_cdf:
            p(f"kpr with k {self.k}, with only {s_cdf} data points")
            c = s_cdf
        else:
            c = self.k
        first_k = cdf.nlargest(c, os_col)
        n_first_ols = (first_k[lab_col] == ol_lab).sum()
        return n_first_ols / len(first_k.index)


class RocAuc(OdMetric):
    NAME = "roc_auc"

    def __init__(self, name = None, eis = None):
        if name is None:
            name = self.NAME
        super(RocAuc, self).__init__(name, eis)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def met(self, df, lab_col, os_col, ol_lab):
        if self.eis is None:
            cdf = df
        else:
            cdf = df.loc[self.eis]
        return roc_auc_score(cdf[lab_col], cdf[os_col])


class Ap(OdMetric):
    NAME = "ap"

    def __init__(self, name = None, eis = None):
        if name is None:
            name = self.NAME
        super(Ap, self).__init__(name, eis)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def met(self, df, lab_col, os_col, ol_lab):
        if self.eis is None:
            cdf = df
        else:
            cdf = df.loc[self.eis]
        return average_precision_score(cdf[lab_col], cdf[os_col],
                pos_label = ol_lab)


class OF1(OdMetric):
    NAME = "of1"

    def __init__(self, name = None, eis = None):
        if name is None:
            name = self.NAME
        super(OF1, self).__init__(name, eis)

    # noinspection PyMethodMayBeStatic
    def met(self, df, lab_col, os_col, ol_lab):
        if self.eis is None:
            cdf = df
        else:
            cdf = df.loc[self.eis]
        return opt_f1(cdf, lab_col, os_col, ol_lab)


class OsAvg(OdMetric):
    NAME = "os_avg"

    def __init__(self, name = None, eis = None):
        if name is None:
            name = self.NAME
        super(OsAvg, self).__init__(name, eis)

    # noinspection PyMethodMayBeStatic
    def met(self, df, lab_col, os_col, ol_lab):
        cdf = df if self.eis is None else df.loc[self.eis]
        return np.average(cdf[os_col])


class OsQtl(OdMetric):
    NAME = "os_qtl"

    def __init__(self, qtl, name = None, eis = None):
        if name is None:
            name = f"{self.NAME}_{qtl}"
        super(OsQtl, self).__init__(name, eis)

        self.qtl = qtl

    # noinspection PyMethodMayBeStatic
    def met(self, df, lab_col, os_col, ol_lab):
        cdf = df if self.eis is None else df.loc[self.eis]
        return np.quantile(cdf[os_col], self.qtl)


class OdModel:
    
    def __init__(self, name):
        self.name = name
        self.converged = True

    def __str__(self):
        return self.name

    def olds(*args, **kwargs):
        raise AttributeError("subclasses of OdModel need to implement olds")
    
    def apply(self, X, smsg = None, emsg = None):
        raise AttributeError("subclasses of OdModel need to implement apply")


class IfModel(OdModel):
    default_kwargs = {
            'n_estimators': 100,
            'max_samples': 'auto',
            'contamination': 'auto',
            'n_jobs': -1,
            }

    def __init__(self, name = "if", **kwargs):
        super(IfModel, self).__init__(name)
        self.kwargs = {**IfModel.default_kwargs, **kwargs}

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'Isolation Forest outlier detection for {X.shape[0]} '\
                    + f'points: starting...'
        if emsg is None:
            emsg = f'    ...Isolation Forest outlier detection completed in '\
                    + '{dur}'
        def task():
            self.model = IsolationForest(**self.kwargs)
            p(f"\nIF parameters: {self.model.get_params()}")

            X_fl = flatten(X) if len(X.shape) > 2 else X
            self.model.fit(X_fl)
            if_score = self.model.score_samples(X_fl)
            return - if_score
        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols


class LofModel(OdModel):
    def __init__(self, name = "lof", **kwargs):
        super(LofModel, self).__init__(name)
        self.kwargs = kwargs

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'LOF outlier detection for {X.shape[0]} '\
                    + f'points: starting...'
        if emsg is None:
            emsg = f'    ...LOF outlier detection completed in ' + '{dur}'
        def task():
            self.model = LocalOutlierFactor(contamination = 'auto',
                    n_jobs = -1, novelty = False, **self.kwargs)
            p(f"\nLOF parameters: {self.model.get_params()}")

            self.model.fit(X)
            return - self.model.negative_outlier_factor_
        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols


class OcsvmModel(OdModel):
    def __init__(self, name = "ocsvm", **kwargs):
        super(OcsvmModel, self).__init__(name)

        if 'gamma' not in kwargs:
            kwargs['gamma'] = 'scale'
        self.kwargs = kwargs

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'OCSVM outlier detection for {X.shape[0]} '\
                    + f'points: starting...'
        if emsg is None:
            emsg = f'    ...OCSVM outlier detection completed in '\
                    + '{dur}'
        def task():
            self.model = OneClassSVM(**self.kwargs)
            p(f"\nOCSVM parameters: {self.model.get_params()}")

            X_fl = flatten(X) if len(X.shape) > 2 else X
            self.model.fit(X_fl)
            p(f"\ngamma = {self.model._gamma}\n")

            return - self.model.score_samples(X_fl)
        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols


class NnbModel(OdModel):
    def __init__(self, k, name = None, **kwargs):
        if name is None:
            name = f"{k}-nnb"
        super(NnbModel, self).__init__(name)

        self.k = k
        self.kwargs = kwargs

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'{self.k}-NNB outlier detection for {X.shape[0]} '\
                    + f'points: starting...'
        if emsg is None:
            emsg = f'    ...{self.k}-NNB outlier detection completed in '\
                    + '{dur}'
        def task():
            self.model = NearestNeighbors(n_neighbors = self.k, n_jobs = -1,
                                          **self.kwargs)
            p(f"\nNNB parameters: {self.model.get_params()}")
            self.model.fit(X)
            # noinspection PyUnresolvedReferences
            return self.model.kneighbors(X)[0][:,self.k-1]
        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols


class Ae(OdModel):
    def __init__(self, ld, epochs, name = None, kreg = None, breg = None,
            areg = None, loss_th = AELT, vb = 1, ep_th = {}, dot_iv = 1000,
            block_iv = 10000):
        if name is None:
            name = f"ae_{ld}"
        super(Ae, self).__init__(name)

        self.ld = ld
        self.epochs = epochs
        self.kreg = kreg
        self.breg = breg
        self.areg = areg
        self.loss_th = loss_th
        self.vb = vb
        self.ep_th = ep_th
        self.dot_iv = dot_iv
        self.block_iv = block_iv

        self.hs = []

    def lls(self):
        return [h.history['loss'][-1] for h in self.hs]

    def cll(self):
        return self.hs[-1].history['loss'][-1]

    def build_model(self):
        raise AttributeError("subclasses of Ae need to implement build_model")

    def callbacks(self):
        return [
                EpochDotBlock(self.dot_iv, self.block_iv),
                NotGonnaMakeIt(self.ep_th),
                ]

    def fit(self, X, vb = None):
        self.converged = True
        X_rs = self.reshape(X)
        v = 0 if od_conf.silent else vb if vb else self.vb
        p(f"fit: {self.epochs} epochs")
        p(f"dot every {self.dot_iv} epochs")
        h = self.ae.fit(
                X_rs, X_rs, epochs=self.epochs, callbacks=self.callbacks(),
                verbose = v)
        self.hs.append(h)
        ll = h.history['loss'][-1]
        ll50 = h.history['loss'][-50:]
        ll50s = "\n".join([str(ll) for ll in ll50])
        
        p(f"\n{'='*80}\nlast loss: {ll}\nloss_th: {self.loss_th}\n{'='*80}")
        if self.loss_th is not None and ll > self.loss_th:
            p(f"WARNING: loss threshold: {self.loss_th}, last loss: {ll}, "\
                    +f"so model didn't converge properly!")
            p("The last losses from history:")
            p(ll50s)
            self.converged = False

        return h

    def predict(self, X):
        sh = X.shape
        X_rs = self.reshape(X)
        X_pred = self.ae.predict(X_rs)
        X_ret = X_pred.reshape(np.r_[-1, sh[1:]])
        return X_ret

    def predict_dec(self, lat_X):
        return self.dec.predict(lat_X)

    def predict_enc(self, X):
        return self.enc.predict(X)

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'{self.name} fitting outlier detection for '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} fitted outlier detection for '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        p("\n---- epochs ----")
        p(self.epochs)

        def task():
            X_rs = self.reshape(X)
            self.build_model()
            self.fit(X_rs)
            if not self.converged:
                ret = None
            else:
                ret = self.apply(X_rs)
            return ret

        _, self.ols = time_it(task, smsg, emsg, not od_conf.silent)
        return self.ols

    def apply(self, X, smsg = None, emsg = None):
        if not self.converged:
            raise ValueException("called apply() on a non-converged model")
        
        if smsg is None:
            smsg = f'{self.name} applies outlier detection to '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} applied outlier detection to '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            X_rs = self.reshape(X)
            res = self.predict(X_rs)
            ret = bo_norms(X_rs-res)
            return ret

        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols


class EpochDotBlock(Callback):
    def __init__(self, every = 1000, block = 10000):
        self.every = every
        self.block = block

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.every == 0:
            p(".", end = '', flush = True)
        if (epoch+1) % self.block == 0:
            p(f"\n{ep_str(epoch+1)} loss: {logs.get('loss'):<30}")


class NotGonnaMakeIt(Callback):
    def __init__(self, ep_th, monitor = "loss"):
        self.ep_th = ep_th
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.ep_th:
            p("")
            lim = self.ep_th[epoch]
            lv = logs.get(self.monitor)
            info = f"epoch: {epoch}, limit: {lim}, loss: {lv}"
            if lv > lim:
                self.model.stop_training = True
                msg = f"ERROR: ain't gonna make it: {info}"
                log(msg)
            else:
                p(f"{info}: fine!")


def exp_decay(s, t, ef):
    p("creating lr-scheduler exp_decay()")
    def scheduler(epoch):
        if epoch < t:
            return s
        else:
            return s * np.exp(ef * (t - epoch))

    return scheduler


class SimpleConvAe(Ae):
    def __init__(self, ld, epochs, name = None, kreg = None, breg = None,
            areg = None, loss_th = AELT, vb = 1, ep_th = {}, dot_iv = 1000,
            block_iv = 10000):
        if name is None:
            if kreg is None:
                rs = "0"
            else:
                rs = f"{kreg.l2:.1e}"
            name = f"scae_{ld}_r{rs}"
        super(SimpleConvAe, self).__init__(
                ld, epochs, name, kreg, breg, areg, loss_th, vb, ep_th, dot_iv,
                block_iv)

    @classmethod
    def reshape(self, X):
        if X.shape[1:] == (28, 28, 1):
            return X
        else:
            return X.reshape(-1, 28, 28, 1)

    def build_model(self):
        enc_in = Input(shape=(28, 28, 1))
        x = Conv2D(16, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(enc_in)
        x = Conv2D(32, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = MaxPooling2D(3)(x)
        x = Conv2D(32, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(16, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = GlobalMaxPooling2D()(x)
        enc_out = Dense(self.ld, activation = "relu",
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        enc = Model(enc_in, enc_out)
        #enc.summary(print_fn = p)

        dec_in = Input(shape=(self.ld,))
        x = Dense(16, activation = 'relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(dec_in)
        x = Reshape((4, 4, 1))(x)
        x = Conv2DTranspose(16, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(32, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = UpSampling2D(3)(x)
        x = Conv2DTranspose(16, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        dec_out = Conv2DTranspose(1, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        dec = Model(dec_in, dec_out)
        #dec.summary(print_fn = p)

        ae_in = Input(shape=(28, 28, 1))
        latent = enc(ae_in)
        recon = dec(latent)
        self.ae = Model(ae_in, recon)

        #self.ae.summary(print_fn = p)
        rconf = f"---- regularizaton conf ----\n"\
                +f"k: {self.kreg.l1 if self.kreg else None} "\
                +f"{self.kreg.l2 if self.kreg else None}\n"\
                +f"b: {self.breg.l1 if self.breg else None} "\
                +f"{self.breg.l2 if self.breg else None}\n"\
                +f"a: {self.areg.l1 if self.areg else None} "\
                +f"{self.areg.l2 if self.areg else None}\n"
        p(rconf)

        self.ae.compile(optimizer = 'adam',
                #run_eagerly = True,
                loss  = 'mean_squared_error')


class ConvAe(Ae):
    """ This model is inspired by the convolutional AE proposed in
    https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers
    """
    def __init__(self, ld, epochs, name = None, kreg = None, breg = None,
            areg = None, loss_th = AELT, vb = 1, ep_th = {}, dot_iv = 1000,
            block_iv = 10000):
        if name is None:
            if kreg is None:
                rs = "0"
            else:
                rs = f"{kreg.l2:.1e}"
            name = f"cae_{ld}_r{rs}"
        super(ConvAe, self).__init__(
                ld, epochs, name, kreg, breg, areg, loss_th, vb, ep_th, dot_iv,
                block_iv)

    @classmethod
    def reshape(self, X):
        if X.shape[1:] == (28, 28, 1):
            return X
        else:
            return X.reshape(-1, 28, 28, 1)

    def build_model(self):
        enc_in = Input(shape=(28, 28, 1))
        x = Conv2D(32, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(enc_in)
        x = Conv2D(64, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(128, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(256, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = GlobalMaxPooling2D()(x)
        enc_out = Dense(self.ld, activation = "relu",
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        enc = Model(enc_in, enc_out)
        #enc.summary(print_fn = p)

        dec_in = Input(shape=(self.ld,))
        x = Dense(144, activation = 'relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(dec_in)
        x = Reshape((12, 12, 1))(x)
        x = Conv2DTranspose(128, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(64, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(32, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        dec_out = Conv2DTranspose(1, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        dec = Model(dec_in, dec_out)
        #dec.summary(print_fn = p)

        ae_in = Input(shape=(28, 28, 1))
        latent = enc(ae_in)
        recon = dec(latent)
        self.ae = Model(ae_in, recon)

        #self.ae.summary(print_fn = p)
        rconf = f"---- regularizaton conf ----\n"\
                +f"k: {self.kreg.l1 if self.kreg else None} "\
                +f"{self.kreg.l2 if self.kreg else None}\n"\
                +f"b: {self.breg.l1 if self.breg else None} "\
                +f"{self.breg.l2 if self.breg else None}\n"\
                +f"a: {self.areg.l1 if self.areg else None} "\
                +f"{self.areg.l2 if self.areg else None}\n"
        p(rconf)

        self.ae.compile(optimizer = 'adam',
                #run_eagerly = True,
                loss  = 'mean_squared_error')


class ConvAe2(Ae):
    """ This model is inspired by the convolutional AE proposed in
    https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers
    """
    def __init__(self, ld, epochs, name = None, kreg = None, breg = None,
            areg = None, loss_th = AELT, vb = 1, ep_th = {}, dot_iv = 1000,
            block_iv = 10000):
        if name is None:
            if kreg is None:
                rs = "0"
            else:
                rs = f"{kreg.l2:g}"
            name = f"cae2_{ld}_r{rs}_{ep_str(epochs)}"
        super(ConvAe2, self).__init__(
                ld, epochs, name, kreg, breg, areg, loss_th, vb, ep_th, dot_iv,
                block_iv)

    @classmethod
    def reshape(self, X):
        if X.shape[1:] == (28, 28, 1):
            return X
        else:
            return X.reshape(-1, 28, 28, 1)

    def build_model(self):
        enc_in = Input(shape=(28, 28, 1))
        x = Conv2D(32, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(enc_in)
        x = Conv2D(64, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(128, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(256, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = GlobalMaxPooling2D()(x)
        enc_out = Dense(self.ld, activation = "relu",
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        self.enc = Model(enc_in, enc_out)
        #self.enc.summary(print_fn = p)

        dec_in = Input(shape=(self.ld,))
        x = Dense(256, activation = 'relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(dec_in)
        x = Reshape((1, 1, 256))(x)
        x = UpSampling2D(12)(x)
        x = Conv2DTranspose(128, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(64, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(32, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        dec_out = Conv2DTranspose(1, 5, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        self.dec = Model(dec_in, dec_out)
        #self.dec.summary(print_fn = p)

        ae_in = Input(shape=(28, 28, 1))
        latent = self.enc(ae_in)
        recon = self.dec(latent)
        self.ae = Model(ae_in, recon)

        #self.ae.summary(print_fn = p)
        rconf = f"---- regularizaton conf ----\n"\
                +f"k: {self.kreg.l1 if self.kreg else None} "\
                +f"{self.kreg.l2 if self.kreg else None}\n"\
                +f"b: {self.breg.l1 if self.breg else None} "\
                +f"{self.breg.l2 if self.breg else None}\n"\
                +f"a: {self.areg.l1 if self.areg else None} "\
                +f"{self.areg.l2 if self.areg else None}\n"
        p(rconf)

        self.ae.compile(optimizer = 'adam',
                #run_eagerly = True,
                loss  = 'mean_squared_error')


class ConvAe3(Ae):
    def __init__(self, ld, epochs, name = None, kreg = None, breg = None,
            areg = None, loss_th = AELT, vb = 1, ep_th = {}, dot_iv = 1000,
            block_iv = 10000):
        if name is None:
            if kreg is None:
                rs = "0"
            else:
                rs = f"{kreg.l2:g}"
            name = f"cae3_{ld}_r{rs}_{ep_str(epochs)}"
        super(ConvAe3, self).__init__(
                ld, epochs, name, kreg, breg, areg, loss_th, vb, ep_th, dot_iv,
                block_iv)

    @classmethod
    def reshape(self, X):
        if X.shape[1:] == (28, 28, 1):
            return X
        else:
            return X.reshape(-1, 28, 28, 1)

    def build_model(self):
        enc_in = Input(shape=(28, 28, 1))
        x = enc_in
        x = Conv2D(16, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(32, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(64, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(128, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(256, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = MaxPooling2D(2)(x)
        x = Conv2D(256, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(128, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(64, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2D(32, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = GlobalMaxPooling2D()(x)
        enc_out = Dense(self.ld, activation = "relu",
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
                
        self.enc = Model(enc_in, enc_out)
        self.enc.summary(print_fn = p)
        
        dec_in = Input(shape=(self.ld,))
        x = dec_in
        x = Dense(32, activation = 'relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Reshape((1, 1, 32))(x)
        x = Conv2DTranspose(64, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(128, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(256, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(256, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = UpSampling2D(2)(x)
        x = Conv2DTranspose(128, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(64, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(32, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        x = Conv2DTranspose(16, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)
        dec_out = Conv2DTranspose(1, 3, activation='relu',
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg)(x)

        self.dec = Model(dec_in, dec_out)
        self.dec.summary(print_fn = p)

        ae_in = Input(shape=(28, 28, 1))
        latent = self.enc(ae_in)
        recon = self.dec(latent)
        self.ae = Model(ae_in, recon)

        self.ae.summary(print_fn = p)
        rconf = f"---- regularizaton conf ----\n"\
                +f"k: {self.kreg.l1 if self.kreg else None} "\
                +f"{self.kreg.l2 if self.kreg else None}\n"\
                +f"b: {self.breg.l1 if self.breg else None} "\
                +f"{self.breg.l2 if self.breg else None}\n"\
                +f"a: {self.areg.l1 if self.areg else None} "\
                +f"{self.areg.l2 if self.areg else None}\n"
        p(rconf)

        self.ae.compile(optimizer = 'adam',
                #run_eagerly = True,
                loss  = 'mean_squared_error')


class FlatAeV(Ae):
    def __init__(self, epochs, dd, enc_l_n_units, dec_l_n_units,
            act = 'elu', use_bias = False, bn = False, name = None,
            kreg = None, breg = None, areg = None, loss_th = AELT, vb = 1,
            ep_th = {}, dot_iv = 1000, block_iv = 10000):
        assert dd == dec_l_n_units[-1]
        self.dd = dd
        self.ld = enc_l_n_units[-1]
        self.enc_l_n_units = enc_l_n_units
        self.dec_l_n_units = dec_l_n_units
        self.act = act
        self.use_bias = use_bias
        self.bn = bn
        self.enc = None
        self.dec = None
        if name is None:
            if kreg is None:
                rs = "0"
            else:
                rs = f"{kreg.l2:.1e}"
            nls = len(self.enc_l_n_units) + len(self.dec_l_n_units)
            name = f"faev_{self.dd}_{self.ld}_{nls}_r{rs}_{ep_str(epochs)}"
        super(FlatAeV, self).__init__(
                1, epochs, name, kreg, breg, areg, loss_th, vb, ep_th, dot_iv,
                block_iv)

    @classmethod
    def reshape(self, X):
        return X

    def create_enc(self):
        enc_in = Input(shape=(self.dd,))
        lyr_pairs = [[
            self.bn_lyr(),
            Dense(ls, activation=self.act, use_bias=self.use_bias,
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg,
                kernel_initializer = self.ki),
            ] for ls in self.enc_l_n_units]

        enc_lyrs = reduce(lambda a,i: a+i, lyr_pairs)
        enc_out = pipe(enc_lyrs)(enc_in)
        self.enc = Model(enc_in, enc_out)

    def create_dec(self):
        dec_in = Input(shape=(self.ld,))
        lyr_pairs = [[
            self.bn_lyr(),
            Dense(ls, activation=self.act, use_bias=self.use_bias,
                kernel_regularizer = self.kreg,
                bias_regularizer = self.breg,
                activity_regularizer = self.areg,
                kernel_initializer = self.ki),
            ] for ls in self.dec_l_n_units]

        dec_lyrs = reduce(lambda a,i: a+i, lyr_pairs)
        dec_out = pipe(dec_lyrs)(dec_in)
        self.dec = Model(dec_in, dec_out)
        #p("\ndecoder:")
        #self.dec.summary(print_fn = p)

    def build_model(self):
        if self.bn:
            self.ki = "he_normal"
        else:
            self.ki = "glorot_uniform"

        if self.bn:
            self.bn_lyr = BatchNormalization
        else:
            self.bn_lyr = lambda: lambda x: x

        self.create_enc()
        self.create_dec()
        ae_in = Input(shape=(self.dd,))
        latent = self.enc(ae_in)
        recon = self.dec(latent)
        self.ae = Model(ae_in, recon)
        #p("\nfull Ae:")
        #self.ae.summary(print_fn = p)

        rconf = f"---- regularizaton conf ----\n"\
                +f"k: {self.kreg.l1 if self.kreg else None} "\
                +f"{self.kreg.l2 if self.kreg else None}\n"\
                +f"b: {self.breg.l1 if self.breg else None} "\
                +f"{self.breg.l2 if self.breg else None}\n"\
                +f"a: {self.areg.l1 if self.areg else None} "\
                +f"{self.areg.l2 if self.areg else None}\n"
        p(rconf)

        self.ae.compile(optimizer = 'adam',
                #run_eagerly = True,
                loss  = 'mean_squared_error')


class Potatoes(OdModel):
    def __init__(self, aec, ae_kwargs, k, name = None, mr = 10, rp = 3,
            check_close_pairs = True):
        self.aes = [aec(**ae_kwargs) for _ in range(k)]
        self.ae_kwargs = ae_kwargs
        self.k = k
        self.mr = mr
        self.rp = rp
        self.check_close_pairs = check_close_pairs

        if name is None:
            name = f"pot_{self.aes[0]}_{k}"
        super(Potatoes, self).__init__(name)

    def __getattr__(self, attr):
        if attr in self.ae_kwargs:
            return self.ae_kwargs[attr]
        else:
            raise AttributeError(f"the Potatoes instance {self} does not "\
                    +f"have an attribute {attr}")

    def i_part(self, X):
        X_ids = np.arange(X.shape[0])
        np.random.shuffle(X_ids)
        X_ids_ps = np.array_split(X_ids, self.k)
        p(f"shapes of the partitions: {[p.shape for p in X_ids_ps]}")
        return X_ids_ps

    def fit(self, X):
        return self.rfit(X)

    def rfit(self, X):
        self.converged = True

        def retry_fit(model, X, encore):
            p(f"in retry_fit: encore: {encore}")
            if encore == 0:
                return 0
            model.build_model()
            model.fit(X)
            if model.converged:
                return encore
            else:
                return retry_fit(model, X, encore-1)

        def retry_part(ancora):
            p(f"in retry_part: ancora: {ancora}")
            if ancora == 0:
                self.converged = False
                return 0
            self.ips = self.i_part(X)
            for mod, ip in zip(self.aes, self.ips):
                left = retry_fit(mod, X[ip], self.mr)
                if left == 0:
                    break

            if left == 0:
                return retry_part(ancora-1)
            else:
                return ancora

        retry_part(self.rp)

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'{self.name} fitting outlier detection for '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} fitted outlier detection for '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            p(f"\n*** in Potatoes.task() ***\nself.k = {self.k}")
            p(f"self.aes: class: {self.aes[0].__class__.__name__}")
            p(f"self.aes ld and epochs: {[(ae.ld, ae.epochs) for ae in self.aes]}")

            self.rfit(X)
            if self.converged:
                ret = self.apply(X)
            else:
                ret = None
            return ret

        _, self.ols = time_it(task, smsg, emsg, not od_conf.silent)
        return self.ols

    def apply(self, X, smsg = None, emsg = None):
        if not self.converged:
            raise ValueException("calling apply() on a non-converged model")

        if smsg is None:
            smsg = f'{self.name} applies outlier detection to '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} applied outlier detection to '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            l_ols = [bo_norms(X - ae.predict(X)) for ae in self.aes]
            if self.check_close_pairs:
                start = datetime.now()
                cps = close_pairs(l_ols)
                end = datetime.now()
                p("\n\n\n#################\n#################\n")
                p(f"checking close pairs lasted: {end-start}")
                p("\n#################\n#################\n\n\n")
                if cps:
                    raise Exception("\n\n==========\nWarning\n===========\n"\
                            +f"We have duplicate ols arrays!\n{cps}\n{l_ols}")

            return np.amax(l_ols, axis = 0)

        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols

    def predict_l(self, X):
        return [ae.predict(X) for ae in self.aes]

    def predict(self, X):
        return np.stack(self.predict_l(X), axis = 1)

    def predict_enc_l(self, X):
        return [ae.predict_enc(X) for ae in self.aes]

    def predict_enc(self, X):
        return np.stack(self.predict_enc_l(X), axis = 1)

    def predict_dec_l(self, lat_X):
        return [ae.predict_dec(lat_X) for ae in self.aes]

    def predict_dec(self, X):
        return np.stack(self.predict_dec_l(X), axis = 1)


def ep_str(eps):
    if eps < 1000:
        es = f"e{eps}"
    elif eps % 1000 == 0:
        es = f"e{eps//1000}t"
    else:
        es = f"e{round(eps/1000, 1)}t"
    return es


class OdEns(OdModel):
    def __init__(self, l_mod_cls, l_mc_kwargs, efun, mr, name,
            check_close_pairs = True):
        self.l_mod_cls = l_mod_cls
        self.l_mc_kwargs = l_mc_kwargs
        self.efun = efun
        self.mr = mr
        self.check_close_pairs = check_close_pairs
        super(OdEns, self).__init__(name)

    def rfit(self, X):
        def retry_fit(mod_cl, mc_kwargs, encore):
            p(f"in retry_fit: encore: {encore}")
            if encore == 0:
                return 0, None
            model = mod_cl(**mc_kwargs)
            model.fit(X)
            if model.converged:
                return encore, model
            else:
                return retry_fit(mod_cl, mc_kwargs, X, encore-1)

        ens = []
        for mod_cl, mc_kwargs in zip(self.l_mod_cls, self.l_mc_kwargs):
            left, m = retry_fit(mod_cl, mc_kwargs, self.mr)
            if left == 0:
                self.converged = False
                ens = None
                break
            else:
                self.converged = True
                ens.append(m)

        return ens

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'{self.name} fitting outlier detection for '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} fitted outlier detection for '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            self.ens = self.rfit(X)
            if self.converged:
                ret = self.apply(X)
            else:
                ret = None
            return ret

        _, self.ols = time_it(task, smsg, emsg, not od_conf.silent)
        return self.ols
    
    def apply(self, X, smsg = None, emsg = None):
        if not self.converged:
            raise ValueException("called apply() on a non-converged ensemble")

        if smsg is None:
            smsg = f'{self.name} applies outlier detection to '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} applied outlier detection to '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            l_ols = [m.apply(X) for m in self.ens]
            ret = self.efun(self.ens, l_ols)
            p("efun values for the last 50 data points:")
            np.set_printoptions(threshold = 50)
            p(ret[-50:])

            if self.check_close_pairs:
                start = datetime.now()
                cps = close_pairs(l_ols)
                end = datetime.now()
                p("\n\n\n#################\n#################\n")
                p(f"checking close pairs lasted: {end-start}")
                p("\n#################\n#################\n\n\n")
                if cps:
                    raise Exception("\n\n==========\nWarning\n===========\n"\
                            +f"We have duplicate ols arrays!\n{cps}\n{l_ols}")

            return ret

        _, ols = time_it(task, smsg, emsg, not od_conf.silent)
        return ols


class OdEnsF(OdModel):
    def __init__(self, l_mod_cls, l_mc_kwargs, efun, mr, name, X_t,
            check_close_pairs = True):
        self.l_mod_cls = l_mod_cls
        self.l_mc_kwargs = l_mc_kwargs
        self.efun = efun
        self.mr = mr
        self.X_t = X_t
        super(OdEnsF, self).__init__(name)
        self.check_close_pairs = self.rfit(X_t)

        self.nfm = OdEns(self.l_mod_cls, self.l_mc_kwargs, self.efun, self.mr,
            self.name, self.check_close_pairs)
        self.ens = self.nfm.rfit(X_t)
        if self.nfm.converged:
            self.converged = True
        else:
            self.converged = False
            raise Exception("didn't converge")

    def olds(self, X, smsg = None, emsg = None):
        if not self.converged:
            raise Exception("calling olds on a non converged model")

        if smsg is None:
            smsg = f'{self.name} fitting outlier detection for '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} fitted outlier detection for '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            l_ols = [m.olds(X) for m in self.ens]
            if self.check_close_pairs:
                start = datetime.now()
                cps = close_pairs(l_ols)
                end = datetime.now()
                p("\n\n\n#################\n#################\n")
                p(f"checking close pairs lasted: {end-start}")
                p("\n#################\n#################\n\n\n")
                if cps:
                    raise Exception("\n\n==========\nWarning\n===========\n"\
                            +f"We have duplicate ols arrays!\n{cps}\n{l_ols}")
            ret = self.efun(self.ens, l_ols)
            p("efun values for the last 50 data points:")
            np.set_printoptions(threshold = 50)
            p(ret[-50:])

            return ret

        _, self.ols = time_it(task, smsg, emsg, not od_conf.silent)
        return self.ols


class Aee(OdModel):
    def __init__(self, aec, ae_kwargs, k, name = None, mr = 10,
            efun = lambda l_ols: np.amin(l_ols, axis=0)):
        self.aes = [aec(**ae_kwargs) for _ in range(k)]
        self.ae_kwargs = ae_kwargs
        self.k = k
        self.mr = mr
        self.efun = efun

        if name is None:
            name = f"aee_{efun.__name__}_{self.aes[0]}_{k}"
        super(Aee, self).__init__(name)

    def __getattr__(self, attr):
        if attr in self.ae_kwargs:
            return self.ae_kwargs[attr]
        else:
            raise AttributeError(f"the Aee instance {self} does not "\
                    +f"have an attribute {attr}")

    def rfit(self, X):
        self.converged = True

        def retry_fit(model, X, encore):
            p(f"in retry_fit: encore: {encore}")
            if encore == 0:
                return 0
            model.build_model()
            model.fit(X)
            if model.converged:
                return encore
            else:
                return retry_fit(model, X, encore-1)

        for mod in self.aes:
            left = retry_fit(mod, X, self.mr)
            if left == 0:
                self.converged = False
                break

        self.hs = self.aes[0].hs

    def olds(self, X, smsg = None, emsg = None):
        if smsg is None:
            smsg = f'{self.name} fitting outlier detection for '\
                    + f'{X.shape[0]} points: starting...'
        if emsg is None:
            emsg = f'    ...{self.name} fitted outlier detection for '\
                    + f'{X.shape[0]} points: completed in ' + '{dur}'

        def task():
            p(f"\n*** in Aee.task() ***\nself.k = {self.k}")
            p(f"self.aes: class: {type(self.aes[0])}")
            p(f"self.aes ld and epochs: {[(ae.ld, ae.epochs) for ae in self.aes]}")
            self.rfit(X)

            if self.converged:
                l_ols = [bo_norms(X - ae.predict(X)) for ae in self.aes]
                start = datetime.now()
                cps = close_pairs(l_ols)
                end = datetime.now()
                p("\n\n\n#################\n#################\n")
                p(f"checking close pairs lasted: {end-start}")
                p("\n#################\n#################\n\n\n")
                if cps:
                    raise Exception("\n\n==========\nWarning\n===========\n"\
                            +f"We have duplicate ols arrays!\n{cps}\n{l_ols}")
                np.set_printoptions(threshold = sys.maxsize)
                ret = self.efun(l_ols)
                p("efun values for the last 50 data points:")
                p(ret[-50:])
            else:
                ret = None

            return ret

        _, self.ols = time_it(task, smsg, emsg, not od_conf.silent)
        return self.ols

    def predict_l(self, X):
        return [ae.predict(X) for ae in self.aes]

    def predict(self, X):
        return np.stack(self.predict_l(X), axis = 1)

    def predict_enc_l(self, X):
        return [ae.predict_enc(X) for ae in self.aes]

    def predict_enc(self, X):
        return np.stack(self.predict_enc_l(X), axis = 1)

    def predict_dec_l(self, lat_X):
        return [ae.predict_dec(lat_X) for ae in self.aes]

    def predict_dec(self, X):
        return np.stack(self.predict_dec_l(X), axis = 1)


class Aee2(OdEnsF):
    def __init__(self, aec, ae_kwargs, k, X_t, name = None, mr = 10,
            efun = lambda l_ols: np.amin(l_ols, axis=0),
            check_close_pairs = True):
        if name is None:
            name = f"aee_{efun.__name__}_{aec.__name__}_{k}"
        super(Aee2, self).__init__([aec]*k, [ae_kwargs]*k, efun, mr, name, X_t,
            check_close_pairs)

    def __getattr__(self, attr):
        if attr in self.ae_kwargs:
            return self.ae_kwargs[attr]
        else:
            raise AttributeError(f"the Aee2 instance {self} does not "\
                    +f"have an attribute {attr}")

    def predict_l(self, X):
        return [ae.predict(X) for ae in self.aes]

    def predict(self, X):
        return np.stack(self.predict_l(X), axis = 1)

    def predict_enc_l(self, X):
        return [ae.predict_enc(X) for ae in self.aes]

    def predict_enc(self, X):
        return np.stack(self.predict_enc_l(X), axis = 1)

    def predict_dec_l(self, lat_X):
        return [ae.predict_dec(lat_X) for ae in self.aes]

    def predict_dec(self, X):
        return np.stack(self.predict_dec_l(X), axis = 1)


class PotEns(OdEns):
    def __init__(self, aec, ae_kwargs, k, s_ens, efun, name = None, pmr = 10,
            prp = 3, mr = 1, check_close_pairs = True):
        if name is None:
            name = f"pe_{efun.__name__}_{aec.__name__}_{k}_{s_ens}"
        pot_kwargs = {'aec': aec, 'ae_kwargs': ae_kwargs, 'k': k,
                'name': None, 'mr': pmr, 'rp': prp}
        super(PotEns, self).__init__([Potatoes]*s_ens, [pot_kwargs]*s_ens,
                efun, mr, name, check_close_pairs)


def roc_of1(odm, X, ls, ol_lab, rf = RF):
    mets = [RocAuc(), OF1()]
    metvals, ode =  Ode.eval(odm, mets, X, ls, ol_lab, rf)
    return metvals


def roc_of1_ova(odm, ld, bc, rm, n_s, rf = RF):
    mets = [OF1(), RocAuc()]
    resdf, ode =  Ode.eval_ova(odm, mets, ld, bc, rm, n_s, rf)
    return resdf


def iforest_problem():
    lc = 100
    td_1 = np.random.random((lc, 2))
    sc = 10
    td_2 = np.random.random((sc, 2)) + np.array([5, 0.])
    td = np.r_[td_1, td_2]

    ed = np.array([[0.3, 5],
                   [5.5, 0.5],
                   [0.5, 0.5]
                  ])
    el = ['malign', 'unclear', 'benign']

    ifor = IsolationForest(contamination='auto')
    ifor.fit(td)
    ifs = ifor.score_samples(ed)

    sdf = pd.DataFrame({'x': ed[:,0], 'y': ed[:,1], 'label': el, 'score': ifs})

    ad = np.r_[td, ed]
    types = np.array(['train']*td.shape[0] + ['eval']*ed.shape[0])
    pdf = pd.DataFrame({'x': ad[:,0], 'y': ad[:,1], 'type': types})
    g = sns.scatterplot(data = pdf, x = 'x', y = 'y', hue = 'type')
    g.set_title(f"scores of the evaluation points\n(the lower the score, "\
            +f"the more abnormal)\n" + str(sdf))
    plt.gcf().suptitle(f"Isolation Forest Problem\nlarge trainings cluster "\
            +f"size: {lc},\nsmall trainings cluster size: {sc}\n",
            fontsize = 17)
    plt.subplots_adjust(top = .5)
    plt.show()


def if_mnist(bc, rm, n_s):
    if_kwargs = {'n_estimators': 100, 'max_samples': 'auto'} # the default
    odm = IfModel(**if_kwargs)
    ld = Mnist('if mnist')

    return roc_of1_ova(odm, ld, bc, rm, n_s)


def check_if_mnist():
    df = if_mnist(0, .01, 20)
    print(df)
    print(df.describe())


def lof_mnist(bc, rm, n_s):
    lof_kwargs = {'n_neighbors': 10, 'n_jobs': -1}
    
    odm = LofModel(**lof_kwargs)
    ld = Mnist('lof mnist')

    return roc_of1_ova(odm, ld, bc, rm, n_s)


def check_lof_mnist():
    df = lof_mnist(0, .01, 20)
    print(df)
    print(df.describe())


def ocsvm_mnist(bc, rm, n_s):
    kwargs = {'gamma': 'scale', 'nu': 0.01}
    
    odm = OcsvmModel(**kwargs)
    ld = Mnist('ocsvm mnist')

    return roc_of1_ova(odm, ld, bc, rm, n_s)


def check_ocsvm_mnist():
    df = ocsvm_mnist(0, .01, 20)
    print(df)
    print(df.describe())


def ocsvm_mnist_tune():
    bc = 0
    rm = 0.01
    rep = 30
    nu = .14
    l_kwargs = [{'gamma': g} for g in np.linspace(1e-7, 1e-6, 20)]

    d2str = lambda d: " ".join([f"{k}: {v:e}" for k, v in d.items()])
    tbls = [roc_of1_ova(OcsvmModel(nu = nu, **kwargs),
                        Mnist(f"ocsvm {d2str(kwargs)}"),
                        bc, rm, rep) for kwargs in l_kwargs]
    roc_means, of1_means = zip(*[[t[RocAuc.name].mean(), t[OF1.name].mean()] \
                                 for t in tbls])
    var_n = 'metric'
    val_n = 'metric value'
    kwa_n = 'kwargs'
    l_s_kwargs = [d2str(d) for d in l_kwargs]

    df = pd.DataFrame({kwa_n: l_s_kwargs,
                       RocAuc.name: roc_means,
                       OF1.name: of1_means})
    print("means for each kwargs combination:")
    print(df)

    tdf = pd.concat(tbls)
    kwa_col = np.array(l_s_kwargs).repeat(rep)
    tdfe = tdf.assign(**{kwa_n: kwa_col})
    tdfm = tdfe.melt(id_vars = [kwa_n], value_vars = [RocAuc.name, OF1.name],
                       value_name = val_n, var_name = var_n)
    fig, ax = plt.subplots(figsize = (5, 7))
    g = sns.lineplot(data = tdfm, x = kwa_n, y = val_n, hue = var_n, ax = ax,
                     sort = False)
    fig.suptitle(f"{str(df)}\n------------------", fontsize = 7)
    g.set_title(f"OCSVM results for various kwargs\n"\
                +f"mnist ova bc {bc} rm {rm}")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom = .25, top = .57)
    plt.show()


def nnb_mnist(bc, rm, n_s):
    k = 10
    odm = NnbModel(k = k)
    ld = Mnist(f'{k}th-nnb mnist')

    return roc_of1_ova(odm, ld, bc, rm, n_s)


def check_nnb_mnist():
    df = nnb_mnist(0, .01, 20)
    print(df)
    print(df.describe())


def scae_mnist(bc, rm, n_s, ld, epochs):
    odm = SimpleConvAe(ld, epochs)
    lab_data = Mnist(f'SimpleConvAe_{ld} mnist', flatten = False,
                     add_channel = True)

    ret = roc_of1_ova(odm, lab_data, bc, rm, n_s)
    return ret


def check_scae_mnist():
    df = scae_mnist(0, .01, 5, 64, 500)
    df = scae_mnist(0, .01, 5, 64, 500)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df)
    print(df.describe())


def cae_mnist(bc, rm, n_s, ld, epochs):
    odm = ConvAe(ld, epochs)
    lab_data = Mnist(f'ConvAe_{ld} mnist', flatten = False,
                     add_channel = True)

    return roc_of1_ova(odm, lab_data, bc, rm, n_s)


def check_cae_mnist():
    df = cae_mnist(0, .01, 20, 32, 1500)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df)
    print(df.describe())


def cae2_mnist(bc, rm, n_s, ld, epochs):
    odm = ConvAe2(ld, epochs)
    lab_data = Mnist(f'ConvAe2_{ld} mnist', flatten = False,
                     add_channel = True)

    return roc_of1_ova(odm, lab_data, bc, rm, n_s)


def check_cae2_mnist():
    df = cae2_mnist(0, .01, 20, 32, 1500)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df)
    print(df.describe())


def pot_mnist(bc, rm, n_s, ld, epochs, aec = ConvAe2, k = 5):
    aec_kwargs = {'ld': ld, 'epochs': epochs}
    odm = Potatoes(aec, aec_kwargs, k = k)
    lab_data = Mnist(f"potatoes_{ld}_{k} mnist")
    return roc_of1_ova(odm, lab_data, bc, rm, n_s)


def check_pot_mnist():
    df = pot_mnist(0, .01, 5, 32, 500)
    print(df)
    print(df.describe())


def do_ova_eval():
    dt = datetime.now().strftime(DATE_TIME_FORMAT)

    tc = None
    tc = 200
    ec = None
    ec = 0
    npds = 5
    dss = [Mnist(t_count = tc, e_count = ec),
           FMnist(t_count = tc, e_count = ec),
           Cifar10(t_count = tc, e_count = ec)]
    dsss = " ".join([str(ds) for ds in dss])
    lds = np.array(dss).repeat(npds)
    bcs = np.tile(np.arange(npds), len(dss))
    rms = [0.01]*bcs.shape[0]
    l_n_s = [20]*bcs.shape[0]
    l_n_s = [2]*bcs.shape[0]

    odms = [IfModel(), LofModel(), OcsvmModel(nu = .15, gamma = 1.131579e-06),
            NnbModel(k = 10), NnbModel(k = 20)]
    odmss = " ".join([str(m) for m in odms])
    mets = [OF1(), RocAuc(), KPr(k=50), KPr(k=100)]
    metss = " ".join([str(m) for m in mets])

    conf = f"{dsss} bcs {npds} rm {rms[0]} n_s {l_n_s[0]} "\
            +f"{odmss} {metss} {dt}"

    print("\n---- conf ----")
    print(conf)

    fdf, ddf, f = Ode.ova_fig(lds, bcs, rms, l_n_s, odms, mets,
            title = f"ova evaluation {conf}", kind = "box", col_wrap = npds,
            height = 3, aspect = 1)
    print("\n---- ddf ----")
    print(ddf)

    fn = "ova eval {} {} " + conf + ".{}"
    fn_fdf = p_join(GENDIR, fn.format(dt, "fulldf", "csv"))
    fdf.to_csv(fn_fdf, index = False)
    print(f"wrote file {fn_fdf}")
    fn_msdf = p_join(GENDIR, fn.format(dt, "ddf", "csv"))
    ddf.to_csv(fn_msdf, index = False)
    print(f"wrote file {fn_msdf}")
    
    plt.tight_layout()
    plt.subplots_adjust(top = .92, left = .04, right = .9)
    plt.show()
    fn_plot = p_join(GENDIR, fn.format(dt, "plot", "pdf"))
    f.savefig(fn_plot)
    print(f"saved plot to {fn_plot}")


###############################################################################
if __name__ == '__main__':
    print(Ode.pivot_bdgid_sm_f(
        "generatedData/datasets/mnist_fmnist_bc_0_1.csv", 'ap'))
