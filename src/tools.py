#!/usr/bin/env python

import numpy as np
from scipy.spatial import distance_matrix
from functools import reduce
from matplotlib import pyplot as plt
plt.style.use("ggplot")

from datetime import datetime


def bounding_box(X):
    axes = tuple(range(len(X.shape)))
    return np.c_[np.min(X, axes), np.max(X, axes)]


def hist_of_dm(X, plt_axis = None, hbins = 300):
    if plt_axis is None:
        _, plt_axis = plt.subplots(1, 1)

    dm = distance_matrix(X, X)
    plt_axis.hist(dm.flatten(), bins = hbins, color = "steelblue")


def check_hist_of_dm():
    d = 100
    count = 10**3
    ol_count = 1
    X = np.random.random((count, d))
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(3, 1, sharex = True)
    fig.suptitle(f"demo of hist_of_dm: hist of distance matrix of {count} "\
            +f"points in {d}D unit cube,\nbelow \"outliers\" ol are actually"\
            +f" IN-liers (zoom in between 7 and 8!)")

    outliers1 = np.full((ol_count, d), .5)
    outliers1[:, 0] = 2
    X_plus_outliers1 = np.r_[X, outliers1]
    hist_of_dm(X_plus_outliers1, axes[0])
    axes[0].set_title("ol = [2, 0.5, ..., 0.5]", fontsize = 10)

    outliers2 = np.full((ol_count, d), .5)
    outliers2[:, 0] = 1.0
    X_plus_outliers2 = np.r_[X, outliers2]
    hist_of_dm(X_plus_outliers2, axes[1])
    axes[2].set_title("ol = [1.0, 0.5, ..., 0.5]", fontsize = 10)

    outliers3 = np.full((ol_count, d), .5)
    X_plus_outliers3 = np.r_[X, outliers3]
    hist_of_dm(X_plus_outliers3, axes[2])
    axes[2].set_title("ol = [.5, .5, .5, ...]", fontsize = 10)

    plt.tight_layout()
    plt.subplots_adjust(top = .85)
    plt.show()


def bo_norms(X):
    return np.sqrt(np.sum(X**2, axis = tuple(range(1, len(X.shape)))))


def dists(p, ps):
    return np.sqrt(np.sum((p - ps)**2, axis = 1))


def hist_of_dists(p, ps, plt_axis = None, hbins = 300):
    if plt_axis is None:
        _, plt_axis = plt.subplots(1, 1)

    ds = dists(p, ps)
    plt_axis.hist(ds, bins = hbins, color = "steelblue")


def check_hist_of_dists():
    d = 100
    count = 10**4
    X = np.random.random((count, d))
    fig, plt_ax = plt.subplots(4, 1, sharex = True)
    fig.suptitle(f"demo of hist_of_dists: distance of a single point in the "\
            +f"unit cube to a set "\
            +f"of\nuniformly random other points in the unit cube, dim {d} "\
            +f"count {count}",
            fontsize = 10)

    hist_of_dists(X[0], X[1:], plt_ax[0])
    plt_ax[0].set_title(f"point is first point from uniform random sample",
            fontsize = 8)

    hist_of_dists(X[1], X[np.arange(count) != 1], plt_ax[1])
    plt_ax[1].set_title(f"point is second point from uniform random sample",
            fontsize = 8)

    hist_of_dists(X[2], X[np.arange(count) != 2], plt_ax[2])
    plt_ax[2].set_title(f"point is third point from uniform random sample",
            fontsize = 8)

    outlier = np.full((1, d), .0)
    hist_of_dists(outlier, X, plt_ax[3])
    plt_ax[3].set_title("point is [0, ..., 0]", fontsize = 8)
    
    plt.tight_layout()
    plt.subplots_adjust(top = .85)
    plt.show()


def time_it(task, start_msg = "\nstarting...",
        end_msg = "... finished in {dur}, result:\n{res}", do_talk = True):
    if do_talk:
        print(start_msg)
    start = datetime.now()
    res = task()
    end = datetime.now()
    dur = str(end - start)
    if do_talk:
        print(end_msg.format(**locals()))
    return dur, res


def v_in(v, a, ax):
    sv = np.r_[f"0,{len(a.shape)},{ax}", v]
    return (np.isclose(a, sv)).all(axis = ax).any()


def v_ind_in(v, a, ax):
    sv = np.r_[f"0,{len(a.shape)},{ax}", v]
    return np.where((np.isclose(a, sv)).all(axis = ax))


def cartesian(lov):
    return np.stack(np.meshgrid(*lov), -1).reshape(-1, len(lov))


def close_pairs(arrs):
    seq = np.arange(len(arrs))
    return [(i,j) for i in seq for j in seq[i+1:] if np.allclose(arrs[i], arrs[j])]


def ex_fi_fun(funs, *args, **kwargs):
    success = False
    for fun in funs:
        try:
            r = fun(*args, **kwargs)
        except (TypeError, AssertionError) as e:
            continue
        success = True
        break
    if not success:
        raise TypeError("in ex_fi_fun: no fitting function found")
    return r


def pipe(iof, faf = True):
    assert len(iof) > 1
    if faf:
        ret = reduce(lambda a,i: lambda x: i(a(x)), iof)
    else:
        ret = reduce(lambda a,i: lambda x: a(i(x)), iof)
    return ret


def v(ln, mn = 0):
    assert ln % 2 == 1

    d = np.arange(mn + (ln-1)//2, mn, -1)
    u = np.arange(mn, mn + (ln+1)//2)

    return np.r_[d, u]


def ewedge(mx, ln, bs):
    assert ln % 2 == 1

    return np.ceil(mx / bs**v(ln)).astype(int)


def flatten(X, kept = 1):
      if  kept not in range(len(X.shape)):
          raise ValueError("argument must be in range(len(X.shape))")
      if kept == 0:
          return X.flatten()

      new_shape = np.r_[X.shape[:kept], np.prod(X.shape[kept:])]
      return X.reshape(new_shape)


def argtopk(v, k):
    # TODO: test cases
    top_inds = np.argpartition(v, -k)[-k:]
    unsorted_top = v[top_inds]
    si = np.argsort(unsorted_top)
    return top_inds[si]


###############################################################################
if __name__ == '__main__':
    #check_hist_of_dm()
    #check_hist_of_dists()
    #print(ewedge(30, 5, 1))
    print(pipe([lambda s: s + '11', lambda s: s + '22'])('foo'))
    #print(v(11))
