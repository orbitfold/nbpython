import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import itertools
import sys
import math
import weave
import functools


CACHE = {}


def entropy(s, start, end):
    """Calculate entropy of a numpy array s starting from start and ending with end (index).

    Parameters
    ----------
    s : NumPy array
        an array of integers to calculate entropy for
    start : int
        starting index to start calculating from
    end : int
        ending index

    Returns
    -------
    float
        entropy with base 2
    """
    try:
        return CACHE[(start, end)]
    except KeyError:
        s_ = s[start:end]
        cs = np.unique(s_)
        n = float(end - start)
        result = -sum([(nc / n) * np.log2(nc / n) for nc in
                       [float(np.where(s_ == c)[0].size) for c in cs]])
        CACHE[(start, end)] = result
        return result

    
def class_information_entropy(s, start, end, cut):
    """Calculate sum entropy of a string after dividing it in two.

    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index
    c : int
        cut point, has to be between start and end

    Returns
    -------
    float
        sum entropy of the two strings after cutting
    """
    n = float(end - start)
    n_s1 = float(cut - start)
    n_s2 = float(end - cut)
    return ((n_s1 / n) * entropy(s, start, cut) +
            (n_s2 / n) * entropy(s, cut, end))


def find_best_cut(s, start, end):
    """Find a cut point in the string s[start:end] that results in lowest sum entropy after
    dividing according to that cut point.

    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index

    Returns
    -------
    int
        index of the cut point, between start and end
    """
    if end - start <= 1:
        return -1
    best_ent = class_information_entropy(s, start, end, start + 1)
    best_cut = start + 1
    current_c = s[start]
    for i in range(start + 1, end - 1):
        # Since best cuts are always at class boundaries we can ignore strings of the same class.
        if s[i] == current_c:
            continue
        else:
            current_c = s[i]
        # Calculate sum entropy of the two cut strings.
        ent = class_information_entropy(s, start, end, i)
        if ent < best_ent:
            best_ent = ent
            best_cut = i
    return best_cut


def gain(s, start, end, c):
    """Information gain after cut. Defined as entropy of a sequence minus the sum entropy
    of the two bits after splitting.

    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index
    c : int
        cut point to evaluate, has to be between start and end

    Returns
    -------
    float
        information gain
    """
    return entropy(s, start, end) - class_information_entropy(s, start, end, c)


def find_k(s, start, end):
    """Find the number unique elements of s between start and end indices.

    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index
    
    Returns
    -------
    int
        number of unique elements
    """
    return float(np.unique(s[start:end]).size)


def delta(s, start, end, c):
    """
    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index
    c : int
        cut point index, between start and end

    Returns
    -------
    float
    """
    return (np.log2(3.0 ** find_k(s, start, end) - 2) -
            (find_k(s, start, end) * entropy(s, start, end) -
             find_k(s, start, c) * entropy(s, start, c) -
             find_k(s, c, end) * entropy(s, c, end)))


def mdlpc_criterion(s, start, end, c):
    """Should the proposed cut be accepted or not?

    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index
    c : int
        cut point index, between start and end

    Returns
    -------
    bool
        should the proposed cut-point c be accepted?
    """
    n = float(end - start)
    if end - start < 2:
        return True
    x1 = gain(s, start, end, c)
    x2 = np.log2(n - 1) / n + delta(s, start, end, c) / n
    if x1 > x2:
        return True
    else:
        return False


def cutup(s, start, end):
    """Recursively find the best cut-point. Stop when MLDPC criterion is no longer satisfied by
    the proposed cut-point.

    Parameters
    ----------
    s : NumPy array
        an array of integers
    start : int
        starting index
    end : int
        ending index

    Returns
    -------
    list of int
        list of cut-points
    """
    print('cutup(s, {}, {})'.format(start, end))
    c = find_best_cut(s, start, end)
    if c < 0:
        return []
    elif not mdlpc_criterion(s, start, end, c):
        return [c]
    else:
        return cutup(s, start, c) + cutup(s, c, end)

    
def merge_bins(bins):
    """Remove zero length bins (happens often if your attribute is integer).

    Parameters
    ----------
    bin : list of lists of type [float, float]
        list of bin boundaries

    Returns
    -------
    list of lists of type [float, float]
        bins with zero length bins removed
    """
    new_bins = []
    for bin_ in bins:
        if bin_[0] != bin_[1]:
            new_bins.append(bin_)
    return new_bins


def bins(attr, classes):
    """Discretize a continous or integer attribute by binning according to information gain
    stopping when binning does not satisfy the MDLPC criterion. Refer to the paper I cannot
    be arsed to find reference to.

    Parameters
    ----------
    attr : NumPy array
        the numerical attribute, sorted in increasing order
    classes : NumPy array
        the class values corresponding to the values of the attribute, i.e. attr[i] \in classes[i]
        for any i
    
    Returns
    -------
    list of lists of type [float, float] corresponding to proposed bin boundaries
    """
    indices = sorted(cutup(classes, 0, classes.size))
    xs = []
    for index in indices:
        xs.append((attr[index] + attr[index - 1]) * 0.5)
    return merge_bins(zip([float('-inf')] + list(xs), list(xs) + [float('+inf')]))


def get_data():
    engine = create_engine('mysql+mysqldb://nagios:vince noir@localhost/qprediction')
    query = 'select nodes_requested, if(time_spent_queued < 3600, 0, 1) as time_spent_queued from jobs where class=\"mpp2_batch\" and site=\"lrz_lx_mpp2\"'
    df = pd.read_sql(query, engine).sort_values(['nodes_requested'])
    return df

