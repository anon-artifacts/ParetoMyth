#!/usr/bin/env python3
# nvim --clean -c "colorscheme lunaperche" -c "set number"

import sys
import random
import math
import re
import subprocess
from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import time
from functools import lru_cache
import builtins
import numpy as np
# Global variables
the = {}
go = {}
_DIST_CACHE = {}

Big = 1E32

# Help function
def help_text(_):
    print(f"""

run_all.py : contains LINE, peek at a few rows, find a near global best
(c) 2024 [Name Redacted for Review], MIT license.

USAGE:
  python3 run_all.py [OPTIONS] [DEMO]

OPTIONS:
  -h        print help
  -a str    acquire via adapt,xplore,xplore (default: {the['acquire']})
  -k int    Bayes control  (default: {the['k']})
  -m int    Bayes control (default: {the['m']})
  -d file   csv file of data (default: {the['data']})
  -p int    coefficient of distance (default: {the['p']})
  -r int    random number seed  (default: {the['rseed']})
  -s int    #samples searched for each new centroid (default: {the['samples']})
  -b int    initial evaluation budget (default: {the['budget']})
  -B int    max evaluation budget (default: {the['Budget']})
  -T float  ratio of training data (default: {the['Trainings']})
  -f float  how far to lok for distant points (default: {the['far']})
  -l int    max length of branches (default: {the['length']})

DEMO:
  --header          header generation
  --csv    [file]   csv reader
  --data   [file]   loading rows
  --x      [file]   sorting rows by x values
  --y      [file]   sorting rows by y values
  --around [file]   find very good examples via kmeans++ initialization
""")

go["-h"] = help_text

# ## Config
the = {
    'p': 2,
    'k': 1, 'm': 2,  # Bayes control
    'data': "../../moot/optimize/misc/auto93.csv",
    'rseed': 1234567891,
    'budget': 4, 'Budget': 24,  # active learning control
    'acquire': "xplore", 'Trainings': 0.33,  # active learning control
    'samples': 32,
    'far': 0.9, 'length': 10
}

go["-a"] = lambda s: the.update({'acquire': s})
go["-b"] = lambda s: the.update({'budget': int(s)})
go["-B"] = lambda s: the.update({'Budget': int(s)})
go["-d"] = lambda s: the.update({'data': s})
go["-f"] = lambda s: the.update({'far': float(s)})
go["-k"] = lambda s: the.update({'k': int(s)})
go["-l"] = lambda s: the.update({'length': int(s)})
go["-m"] = lambda s: the.update({'m': int(s)})
go["-p"] = lambda s: the.update({'p': int(s)})
go["-r"] = lambda s: (the.update({'rseed': int(s)}), random.seed(int(s)))
go["-s"] = lambda s: the.update({'samples': int(s)})
go["-T"] = lambda s: the.update({'Trainings': float(s)})

# -----------------------------------------------------------------------------
# ## Code Conventions
# 
# - Constructors are functions with Upper case names; e.g. Num()
# - Optional args denoted by 2 blanks
# - Local variables denotes by 4 spaces.
# - In function argument lists, for non-local vars
#   - n=num, s=string, b=bool,a=array,f=fun, x=anything;   
#     e.g. sfile is a string that is a file name
#   - u is an output array.
#   - d=dict,t=a or d,
#   - ns,ss,bs = array of num,s,bs
#   - nums,cols,datas = array of Num, Cols, Datas
# - Often, 
#   - i,j,are loop counters
#   - k,v are key value pairs from dict.
#   - e,g,h,m,p,q,r,u,w,x,y are anything at all
#   - z is a variable catching the output
#
# ## References
#
# [1] https://en.wikipedia.org/wiki/Fisher-Yates_shuffle
# [2] https://link.springer.com/article/10.1007/BF00153759, section 2.4
# [3] http://tiny.cc/welford, 
# [4] chapter 20. https://doi.org/10.1201/9780429246593
# [5] https://journals.sagepub.com/doi/pdf/10.3102/10769986025002101,table1
# [6] p4 of Yang, Ying, and Geoffrey I. Webb. "A comparative study of 
#     discretization methods for naive-bayes classifiers." PKAW'02. 
# [7] https://doi.org/10.1145/568271.223812, p 169
# -----------------------------------------------------------------------------
#             
#  |   o  |_  
#  |_  |  |_) 

# ### Lists
def push(a, x):  # -> x, added to end of `a`.
    a.append(x)
    return x

def any(a):  # -> x (any items of `a`)
    tmp = a[random.randint(0, len(a) - 1)]
    assert tmp is not None, "nil selected"
    return tmp

def many(a, n=None):  # -> a (`n` random items from `a`).
    n = n or len(a)
    z = []
    for i in range(n):
        z.append(any(a))
    return z

def items(a):  # -> f. Iterator for arrays.
    return iter(a)

def split(a, n):  # -> a,a. Return all before and after `n`th item.
    u, v = [], []
    for j, x in enumerate(a):
        push(u if j < n else v, x)
    return u, v

def split_csv(s, delimiter):
    result = []
    for value in s.split(delimiter):
        result.append(value)
    return result

# ### Map
def sum(a, f):  # -> a. Return sum of items in `a` filtered by `f`.
    n = 0
    for _, x in enumerate(a):
        n = n + f(x)
    return n

def map(a, f):  # -> a. Return items in `a` filtered through `f`.
    z = []
    for _, x in enumerate(a):
        z.append(f(x))
    return z

def min_item(a, f):  # -> x (item in `a` that scores least on `f`).  # RENAMED from min
    lo = math.inf
    z = None
    any_val = None
    for _, x in enumerate(a):
        any_val = any_val or x
        n = f(x)
        if n < lo:
            lo, z = n, x
    return z or any_val

def find(a, f):
    for _, x in enumerate(a):
        if f(x):
            return x
    return None

# ### Sort
def two(f):  # -> f, sorted by `f(x) < f(y)`.
    return lambda p, q: f(p) < f(q)
def lt(x):  # -> f, sorted by `p[s] < q[s]`.
    return lambda p, q: (p[x] if isinstance(p, (list, dict)) else getattr(p, x)) < (q[x] if isinstance(q, (list, dict)) else getattr(q, x))

def gt(x):  # -> f, sorted by `a1[s] > a2[s]`.
    return lambda a1, a2: (a1[x] if isinstance(a1, (list, dict)) else getattr(a1, x)) > (a2[x] if isinstance(a2, (list, dict)) else getattr(a2, x))

def sort(a, f):  # -> a, sorted via `f`.
    from functools import cmp_to_key
    if f:
        a_copy = a[:]
        a_copy.sort(key=cmp_to_key(lambda x, y: -1 if f(x, y) else 1))
        return a_copy
    else:
        return sorted(a)

def keysort(a, f):  # -> a. Sorted via single argument function `f`.
    decorate = lambda x: [f(x), x]
    undecorate = lambda x: x[1]
    decorated = map(a, decorate)
    sorted_decorated = sort(decorated, lt(0))
    return map(sorted_decorated, undecorate)

def shuffle(a):  # -> a, randomly re-ordered via Fisher-Yates [1].
    for i in range(len(a) - 1, 1, -1):
        j = random.randint(0, i)
        a[i], a[j] = a[j], a[i]
    return a

# ## Strings to Things (and back again)
def yellow(s):
    return "\033[33m" + s + "\033[0m"

def green(s):
    return "\033[32m" + s + "\033[0m"

def red(s):
    return "\033[31m" + s + "\033[0m"

import csv as csvmod

def csv(filename):
    with open(filename, newline='') as f:
        reader = csvmod.reader(f)
        for row in reader:
            # ALWAYS treat fields as plain strings
            yield [cell.strip() for cell in row]


fmt = lambda template, *args: template % args if args else str(template)

def o(x):  # -> s. Generate a string for `x`.
    if isinstance(x, (int, float)):
        if isinstance(x, int) or x == int(x):
            return str(int(x))
        else:
            return f"{x:.3g}"
    if not isinstance(x, (list, dict)):
        return str(x)
    
    t = []
    if isinstance(x, list):
        for k, v in enumerate(x):
            t.append(o(v))
    else:
        for k, v in sorted(x.items()):
            t.append(f":{k} {o(v)}")
    
    return "{" + " ".join(t) + "}"

# ### Polymorphism
def new(methods, a):  # a, attached to a delegation table of `methods`.
    class Wrapper:
        def __init__(self, data):
            self.__dict__.update(data)
        
        def __str__(self):
            if hasattr(methods, '__tostring'):
                return methods.__tostring(self)
            return o(self.__dict__)
    
    obj = Wrapper(a)
    for name, method in methods.__dict__.items():
        if callable(method) and not name.startswith('_'):
            setattr(obj, name, lambda *args, m=method, o=obj, **kwargs: m(o, *args, **kwargs))
    
    return obj

# -----------------------------------------------------------------------------
# ### Structs
class Sym:
    """Summarize symbolic columns"""
    
    @staticmethod
    def new(s="", n=0):  # Summarize numeric columns
        return new(Sym, {
            'txt': s,         # text about this column
            'pos': n,         # column number
            'n': 0,           # how many items?
            'has': {},        # Symbol counts seen so far
            'mode': None,     # most common symbol
            'most': 0         # frequency of most common symbol
        })
    
    def add(self, x):  # -> x. Updates Sym
        if x == "?":
            return x
        self.n = self.n + 1
        self.has[x] = 1 + self.has.get(x, 0)
        if self.has[x] > self.most:
            self.most, self.mode = self.has[x], x
        return x
    
    def dist(self, p, q):  # -> n. Distance between two symbols.
        if p == "?" and q == "?":
            return 1
        return 0 if p == q else 1
    
    def like(self, x, nPrior):  # -> n. How much this `Sym` likes this `n`. [6]
        return ((self.has.get(x, 0) + the['m'] * nPrior) / (self.n + the['m']))


class Num:
    """Summarize numeric columns"""
    
    @staticmethod
    def new(s="", n=0):  # Summarize numeric columns
        return new(Num, {
            'txt': s,                                    # text about this column
            'pos': n,                                    # column number
            'n': 0,                                      # how many items?
            'lo': Big,                                   # smallest number seen in a column
            'hi': -Big,                                  # largest number seen in a column
            'utopia': 0 if (s or "").find("-") >= 0 and (s or "").endswith("-") else 1,  # (min|max)imize = 0,1
            'mu': 0,                                     # mean
            'm2': 0,                                     # second moment (used for sd calculation)
            'sd': 0                                      # standard deviation
        })
    
    def add(self, n):  
        if n == "?":
            return n
        self.n += 1

        # Handle True/False
        if isinstance(n, str):
            if n.lower() == "true":
                n = 1.0
            elif n.lower() == "false":
                n = 0.0

        try:
            n = float(n)
        except Exception:
            raise TypeError(
                f"Column '{self.txt}' expected numeric but got '{n}' (type={type(n)})"
            )

        delta = n - self.mu
        self.mu += delta / self.n
        self.m2 += delta * (n - self.mu)
        self.sd = 0 if self.n < 2 else (self.m2 / (self.n - 1)) ** 0.5
        self.lo = min(n, self.lo)
        self.hi = max(n, self.hi)
        return n

    
    def normalize(self, n):  # -> 0...1
        if n == "?":
            return n
        return (n - self.lo) / (self.hi - self.lo + 1 / Big)
    
    def dist(self, p, q):  # -> n. Distance between two numbers.
        if p == "?" and q == "?":
            return 1  # if all unknown, assume max
        p = self.normalize(p) if p != "?" else (0 if q < 0.5 else 1)  # when in doubt, assume max
        q = self.normalize(q) if q != "?" else (0 if p < 0.5 else 1)  # when in doubt, assume max
        return abs(p - q)
    
    def like(self, x, _=None):  # -> n. How much this `Num` likes this `n`.
        v = self.sd ** 2 + 1 / Big
        tmp = math.exp(-1 * (x - self.mu) ** 2 / (2 * v)) / (2 * math.pi * v) ** 0.5
        return max(0, min(1, tmp + 1 / Big))


class Data:
    """Holds the rows and column summaries."""

    @staticmethod
    def new(src):  # Holds the rows and column summaries.
        first_row = next(src) if callable(src) or hasattr(src, '__next__') else src
        self = new(Data, {
            'cols': Cols.new(first_row),  # column information
            'rows': []                     # set of rows
        })
        if callable(src) or hasattr(src, '__next__'):
            return self.adds(src)
        return self
    
    def clone(self, rows=None):  # -> Data. Copies self's column structure.
        self_new = new(Data, {
            'cols': Cols.new(self.cols.names),
            'rows': []
        })
        if rows:
            return self_new.adds(items(rows))
        return self_new
    
    def add(self, a):  # -> Data, updated with one row.
        for _, col in enumerate(self.cols.all):
            a[col.pos] = col.add(a[col.pos])
        push(self.rows, a)
        return self
    
    def adds(self, src):  # -> Data, updated with many rows
        for a in src:
            self.add(a)
        return self
    
    def xdist(self, r1, r2):
        key = (id(r1), id(r2)) if id(r1) < id(r2) else (id(r2), id(r1))
        if key in _DIST_CACHE:
            return _DIST_CACHE[key]

        d, n = 0, 0
        for col in self.cols.x:
            x = r1[col.pos]
            y = r2[col.pos]
            if x != '?' and y != '?':
                n += 1
                d += abs(col.dist(x, y)) ** the['p']

        dist = (d / n) ** (1 / the['p']) if n > 0 else 1
        _DIST_CACHE[key] = dist
        return dist



        
    def ydist(self, row):  # -> n. Distance of y cols to utopia points.
        d = 0
        f = lambda col: col.normalize(row[col.pos]) - col.utopia
        for _, col in enumerate(self.cols.y):
            d = d + abs(f(col)) ** the['p']
        return (d / len(self.cols.y)) ** (1 / the['p'])
    
    def neighbors(self, row1, rows=None):  # -> a (rows, sorted by distance to row1)
        f = lambda row2: self.xdist(row1, row2)
        return keysort(rows or self.rows, f)
    
    def anys(self, budget):  # -> rows
        return many(self.rows, budget)
    
    def dehb(self, file=None, budget=0, rp=0):
        # Define the Python command to execute
        filepath = file or the['data']
        command = f'python3 ../FileResultsReader.py --data_file_path {filepath} --folder_name ../../results/results_DEHB/DEHB --budget {budget}'
        
        # Run the command using subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        
        # Step 1: Extract the two lists using pattern matching
        match = re.search(r'\((\[.*?\])\s*,\s*(\[.*?\])\)', output, re.DOTALL)
        
        if not match:
            return [], []
        
        list1_str = match.group(1)
        list2_str = match.group(2)
        
        # Helper function to process each list string
        def parse_list(list_str):
            numbers = []
            for value in re.findall(r"'(.*?)'", list_str):
                num = float(value.strip())
                numbers.append(num)
            return numbers
        
        # Step 2: Parse both lists
        scores = parse_list(list1_str)
        times = parse_list(list2_str)
        
        return scores, times
    
    def actLearn(self, file=None, budget=0, rp=0):
        # Define the Python command to execute
        filepath = file or the['data']
        command = f'python3 ../FileResultsReader.py --data_file_path {filepath} --folder_name ../../results/results_Active_Learning/Active_Learning --budget {budget}'
        
        # Run the command using subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        
        # Step 1: Extract the two lists using pattern matching
        match = re.search(r'\((\[.*?\])\s*,\s*(\[.*?\])\)', output, re.DOTALL)
        
        if not match:
            return [], []
        
        list1_str = match.group(1)
        list2_str = match.group(2)
        
        # Helper function to process each list string
        def parse_list(list_str):
            numbers = []
            for value in re.findall(r"'(.*?)'", list_str):
                num = float(value.strip())
                numbers.append(num)
            return numbers
        
        # Step 2: Parse both lists
        scores = parse_list(list1_str)
        times = parse_list(list2_str)
        
        return scores, times

    def random(self, file=None, budget=0, rp=0):
        filepath = file or the['data']
        command = (
            f'python3 ../FileResultsReader.py '
            f'--data_file_path {filepath} '
            f'--folder_name ../../results/results_RandomSearch/RandomSearch '
            f'--budget {budget}'
        )

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout

        match = re.search(r'\((\[.*?\])\s*,\s*(\[.*?\])\)', output, re.DOTALL)
        if not match:
            return [], []

        list1_str = match.group(1)
        list2_str = match.group(2)

        def parse_list(list_str):
            numbers = []
            for value in re.findall(r"'(.*?)'", list_str):
                numbers.append(float(value.strip()))
            return numbers

        scores = parse_list(list1_str)
        times = parse_list(list2_str)

        return scores, times

    def smac(self, file=None, budget=0, rp=0):
        # Define the Python command to execute
        filepath = file or the['data']
        command = f'python3 ../FileResultsReader.py --data_file_path {filepath} --folder_name ../../results/results_SMAC/SMAC --budget {budget}'
        
        # Run the command using subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        
        # Step 1: Extract the two lists using pattern matching
        match = re.search(r'\((\[.*?\])\s*,\s*(\[.*?\])\)', output, re.DOTALL)
        
        if not match:
            return [], []
        
        list1_str = match.group(1)
        list2_str = match.group(2)
        
        # Helper function to process each list string
        def parse_list(list_str):
            numbers = []
            for value in re.findall(r"'(.*?)'", list_str):
                num = float(value.strip())
                numbers.append(num)
            return numbers
        
        # Step 2: Parse both lists
        scores = parse_list(list1_str)
        times = parse_list(list2_str)
        
        return scores, times

    def kmplusplus(self, file=None, budget=0, rp=0):
        # Define the Python command to execute
        filepath = file or the['data']
        command = f'python3 ../FileResultsReader.py --data_file_path {filepath} --folder_name ../../results/results_KMPlusPlus/KMPlusPlus --budget {budget}'
        
        # Run the command using subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        
        # Step 1: Extract the two lists using pattern matching
        match = re.search(r'\((\[.*?\])\s*,\s*(\[.*?\])\)', output, re.DOTALL)
        
        if not match:
            return [], []
        
        list1_str = match.group(1)
        list2_str = match.group(2)
        
        # Helper function to process each list string
        def parse_list(list_str):
            numbers = []
            for value in re.findall(r"'(.*?)'", list_str):
                num = float(value.strip())
                numbers.append(num)
            return numbers
        
        # Step 2: Parse both lists
        scores = parse_list(list1_str)
        times = parse_list(list2_str)
        
        return scores, times


    def to_vec(self, row):
        """
        Convert a row into a numeric NumPy vector of its x columns.
        Compatible with Wrapper-style rows or dicts.
        """
        try:
            vals = []

            if hasattr(self, "cols") and hasattr(self.cols, "x") and self.cols.x:
                # For each column in x
                for col in self.cols.x:
                    key = col if isinstance(col, str) else getattr(col, "name", None) or getattr(col, "txt", None)
                    if key is None:
                        continue

                    # Try to access either attribute or dict-style
                    val = getattr(row, key, None)
                    if val is None and isinstance(row, dict):
                        val = row.get(key)

                    if isinstance(val, (int, float)):
                        vals.append(val)

            # Fallback if cols.x didn’t work or was empty
            if not vals:
                if isinstance(row, dict):
                    vals = [v for v in row.values() if isinstance(v, (int, float))]
                else:
                    # Collect all numeric attributes in Wrapper
                    vals = [getattr(row, k) for k in dir(row)
                            if not k.startswith("_") and isinstance(getattr(row, k), (int, float))]

            return np.array(vals, dtype=np.float64)
        except Exception as e:
            raise TypeError(f"Cannot convert row to numeric vector: {e}")



    def around(self, budget, rows=None):
        rows = rows or self.rows
        z = [any(rows)]

        # --- Pre-sample once ---
        sample_size = builtins.min(the['samples'] * budget, len(rows))
        candidates = many(rows, sample_size)
        cand_idx = 0

        # --- Convert to NumPy arrays for vectorized math ---
        # Assuming each row can be represented as numeric array of features
        # (e.g., self.to_vec(row) returns a numeric numpy array)
        vec_cache = {}
        def get_vec(r):
            # Cache the numeric version of each row
            if id(r) not in vec_cache:
                vec_cache[id(r)] = np.asarray(self.to_vec(r), dtype=np.float64)
            return vec_cache[id(r)]

        # Convert all rows for fast indexing
        for r in z + candidates:
            get_vec(r)

        for k in range(2, budget + 1):
            all_sum = 0.0
            u = []

            # Determine how many candidates to process this round
            inner_count = builtins.min(the['samples'], len(candidates) - cand_idx)
            new_rows = candidates[cand_idx : cand_idx + inner_count]
            cand_idx += inner_count

            # Stack centroids into an array (n_centroids, dim)
            z_vecs = np.stack([get_vec(c) for c in z])

            # Vectorized distance computation for all new rows
            new_vecs = np.stack([get_vec(r) for r in new_rows])  # shape (inner_count, dim)
            # Compute pairwise distances: (inner_count, n_centroids)
            # Efficient broadcasted Euclidean distance
            diffs = new_vecs[:, None, :] - z_vecs[None, :, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=2))

            # Find minimum distance per candidate
            closest_dists = np.min(dists, axis=1)
            d_sq = closest_dists ** 2

            # Compute cumulative weighted selection pool
            for row, d_val in zip(new_rows, d_sq):
                all_sum += push(u, {'row': row, 'd': d_val})['d']

            # --- Random pick step ---
            r = all_sum * random.random()
            one = None
            for x in u:
                r -= x['d']
                one = x['row']
                if r <= 0:
                    break
            push(z, one)

        return z


    
    def twoFar(self, repeats, rows, sortp, above=None):  # -> n,row,row
        far = int(the['far'] * len(rows))
        a = above or self.neighbors(any(rows), rows)[far]
        b = self.neighbors(a, rows)[far]
        if sortp and self.ydist(b) < self.ydist(a):
            a, b = b, a
        return self.xdist(a, b), a, b
    
    def half(self, rows, sortp, above=None):  # -> rows,rows,rows,row,m [7]
        lefts, rights = [], []
        c, left, right = self.twoFar(the['far'], rows, sortp, above)
        
        def cos(a, b):
            return (a ** 2 + c ** 2 - b ** 2) / (2 * c + 1E-32)
        
        def f(r):
            return {
                'd': cos(self.xdist(r, left), self.xdist(r, right)),
                'row': r
            }
        
        sorted_items = sort(map(rows, f), lt('d'))
        for i, one in enumerate(sorted_items, 1):  # Start from 1 like Lua
            push(lefts if i <= len(rows) // 2 else rights, one['row'])
        
        return lefts, left, rights, right, self.xdist(left, rights[0])
    
    def branch(self, budget, rows=None, above=None):  # -> rows
        rows = rows or self.rows
        budget = budget or the['budget']
        if budget < 1 or len(rows) < 2:
            return rows
        lefts, left, _, _, _ = self.half(rows, True, above)
        return self.branch(budget - 1, lefts, left)

    
    def loglike(self, row, nall, nh):  # -> n. How much does Data likes row?
        prior = (len(self.rows) + the['k']) / (nall + the['k'] * nh)
        
        def l(n):
            return math.log(n) if n > 0 else 0
        
        def f(x):
            return l(x.like(row[x.pos], prior))
        
        return l(prior) + sum(self.cols.x, f)
    
    def acquire(self, budget=None):
        budget = budget or the['budget']
        
        Y = lambda r: self.ydist(r)
        
        def B(r):
            return math.exp(best.loglike(r, len(done), 2))
        
        def R(r):
            return math.exp(rest.loglike(r, len(done), 2))
        
        def BR(r):
            return acq[the['acquire']](B(r), R(r), len(done) / the['Budget'])
        
        n = min(500, int(the['Trainings'] * len(self.rows)))
        train, test = split(shuffle(self.rows[:]), n)
        test, _ = split(test, min(500, len(test)))
        done, todo = split(train, the['budget'])  # --- 1.
        
        while True:
            done = keysort(done, Y)
            if len(done) > the['Budget'] - 4 or len(todo) < 5:
                break  # --- 6.
            best_split, rest_split = split(done, int(math.sqrt(len(done))))  # --- 2.
            best, rest = self.clone(best_split), self.clone(rest_split)      # --- 3.
            todo = keysort(todo, BR)                                          # --- 4.
            for _ in range(2):                                                # --- 5.
                push(done, todo.pop())
                push(done, todo.pop(0))
        
        return done, test, BR  # --- 7.
    
    def find_row(self, csv_string):
        row = split_csv(csv_string, ",")  # Convert CSV string to a table
        for _, existing_row in enumerate(self.rows):
            match = True
            for i, value in enumerate(row):  # Use enumerate for ordered traversal
                print(existing_row[i])
                if str(existing_row[i]) != value:
                    match = False
                    break
            if match:
                return existing_row  # Return the matching row
        return False  # Return false if no match is found


class Cols:
    """Make and store Nums and Syms"""
    
    @staticmethod
    def new(ss):  # Make and store Nums and Syms
        self = new(Cols, {
            'names': ss,      # all the names
            'klass': None,    # klass column, if it exists
            'all': [],        # all the cols
            'x': [],          # the independent columns
            'y': []           # the dependent columns
        })
        return self.initialize(ss)
    
    def initialize(self, ss):  # -> Cols, col names in `ss` turned to Nums or Syms
        for n, s in enumerate(ss):
            col = (Num if s[0].isupper() else Sym).new(s, n)  # make Nums or Syms
            push(self.all, col)                                # put `col` in `all`
            if not s.endswith('X'):                            # ignore some cols
                push(self.y if re.search(r'[!+\-]$', s) else self.x, col)  # keep in `x` or `y`
                if s.endswith('!'):
                    self.klass = col
        return self


class Sample:
    """Like Num, but also keeps all the nums."""
    
    @staticmethod
    def new(s=""):  # -> Sample. Like Num, but also keeps all the nums.
        return new(Sample, {
            'txt': s,
            'n': 0,
            'mu': 0,
            'm2': 0,
            'sd': 0,
            'lo': Big,
            'hi': -Big,
            'all': []
        })
    
    def add(self, n):  # -> n. Update a Sample with `n`.
        if n == "?":
            return n
        self.n = self.n + 1
        n = float(n)  # ensure we have numbers
        delta = n - self.mu
        self.mu = self.mu + delta / self.n
        self.m2 = self.m2 + delta * (n - self.mu)
        self.sd = 0 if self.n < 2 else (self.m2 / (self.n - 1)) ** 0.5
        self.lo = min(n, self.lo)
        self.hi = max(n, self.hi)
        return push(self.all, n)
    
    def normalize(self, x):
        return (x - self.lo) / (self.hi - self.lo + 1 / Big)
    
    @staticmethod
    def delta(i, j):  # -> n. Report mean difference, normalized by sd.
        return abs(i.mu - j.mu) / ((1E-32 + i.sd ** 2 / i.n + j.sd ** 2 / j.n) ** 0.5)
    
    def cohen(self, other, d=0.35):  # -> b. Parametric effect size.
        i, j = self, other
        sd = (((i.n - 1) * i.sd ** 2 + (j.n - 1) * j.sd ** 2) / (i.n + j.n - 2)) ** 0.5
        return abs(i.mu - j.mu) <= d * sd
    
    def same(self, other, delta=None, straps=None, conf=None):
        i, j = self, other

        if len(i.all) == 0 or len(j.all) == 0:
            return False

        delta  = delta  if delta  is not None else 0.197
        straps = straps if straps is not None else 512
        conf   = conf   if conf   is not None else 0.05

        return cliffs(i.all, j.all, delta) and boot(i.all, j.all, straps, conf)

    
    def __str__(self):  # -> s. Reports some details of Samples.
        return fmt("Sample{%s %g %g %g}", self.txt, self.n, self.mu, self.sd)
    
    def merge(self, other, eps=0.01):
        i, j = self, other
        if abs(i.mu - j.mu) < eps or i.same(j):
            k = Sample.new(i.txt)
            for _, t in enumerate([i.all, j.all]):
                for _, x in enumerate(t):
                    k.add(x)
            return k
        return None
    
    @staticmethod
    def merges(samples, eps):
        pos = {}
        t = None
        for k, sample in enumerate(samples):
            if t:
                merged = sample.merge(t[-1], eps)
                if merged:
                    t[-1] = merged
                else:
                    push(t, sample)
            else:
                t = [sample]
            pos[k] = [sample, len(t)]
        
        for _, two in pos.items():
            sample = two[0]
            sample._meta = t[two[1] - 1]
            sample._meta.rank = chr(96 + two[1])
        
        return samples


# -------------------------------------------------------------------------------
#  ___                    
# |_)   _.       _    _ 
# |_)  (_|  \/  (/_  _> 
#           /           

acq = {
    'xplore': lambda b, r, _: (b + r) / abs(b - r + 1 / Big),
    'xploit': lambda b, r, _: b / (r + 1 / Big),
    'adapt': lambda b, r, p: (b + r * (1 - p)) / abs(b * (1 - p) - r + 1 / Big)
}

# -------------------------------------------------------------------------------
#   __                    
# (_   _|_   _.  _|_   _ 
# __)   |_  (_|   |_  _> 

def adds(ns):  # -> Sample. Load numbers in `ns` into a Sample.
    s = Sample.new()
    for _, n in enumerate(ns):
        s.add(n)
    return s

# Checks how rare are the observed differences between samples of this data.
# If not rare, then these sets are the same. From [4]
def boot(y0, z0, straps=512, conf=0.05):
    z, y = adds(z0), adds(y0)
    x = adds(y0 + z0)
    yhat = map(y0, lambda y1: y1 - y.mu + x.mu)
    zhat = map(z0, lambda z1: z1 - z.mu + x.mu)
    n = 0
    for _ in range(straps):
        if adds(many(yhat)).delta(adds(many(zhat))) > y.delta(z):
            n = n + 1
    return n / straps >= conf

# How central are `ys` items in `xs`? If central, then the sets are the same.
def cliffs(xs, ys, delta=None):  # -> b. [5]
    delta = delta or 0.197
    lt, gt, n = 0, 0, 0
    for _, x in enumerate(xs):
        for _, y in enumerate(ys):
            n = n + 1
            if y > x:
                gt = gt + 1
            if y < x:
                lt = lt + 1
    return abs(gt - lt) / n <= delta  # 0.195

def normal(mu=0, sd=1):
    return (mu + sd * math.sqrt(-2 * math.log(random.random())) *
            math.cos(2 * math.pi * random.random()))

# -------------------------------------------------------------------------------
#  ___                                       
#   |    _    _  _|_     _   _.   _   _    _ 
#   |   (/_  _>   |_    (_  (_|  _>  (/_  _> 
                                           
def run(ss, funs, rSeed=1234567891):
    fails = 0
    for _, one in enumerate(ss):
        random.seed(rSeed)
        try:
            funs[one](None)
            print(green(f"pass for '{one}'"))
        except Exception as msg:
            print(red(f"FAILURE for '{one}' :{msg}"))
            import traceback
            traceback.print_exc()
            fails = fails + 1
    print(yellow(fmt("%s failure(s)", fails)))
    sys.exit(fails)

# e.g. command line option `python run_all.py --the` calls `go["--the"]()`
def go_the(_):
    print(o(the))

go["--the"] = go_the

def go_header(file=None):
    data = Data.new([["name", "Age", "Shoesize-"]])
    for k, col in enumerate(data.cols.x):
        print("x", k, col)
    for k, col in enumerate(data.cols.y):
        print("y", k, col)

go["--header"] = go_header

def go_csv(file=None):
    k = 0
    for row in csv(file or the['data']):
        k = k + 1
        if k == 1 or k % 30 == 0:
            print(o(row))

go["--csv"] = go_csv

def go_data(file=None):
    data = Data.new(csv(file or the['data']))
    print(len(data.rows), o(data.cols.y[0].__dict__))

go["--data"] = go_data

def go_xs(file=None):
    data = Data.new(csv(file or the['data']))
    X = lambda row: data.xdist(data.rows[0], row)
    XX = lambda a, b: X(a) < X(b)
    for k, row in enumerate(sort(data.rows, XX)):
        if k == 0 or k % 30 == 0:
            print(o(row), X(row))

go["--xs"] = go_xs

def go_ys(file=None):
    data = Data.new(csv(file or the['data']))
    Y = lambda row: data.ydist(row)
    YY = lambda a, b: Y(a) < Y(b)
    for k, row in enumerate(sort(data.rows, YY)):
        if k == 0 or k % 30 == 0:
            print(o(row), Y(row))

go["--ys"] = go_ys

def go_bayes(file=None):
    data = Data.new(csv(file or the['data']))
    like = lambda row: data.loglike(row, 1000, 2)
    for i in range(200):
        print(fmt("%.2f", like(any(data.rows))))

go["--bayes"] = go_bayes

def go_around(file=None):
    data = Data.new(csv(file or the['data']))
    Y = lambda row: data.ydist(row)
    budget = the['Budget']
    for _ in range(20):
        shuffle(data.rows)
        print(Y(sort(data.around(budget), two(Y))[0]))

go["--around"] = go_around

def go_stats1(_):
    r = 5
    print("r", fmt("\tmu\t\tsd"))
    while r < 50000:
        r = r * 2
        s = Sample.new()
        for i in range(r):
            s.add(normal(100, 20))
        print(r, fmt("\t%g\t\t%g", s.mu, s.sd))

go["--stats1"] = go_stats1

def go_stats2(_):
    dot = lambda s: "y" if s else "."
    print("d\tclif\tboot\tsame\tcohen")
    d = 1.0
    while d <= 1.25:
        t = Sample.new()
        for i in range(50):
            t.add(normal(5, 1) + normal(10, 2) ** 2)
        u = Sample.new()
        for _, x in enumerate(t.all):
            u.add(x * d)
        print(fmt("%.3f\t%s\t%s\t%s\t%s",
                  d,
                  dot(cliffs(t.all, u.all)),
                  dot(boot(t.all, u.all)),
                  dot(t.same(u)),
                  dot(t.cohen(u))))
        d = d + 0.02

go["--stats2"] = go_stats2

def go_acquire(file=None):
    data = Data.new(csv(file or the['data']))
    Y = lambda row: data.ydist(row)
    for i in range(20):
        done, _, _ = data.acquire()
        print(Y(keysort(done, Y)[0]))

go["--acquire"] = go_acquire

def go_compare(file=None):
    data = Data.new(csv(file or the['data']))
    Y = lambda row: data.ydist(row)
    for i in range(20):
        done = data.around(25)
        print(Y(keysort(done, Y)[0]))

go["--compare"] = go_compare

def _asIs(file=None):
    data = Data.new(csv(file or the['data']))
    Y = lambda row: data.ydist(row)
    return data, adds(map(data.rows, Y)), Y

def go_branch(file=None):
    data, b4, Y = _asIs(file)
    S = lambda x: print(fmt("%.0f", 100 * x))
    print(fmt("%.0f", 100 * b4.mu))
    for _ in range(20):
        shuffle(data.rows)
        S(Y(keysort(data.branch(20), Y)[0]))

go["--branch"] = go_branch

def go_comparez(file=None):
    _comparez(file, "mu")

go["--comparez"] = go_comparez

def printTable(tbl, indent=""):
    for key, value in tbl.items():
        if isinstance(value, dict):
            # If the value is a table, print its key and recurse
            print(indent + str(key) + ":")
            printTable(value, indent + "  ")
        else:
            # Otherwise, print the key-value pair
            print(indent + str(key) + ": " + str(value))
            
def _acquire_xploit(data, budget):
    the['acquire'] = 'xploit'
    done, test, BR = data.acquire(budget)
    return done

def _acquire_xplore(data, budget):
    the['acquire'] = 'xplore'
    done, test, BR = data.acquire(budget)
    return done

def _acquire_adapt(data, budget):
    the['acquire'] = 'adapt'
    done, test, BR = data.acquire(budget)
    return done

import time, re, sys
def _comparez(file=None, IT="mu"):
    file = file or the['data']
    Budget  = 25
    Repeats = 20
    Epsilon = 0.35

    data = Data.new(csv(file))
    Y = lambda row: data.ydist(row)
    BEST = lambda a: Y(keysort(a, Y)[0])
    N = lambda x: fmt("%.0f", 100 * x)

    # ----------------------------------------------------------
    # B4 = adds(map(data.rows, Y))    (Lua behaviour)
    # ----------------------------------------------------------
    B4 = Sample.new("Before")
    for r in data.rows:
        B4.add(Y(r))

    # ----------------------------------------------------------
    # TASK DEFINITIONS
    # ----------------------------------------------------------
    TASKS = [
        ["ACT-6",   lambda: data.actLearn(file, 6, Repeats)],
        ["ACT-12",  lambda: data.actLearn(file, 12, Repeats)],
        ["ACT-18",  lambda: data.actLearn(file, 18, Repeats)],
        ["ACT-24",  lambda: data.actLearn(file, 24, Repeats)],
        ["ACT-50",  lambda: data.actLearn(file, 50, Repeats)],
        ["ACT-100", lambda: data.actLearn(file, 100, Repeats)],
        ["ACT-200", lambda: data.actLearn(file, 200, Repeats)],

        ["SMAC-6",   lambda: data.smac(file, 6, Repeats)],
        ["SMAC-12",  lambda: data.smac(file, 12, Repeats)],
        ["SMAC-18",  lambda: data.smac(file, 18, Repeats)],
        ["SMAC-24",  lambda: data.smac(file, 24, Repeats)],
        ["SMAC-50",  lambda: data.smac(file, 50, Repeats)],
        ["SMAC-100", lambda: data.smac(file, 100, Repeats)],
        ["SMAC-200", lambda: data.smac(file, 200, Repeats)],

        # ---------- REPLACED KM++ WITH AROUND ----------
        ["AROUND-6",   ("around", 6)],
        ["AROUND-12",  ("around", 12)],
        ["AROUND-18",  ("around", 18)],
        ["AROUND-24",  ("around", 24)],
        ["AROUND-50",  ("around", 50)],
        ["AROUND-100", ("around", 100)],
        ["AROUND-200", ("around", 200)],

        ["RAND-6",   lambda: data.random(file, 6, Repeats)],
        ["RAND-12",  lambda: data.random(file, 12, Repeats)],
        ["RAND-18",  lambda: data.random(file, 18, Repeats)],
        ["RAND-24",  lambda: data.random(file, 24, Repeats)],
        ["RAND-50",  lambda: data.random(file, 50, Repeats)],
        ["RAND-100", lambda: data.random(file, 100, Repeats)],
        ["RAND-200", lambda: data.random(file, 200, Repeats)],
    ]

    # ----------------------------------------------------------
    # EVALUATION LOOP (Lua-correct for around)
    # ----------------------------------------------------------
    rxs = []

    for task in TASKS:
        name = task[0]
        entry = task[1]

        sys.stderr.write("<" + name)

        score_samp = Sample.new(name)
        time_samp  = Sample.new(name + "_time")
        task.append(score_samp)
        task.append(time_samp)
        rxs.append(score_samp)

        # ----------------------------------------------
        # SPECIAL CASE: AROUND optimizer (Lua semantics)
        # ----------------------------------------------
        if isinstance(entry, tuple) and entry[0] == "around":
            budget = entry[1]

            for _ in range(Repeats):
                shuffle(data.rows)
                rows = data.around(budget)            # list of k centroids
                best = Y(keysort(rows, Y)[0])         # EXACT Lua logic
                score_samp.add(best)
                time_samp.add(0)                      # Lua does not time around()

        else:
            # ------------------------------------------
            # Standard optimizers (DEHB / SMAC / ACT / RANDOM)
            # ------------------------------------------
            scores, times = entry()

            if not scores:
                score_samp.add(0)
                time_samp.add(0)
            else:
                for s in scores: score_samp.add(float(s))
                for t in times:  time_samp.add(float(t))

        sys.stderr.write(">")

    # ----------------------------------------------------------
    # SKIP DATASET IF ANY OPTIMIZER HAS NO RESULTS
    # ----------------------------------------------------------
    for task in TASKS:
        name = task[0]
        if len(task) < 4:
            continue

        score_samp = task[2]

        if len(score_samp.all) == 0 or (len(score_samp.all) == 1 and score_samp.all[0] == 0):
            sys.stderr.write(
                f"\n[SKIP] Dataset '{file}' skipped because optimizer '{name}' has no results.\n"
            )
            return

    # ----------------------------------------------------------
    # ADD B4 + MERGE CLUSTERS (same as Lua)
    # ----------------------------------------------------------
    rxs.append(B4)
    TASKS.append(["Before", None, B4])

    sorted_rxs = Sample.merges(sort(rxs, lt('mu')), B4.sd * Epsilon)

    # ----------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------
    print(" ")

    names = map(TASKS, lambda task: task[0] + "," + task[0] + "_time")
    print("D,#R,#X,#Y,B4."+IT+",B4.lo,B4.sd,2B."+IT+","+",".join(names)+",File")

    report = [
        N((B4.mu - sorted_rxs[0]._meta.mu) / (B4.mu - B4.lo)),
        len(data.rows),
        len(data.cols.x),
        len(data.cols.y),
        fmt("%.0f", 100 * B4.mu),
        fmt("%.0f", 100 * B4.lo),
        fmt("%.0f", 100 * B4.sd),
        fmt("%.0f", 100 * getattr(sorted_rxs[0]._meta, IT)),
    ]

    for task in TASKS:
        if len(task) < 4:
            continue
        score_samp, time_samp = task[2], task[3]
        report.append(fmt("%.0f %s", 100 * getattr(score_samp, IT), score_samp._meta.rank))
        report.append(fmt("%.2f", 1000 * getattr(time_samp, IT)))

    report.append(re.sub(r'^.*/', '', file))
    print(", ".join(map(report, str)))




def go_xomo(file=None):
    Data_obj, B4, Y = _asIs(file or the['data'])
    print(B4.mu, B4.lo)
    print(len(Data_obj.around(30)))
    for _, row in enumerate(Data_obj.around(30)):
        print(Y(row))

go["--xomo"] = go_xomo

def go_compares(file=None):
    def SORTER(a, b):
        return (a._meta.mu < b._meta.mu or
                (a._meta.mu == b._meta.mu and a.txt < b.txt))
    
    G = lambda x: fmt("%.2f", x)
    G0 = lambda x: int(100 * x)
    repeats = 50
    data = Data.new(csv(file or the['data']))
    Y = lambda row: data.ydist(row)
    b4 = Sample.new(0)
    all_samples = [b4]
    copy = {0: b4}
    
    for _, r in enumerate(data.rows):
        all_samples[0].add(Y(r))
    
    random.seed(1)
    for _, k in enumerate([15, 20, 25, 30, 40, 80, 160]):
        push(all_samples, Sample.new(k))
        copy[k] = all_samples[-1]
        for _ in range(repeats):
            shuffle(data.rows)
            all_samples[-1].add(sort(map(data.around(k), Y))[0])
    
    # ---------
    first = sort(Sample.merges(sort(all_samples, lt('mu')), b4.sd * 0.35), SORTER)[0]
    want = first.txt
    rand = Sample.new(-1)
    push(all_samples, rand)
    copy[-1] = rand
    
    for _ in range(repeats):
        shuffle(data.rows)
        u = []
        for j, row in enumerate(data.rows):
            if j >= want:
                break
            else:
                push(u, Y(row))
        rand.add(sort(u)[0])
    
    # ---------
    all_samples = sort(Sample.merges(sort(all_samples, lt('mu')), b4.sd * 0.35), SORTER)
    report = [
        G0((b4.mu - first._meta.mu) / (b4.mu - b4.lo)),  # 1 = delta
        len(data.rows),  # 2 = nrows
        data.cols.x,     # 3 = xs
        data.cols.y,     # 4 = ys
        G(b4.lo)         # 5 = lo
    ]
    
    for _, k in enumerate([0, -1, 15, 20, 25, 30, 40, 80, 160]):
        push(report, fmt("%.2f%s", copy[k].mu, copy[k]._meta.rank))
    
    push(report, re.sub(r'^.*/', '', file or the['data']))
    print(", ".join(map(report, str)))

go["--compares"] = go_compares

def go_all(_):
    run(["--header", "--csv", "--data", "--xs", "--ys", "--around"], go, the['rseed'])

go["--all"] = go_all

# -------------------------------------------------------------------------------
#   __                                
# (_   _|_   _.  ._  _|_         ._  
# __)   |_  (_|  |    |_    |_|  |_) 
#                                |   

random.seed(the['rseed'])

if __name__ == '__main__':
    k = 0
    while k < len(sys.argv):
        v = sys.argv[k]
        if v in go:
            next_arg = sys.argv[k + 1] if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith('-') else None
            go[v](next_arg)
        k = k + 1