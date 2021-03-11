import numpy as num
import pytest

from hypothesis import given
from hypothesis.strategies import floats, integers
from hypothesis.extra.numpy import arrays


from pyrocko.trace import Trace

from idas_convert.idas_convert import split, process_data

NSAMPLES = 1000


@given(arrays(num.int32, 1000, elements=integers(-1000, 1000)))
def test_split(data):
    tr = Trace(
        ydata=data,
        deltat=0.01)
    tmin_half = (tr.tmax - tr.tmin) / 2
    traces = split(tr, tmin_half)

    assert len(traces) == 2
    assert sum(t.ydata.size for t in traces) == tr.ydata.size
    assert traces[0].tmin == tr.tmin
    assert traces[1].tmax == tr.tmax


@given(arrays(num.int32, 10000, elements=integers(-1000, 1000)))
def test_process_data(data):
    tr = Trace(
        ydata=data,
        deltat=0.001)
    
    tmin = tr.tmin + .1
    tmax = tr.tmax - .1
    deltat = 0.005
    chunk = (tr, deltat, tmin, tmax)
    ptr = process_data(chunk)
    assert ptr.deltat == deltat

