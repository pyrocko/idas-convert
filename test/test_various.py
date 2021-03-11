from pyrocko.guts import Object

from idas_convert.meta import DataSize
from idas_convert.plugin import Plugin

def test_data_size():

    class Test(Object):
        data_size = DataSize.T()
    
    t = Test(data_size=10000)
    assert t.data_size == 10000


def test_plugin_base():
    p = Plugin()
    p.set_parent('123')