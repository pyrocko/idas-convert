def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class Signal(object):

    def __init__(self, parent):
        self.parent = parent
        self.subscribers = set()

    def register(self, callback):
        assert callable(callback), '%s it not callable' % callback
        self.subscribers.add(callback)

    def unregister(self, callback):
        try:
            self.subscribers.remove(callback)
        except KeyError:
            pass

    def dispatch(self, *args, **kwargs):
        for callback in self.subscribers:
            callback(*args, **kwargs)
