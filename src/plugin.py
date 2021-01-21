from pyrocko.guts import Object, Bool

PLUGINS_AVAILABLE = []


def register_plugin(plugin_config):
    global PLUGINS_AVAILABLE
    PLUGINS_AVAILABLE.append(plugin_config)


class Plugin(object):

    def set_parent(self, parent):
        self.parent = parent


class PluginConfig(Object):
    enabled = Bool.T(
        default=False)

    def get_plugin(self):
        return Plugin()
