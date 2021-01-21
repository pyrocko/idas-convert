import logging
from time import time
from datetime import timedelta

from pyrocko.guts import String, Float

from .plugin import Plugin, PluginConfig, register_plugin
from .utils import sizeof_fmt

logger = logging.getLogger(__name__)


class TelegramBot(Plugin):

    def __init__(self, token, chat_id,
                 status_interval=60.):
        import telebot

        self.token = token
        self.chat_id = chat_id

        self.status_interval = status_interval
        self.started = time()
        self._next_status = self.started + status_interval

        self.bot = telebot.TeleBot(self.token)

    def set_parent(self, parent):
        super().set_parent(parent)

        bot = self

        class LogFilter(logging.Filter):
            def filter(self, record):
                if record.levelno < logging.WARNING:
                    return False
                if not record.name.startswith(__name__.split('.')[0]):
                    return False
                return True

        class LogHandler(logging.StreamHandler):
            terminator = ''

            def emit(self, record):
                bot.send_message('{level}: {msg}'.format(
                    level=record.levelname,
                    msg=record.getMessage()))

            def flush(self):
                ...

        handler = LogHandler()
        handler.addFilter(LogFilter())

        logging.getLogger().addHandler(handler)

        parent.finished_batch.register(self.send_status)
        self.send_message(
            'Conversion of %d files started.' % self.parent.nfiles)

    def send_message(self, message):
        try:
            return self.bot.send_message(self.chat_id, message)
        except Exception as e:
            logger.exception(e)

    def send_status(self, *args):
        if time() < self._next_status:
            return

        logger.debug('sending status message')
        stats = self.parent.stats
        self.send_message(
            'Converted {s.nfiles_processed}/{s.nfiles_total} files'
            ' (Total {size_processed}), {s.time_remaining} remaining. '
            'Average input {s.io_load_speed_avg:.1f} MB/s.'.format(
                s=stats,
                size_processed=sizeof_fmt(stats.io_load_bytes_total)))

        self._next_status += self.status_interval

    def __del__(self):
        dt = timedelta(seconds=time() - self.started)
        self.send_message('Conversion finished in %s.' % dt)


class TelegramBotConfig(PluginConfig):
    token = String.T(
        default='Telegram Token',
        help='Telegram Bot API token')
    chat_id = String.T(
        default='Telegram Chat ID, e.g. -456413218',
        help='Telegram Chat ID')

    status_interval = Float.T(
        default=3600.,
        help='Report statistics at this interval [s].')

    def get_plugin(self):
        return TelegramBot(self.token, self.chat_id, self.status_interval)


register_plugin(TelegramBotConfig)
