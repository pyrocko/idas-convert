import os
import re
import logging
import threading

from time import time
from glob import iglob
from datetime import timedelta, datetime
from itertools import repeat, chain
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as num
from pyrocko import io, trace
from pyrocko.io.tdms_idas import detect as detect_tdms
from pyrocko.util import tts

from pyrocko.guts import Object, String, Int, List, Timestamp

from .plugin import PluginConfig, PLUGINS_AVAILABLE
from .meta import Path
from .utils import Signal

guts_prefix = 'idas'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('idas_convert')

nptdms_logger = logging.getLogger('nptdms.base_segment')
nptdms_logger.setLevel(logging.ERROR)
op = os.path

day = 3600. * 24


@dataclass
class Stats(object):
    io_load_t: float = 0.
    io_load_t_total: float = 0.
    io_save_t: float = 0.
    io_save_t_total: float = 0.

    io_load_bytes: int = 0
    io_load_bytes_total: int = 0
    tprocessing: float = 0.
    tprocessing_total: float = 0.

    nfiles_total: int = 0
    nfiles_processed: int = 0
    time_processing: float = 0.

    time_start: float = time()

    def new_io_load(self, t, bytes):
        self.io_load_t = t
        self.io_load_bytes = bytes
        self.io_load_bytes_total += bytes
        self.io_load_t_total += t

    def new_io_tsave(self, t):
        self.io_save_t = t
        self.io_save_t_total += t

    def new_tprocessing(self, t):
        self.tprocessing = t
        self.tprocessing_total += t

    def finished_batch(self, nfiles):
        self.nfiles_processed += nfiles

    @property
    def nfiles_remaining(self):
        return self.nfiles_total - self.nfiles_processed

    @property
    def time_remaining(self):
        proc_time = time() - self.time_start
        s = self.nfiles_remaining*(proc_time/(self.nfiles_processed or 1))
        return timedelta(seconds=s)

    @property
    def io_load_speed(self):
        return (self.io_load_bytes / 1e6) / \
            (self.io_load_t or 1.)

    @property
    def io_load_speed_avg(self):
        return (self.io_load_bytes_total / 1e6) / \
            (self.io_load_t_total or 1.)


def split(trace, time):
    if not trace.tmin < time < trace.tmax:
        return (trace, )

    return (trace.chop(trace.tmin, time, inplace=False),
            trace.chop(time, trace.tmax, inplace=False, include_last=True))


def tdms_guess_time(path):
    fn = op.splitext(op.basename(path))[0]
    time_str = fn[-19:]
    return datetime.strptime(time_str, '%Y%m%d_%H%M%S.%f').timestamp()


def detect_files(path):
    if op.isfile(path):
        return [path]

    files = []
    for ipath in iglob(path):
        if op.isfile(ipath):
            files.append(ipath)

        elif op.isdir(ipath):
            logger.debug('Scanning folder %s', ipath)
            files.extend(
                [op.join(root, f)
                 for root, dirs, files in os.walk(ipath)
                 for f in files])

    return files


def process_data(args):
    trace, deltat, tmin, tmax = args

    tmin = tmin or trace.tmin
    tmax = tmax or trace.tmax

    if tmin > trace.tmax or tmax < trace.tmin:
        return None

    trace.downsample_to(deltat)
    trace.chop(
        tmin=tmin,
        tmax=tmax,
        inplace=True)

    trace.ydata = trace.ydata.astype(num.int32)
    return trace


def load_idas(fn):
    logger.debug(
        'Loading %s (thread: %s)', op.basename(fn), threading.get_ident())
    return io.load(fn, format='tdms_idas')


class iDASConvert(object):

    def __init__(
            self, paths, outpath,
            downsample_to=200., record_length=4096,
            new_network_code='ID', new_channel_code='HSF',
            channel_selection=None,
            tmin=None, tmax=None,

            nthreads=8, batch_size=1, plugins=[]):

        logger.info('Detecting files...')
        files = []
        for path in paths:
            files.extend(detect_files(path))

        def in_timeframe(fn):
            try:
                fn_tmin = tdms_guess_time(fn)
            except ValueError:
                return True

            if tmax is not None and fn_tmin > tmax:
                return False
            if tmin is not None and (fn_tmin + 60.) < tmin:
                return False
            return True

        if tmin is not None or tmax is not None:
            nfiles_before = len(files)
            files = list(filter(in_timeframe, files))
            logger.info('Filtered %d files', nfiles_before - len(files))

        if not files:
            raise OSError('No files selected for conversion.')

        logger.info('Sorting %d files', len(files))

        self.files = sorted(files, key=lambda f: op.basename(f))
        self.files_all = self.files.copy()
        self.nfiles = len(self.files_all)

        self.stats = Stats(nfiles_total=self.nfiles)

        self.outpath = outpath
        self.channel_selection = None if not channel_selection \
            else re.compile(channel_selection)
        self.new_network_code = new_network_code
        self.new_channel_code = new_channel_code

        self.downsample_to = downsample_to
        self.record_length = record_length

        self.tmin = tmin
        self.tmax = tmax

        self.nthreads = nthreads
        self.batch_size = batch_size

        self.before_batch_load = Signal(self)
        self.finished_batch = Signal(self)
        self.before_file_read = Signal(self)
        self.new_traces_converted = Signal(self)

        self.plugins = plugins
        for plugin in self.plugins:
            logger.info('Activating plugin %s', plugin.__class__.__name__)
            plugin.set_parent(self)

        self._trs_prev = []
        self._tmax_prev = None

    @property
    def nfiles_left(self):
        return len(self.nfiles)

    def start(self):
        logger.info('Starting conversion of %d files', self.nfiles)
        stats = self.stats
        t_start = time()

        files = self.files
        trs_overlap = None
        tmax_prev = None

        while self.files:
            load_files = []
            self.before_batch_load.dispatch(self.files)

            while files and len(load_files) < self.batch_size:
                fn = self.files.pop(0)
                self.before_file_read.dispatch(fn)

                load_files.append(fn)

                with open(fn, 'rb') as f:
                    if not detect_tdms(f.read(512)):
                        logger.warning('Not a tdms file %s', fn)
                        continue

            batch_tmin = tmax_prev
            if self.tmin is not None:
                batch_tmin = max(self.tmin, batch_tmin or -1.)

            traces, batch_tmin, batch_tmax, trs_overlap = self.convert_files(
                load_files,
                tmin=batch_tmin,
                tmax=self.tmax,
                overlap=trs_overlap)

            self.finished_batch.dispatch(load_files)
            stats.finished_batch(len(load_files))

            tmax_prev = batch_tmax + 1./self.downsample_to

            logger.info(
                'Processed {s.nfiles_processed}/{s.nfiles_total} files'
                ' (DS: {s.tprocessing:.2f},'
                ' IO: {s.io_load_t:.2f}/{s.io_save_t:.2f}'
                ' [load: {s.io_load_speed:.2f} MB/s]).'
                ' {s.time_remaining} remaining'.format(s=stats))

        logger.info('Finished. Processed %d files in %.2f s',
                    stats.nfiles_processed, time() - t_start)

    def convert_files(self, files, tmin=None, tmax=None, overlap=False):
        nfiles = len(files)
        stats = self.stats
        max_workers = nfiles if nfiles < self.batch_size else self.batch_size

        t_start = time()
        nbytes = sum(op.getsize(fn) for fn in files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            trs_all = list(chain(*executor.map(load_idas, files)))

        if self.channel_selection:
            trs_all = [tr for tr in trs_all
                       if self.channel_selection.match(tr.station)]
        if not trs_all:
            raise TypeError('Did not load any traces!')

        stats.new_io_load(time() - t_start, nbytes)

        trs = sorted(trs_all + (overlap or []), key=lambda tr: tr.full_id)
        trs_degapped = trace.degapper(trs)
        if (overlap and tmin) and len(trs_degapped) != len(overlap):
            logger.warning('Gap detected at %s', tts(tmin))
            trs_degapped = trace.degapper(
                sorted(trs_all, key=lambda tr: tr.full_id))

        trs_overlap = self.get_traces_end(trs_degapped)

        t_start = time()
        with ThreadPoolExecutor(max_workers=self.nthreads) as executor:
            trs_ds = list(executor.map(
                process_data,
                zip(trs_degapped,
                    repeat(1./self.downsample_to),
                    repeat(tmin), repeat(tmax))))

        trs_ds = list(filter(lambda tr: tr is not None, trs_ds))

        if not trs_ds:
            return [], None, None, trs_overlap

        for tr in trs_degapped:
            tr.set_network(self.new_network_code)
            tr.set_channel(self.new_channel_code)

        batch_tmax = max(tr.tmax for tr in trs_ds)
        batch_tmin = min(tr.tmin for tr in trs_ds)

        stats.new_tprocessing(time() - t_start)

        self.new_traces_converted.dispatch(trs_ds)

        t_start = time()

        # Split traces at day break
        dt_min = datetime.fromtimestamp(batch_tmin)
        dt_max = datetime.fromtimestamp(batch_tmax)

        if dt_min.date() != dt_max.date():
            dt_split = datetime.combine(dt_max.date(), datetime.min.time())
            tsplit = dt_split.timestamp()
            trs_ds = list(chain(*(split(tr, tsplit) for tr in trs_ds)))

        io.save(
            trs_ds, self.outpath,
            format='mseed',
            record_length=self.record_length,
            append=True)
        stats.new_io_tsave(time() - t_start)

        return trs_ds, batch_tmin, batch_tmax, trs_overlap

    def get_traces_end(self, traces, overlap=1.):
        trs_chopped = []
        for tr in traces:
            trs_chopped.append(
                tr.chop(tr.tmax - overlap, tr.tmax, inplace=False))
        return trs_chopped


class iDASConvertConfig(Object):
    batch_size = Int.T(
        default=1,
        help='Number of parallel loaded TDMS files and processed at once.')
    nthreads = Int.T(
        default=8,
        help='Number of threads for processing data.')

    paths = List.T(
        Path.T(),
        default=[os.getcwd()],
        help='TDMS paths to convert.')
    outpath = Path.T(
        default='%(tmin_year)s%(tmin_month)s%(tmin_day)s/%(network)s.%(station)s_%(tmin_year)s%(tmin_month)s%(tmin_day)s.mseed',  # noqa
        help='Outfile in pyrocko.io.save format.')

    channel_selection = String.T(
        optional=True,
        help='Limit the conversion so these channels, regex allowed')
    new_network_code = String.T(
        default='ID',
        help='Network code for MiniSeeds.')
    new_channel_code = String.T(
        default='HSF')

    tmin = Timestamp.T(
        optional=True,
        help='Start time for the conversion')
    tmax = Timestamp.T(
        optional=True,
        help='End time for the conversion')

    downsample_to = Int.T(
        default=200.,
        help='Target sample rate for mseed.')
    record_length = Int.T(
        default=4096,
        help='MiniSeeds record length in bytes.')

    plugins = List.T(
        PluginConfig.T(),
        default=[p() for p in PLUGINS_AVAILABLE],
        help='Plugins for the converter')

    def get_converter(self):
        plugins = []
        for plugin_config in self.plugins:
            if plugin_config.enabled:
                plugins.append(plugin_config.get_plugin())

        return iDASConvert(
            self.paths, self.outpath,
            self.downsample_to, self.record_length,
            self.new_network_code, self.new_channel_code,
            self.channel_selection,
            self.tmin, self.tmax,
            self.nthreads, self.batch_size, plugins)
