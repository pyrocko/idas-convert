import os
import re
import logging
import threading
import queue
import tempfile
import yaml

from time import time, sleep
from glob import iglob
from datetime import timedelta, datetime
from itertools import repeat, chain
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as num

from pyrocko import io, trace
from pyrocko.io.tdms_idas import detect as detect_tdms
from pyrocko.util import tts
from pyrocko.gui.marker import Marker

from pyrocko.guts import Object, String, Int, List, Timestamp

from .plugin import PluginConfig, PLUGINS_AVAILABLE
from .meta import Path
from .utils import Signal, sizeof_fmt

guts_prefix = 'idas'

DEFAULT_NETWORK_CODE = 'ID'
DEFAULT_CHANNEL_CODE = 'HSF'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('idas_convert')

nptdms_logger = logging.getLogger('nptdms.base_segment')
nptdms_logger.setLevel(logging.ERROR)
op = os.path

day = 3600. * 24


def split_trace(tr, time):
    try:
        return (tr.chop(tr.tmin, time, inplace=False),
                tr.chop(time, tr.tmax, inplace=False, include_last=True))
    except trace.NoData:
        return (tr, )


def tdms_guess_time(path):
    fn = op.splitext(op.basename(path))[0]
    time_str = fn[-19:] + '+0000'
    return datetime.strptime(time_str, '%Y%m%d_%H%M%S.%f%z').timestamp()


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


def same_markers(m1, m2):
    diff_tmin = None
    diff_tmax = None

    if m1.tmin and m2.tmin and \
            round(m1.tmin, 3) == round(m2.tmin, 3):
        return True

    if m1.tmax and m2.tmax and \
            round(m1.tmax, 3) == round(m2.tmax, 3):
        return True

    return False

class MarkerLog(object):


    def __init__(
            self,
            network_code=DEFAULT_NETWORK_CODE,
            channel_code=DEFAULT_CHANNEL_CODE):

        self.marker_file = None
        self.network_code = network_code
        self.channel_code = channel_code

        self.markers = []

    def set_marker_file(self, marker_file):
        if marker_file and op.exists(marker_file):
            logger.info('Loading existing markers from %s', marker_file)
            self.markers = Marker.load_markers(marker_file)
        self.marker_file = marker_file

    @property
    def nmarkers(self):
        return len(markers)

    def new_marker(self, tmin, tmax=None, kind=0):

        marker = Marker(
            nslc_ids=[(self.network_code, '*', '', self.channel_code)],
            tmin=tmin,
            tmax=tmax,
            kind=kind
        )

        for m in self.markers:
            if same_markers(m, marker):
                return

        self.markers.append(marker)
        self.save_markers(marker)

    def save_markers(self, marker):
        if not self.marker_file:
            return
        logger.debug('Writing out marker')
        Marker.save_markers(self.markers, self.marker_file)


class LoadTDMSThread(threading.Thread):

    def __init__(self, fn_queue, traces_queue, marker_log,
                 check_zero_value=True):
        super().__init__()
        self.fn_queue = fn_queue
        self.traces_queue = traces_queue
        self.bytes_loaded = 0
        self.time_loading = 0.

        self.stop = threading.Event()

        self.check_zero_value = check_zero_value
        self.marker_log = marker_log

    @property
    def bytes_input_rate(self):
        return self.bytes_loaded / (self.time_loading or 1.)

    def qc_zero_value_trace(self, traces):
        if not traces:
            return

        tr = traces[0]
        if tr.ydata[-1] != 0.:
            return

        tmax = tr.tmax
        tmin = tmax
        nsamples = 0
        for sample in tr.ydata[::-1]:
            if sample != 0.:
                break
            tmin -= tr.deltat
            nsamples += 1

        if nsamples == 1:
            return

        logger.warning(
            'Zero value gap detected %s - %s (%d samples)',
            tts(tmin), tts(tmax), nsamples)

        self.marker_log.new_marker(tmin, tmax, kind=0)

    def run(self):
        logger.info('Starting loading thread %s', self.name)
        while not self.stop.is_set():
            try:
                ifn, fn = self.fn_queue.get(timeout=1.)
            except queue.Empty:
                continue
            logger.debug(
                'Loading %s (thread: %s)',
                op.basename(fn), threading.get_ident())
            t_start = time()
            traces = io.load(fn, format='tdms_idas')
            fsize = op.getsize(fn)
            self.time_loading += time() - t_start
            self.bytes_loaded += fsize

            if self.check_zero_value:
                self.qc_zero_value_trace(traces)

            self.traces_queue.put((ifn, traces, fn, fsize))

            self.fn_queue.task_done()

    def stop(self):
        self._stop = True


def process_data(args):
    trace, deltat, tmin, tmax = args

    tmin = tmin or trace.tmin
    tmax = tmax or trace.tmax

    ds = datetime.utcfromtimestamp

    if tmin > trace.tmax or tmax < trace.tmin:
        print('limits', ds(tmin), ds(tmax))
        print('trace', ds(trace.tmin), ds(trace.tmax))
        return None

    trace.downsample_to(deltat)
    trace.chop(
        tmin=tmin,
        tmax=tmax,
        inplace=True)

    trace.ydata = trace.ydata.astype(num.int32)
    return trace


def get_traces_end(traces, overlap=1.):
    trs_chopped = []
    for tr in traces:
        try:
            trs_chopped.append(
                tr.chop(tr.tmax - overlap, tr.tmax, inplace=False))
        except trace.NoData:
            return []

    return trs_chopped


class ProcessingThread(threading.Thread):

    def __init__(self, processing_queue, out_queue,
                 downsample_to=200.,
                 new_network_code=DEFAULT_NETWORK_CODE,
                 new_channel_code=DEFAULT_CHANNEL_CODE,
                 channel_selection=None,
                 tmin=None, tmax=None, nthreads=12, batch_size=12):
        super().__init__()

        self.in_traces = processing_queue
        self.out_traces = out_queue
        self.downsample_to = downsample_to

        self.channel_selection = channel_selection
        self.new_network_code = new_network_code
        self.new_channel_code = new_channel_code

        self.tmin = tmin
        self.tmax = tmax

        self.deltat = 1./self.downsample_to
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=nthreads)

        self.ifn = 0
        self.nfiles = 0
        self.bytes_in = 0
        self.time_processing = 0.

        self.stop = threading.Event()

    @property
    def processing_file_rate(self):
        return (self.ifn + 1) / (self.time_processing or 1.)

    @property
    def processing_rate(self):
        return self.bytes_in / (self.time_processing or 1.)

    def get_new_traces(self):
        new_trs = []
        fns = []
        while len(new_trs) < self.batch_size:
            try:
                this_ifn, trs, fn, size = self.in_traces.get(timeout=10.)
                if this_ifn != self.ifn:
                    self.in_traces.put((this_ifn, trs, fn, size))
                    self.in_traces.task_done()
                    sleep(.1)
                    continue

                new_trs.append(trs)
                fns.append(fn)
                self.ifn += 1
                self.bytes_in += size
            except queue.Empty:
                break

        return new_trs, fns

    def run(self):
        logger.info('Starting processing thread')

        trs_overlap = []
        batch_tmin = self.tmin

        while not self.stop.is_set():
            new_trs, new_fns = self.get_new_traces()
            nnew_trs = len(new_trs)
            new_trs = list(chain(*new_trs))

            if self.channel_selection:
                new_trs = [tr for tr in new_trs
                           if self.channel_selection.match(tr.station)]
            if self.stop.is_set() and not new_trs:
                break
            elif not new_trs:
                continue

            logger.debug('Start processing %d trace groups', nnew_trs)
            t_start = time()

            trs_degapped = trace.degapper(
                sorted(new_trs + trs_overlap, key=lambda tr: tr.full_id))
            if trs_overlap and batch_tmin \
                    and len(trs_degapped) != len(trs_overlap):
                logger.warning('Gap detected at %s', tts(batch_tmin))
                trs_degapped = trace.degapper(
                    sorted(new_trs, key=lambda tr: tr.full_id))

            trs_overlap = get_traces_end(trs_degapped)

            trs_ds = list(self.executor.map(
                process_data,
                zip(trs_degapped,
                    repeat(1./self.downsample_to),
                    repeat(batch_tmin), repeat(self.tmax))))
            trs_ds = list(filter(lambda tr: tr is not None, trs_ds))
            if not trs_ds:
                logger.warning('Downsampled traces are empty')
                continue

            for tr in trs_ds:
                tr.set_network(self.new_network_code)
                tr.set_channel(self.new_channel_code)

            batch_tmax = max(tr.tmax for tr in trs_ds)
            batch_tmin = min(tr.tmin for tr in trs_ds)

            # Split traces at day break
            dt_min = datetime.utcfromtimestamp(batch_tmin)
            dt_max = datetime.utcfromtimestamp(batch_tmax)

            if dt_min.date() != dt_max.date():
                dt_split = datetime.combine(dt_max.date(), datetime.min.time())
                tsplit = dt_split.timestamp()
                trs_ds = list(chain(*(split_trace(tr, tsplit) for tr in trs_ds)))

            batch_tmax += self.deltat
            batch_tmin = batch_tmax
            self.out_traces.put((batch_tmax, trs_ds, new_fns))

            for _ in range(nnew_trs):
                self.in_traces.task_done()

            self.time_per_batch = time() - t_start
            self.time_processing += self.time_per_batch

            logger.info(
                'Processed %d groups in %.2f s', nnew_trs, self.time_per_batch)

        self.executor.shutdown()
        logger.info('Shutting down processing thread')


class SaveMSeedThread(threading.Thread):

    def __init__(self, in_queue, outpath, marker_log,
                 record_length=4096, steim=2, checkpt_file=None):
        super().__init__()
        assert steim in (1, 2)
        assert record_length in io.mseed.VALID_RECORD_LENGTHS

        self.queue = in_queue
        self.outpath = outpath
        self.marker_log = marker_log
        self.record_length = record_length
        self.steim = steim
        self.checkpt_file = checkpt_file

        self.touched_files = set()
        self.tmax = 0.

        self.processed_files = queue.Queue()

        self.out_files = {}

    def set_checkpt_file(self, path):
        self.checkpt_file = path

    def get_tmax(self):
        return self.tmax

    @property
    def bytes_written(self):
        return sum(s for s in self.out_files.values())

    @property
    def nfiles_written(self):
        return len(self.out_files)

    def run(self):
        logger.info('Starting MiniSeed saving thread')
        while True:
            tmax, traces, fns = self.queue.get()
            if traces is False or not traces:
                logger.debug('Shutting down saving thread')
                self.queue.task_done()
                return

            t_start = time()

            filenames = set(tr.fill_template(self.outpath) for tr in traces)
            for tr, fn in zip(traces, filenames):
                if op.exists(fn) and fn not in self.touched_files:
                    tr_existing = io.load(fn, getdata=False)[0]
                    if tr_existing.tmax > tr.tmax:
                        logger.warn('file %s exists! Skipping trace', fn)
                        traces.remove(tr)

            self.touched_files.update(filenames)

            out_files = io.save(
                traces, self.outpath,
                format='mseed',
                record_length=self.record_length,
                steim=self.steim,
                append=True)

            if self.checkpt_file is not None:
                    with open(self.checkpt_file, 'w') as f:
                        f.write(str(tmax))

            for fn in out_files:
                self.out_files[fn] = op.getsize(fn)

            rtr = traces[0]
            for dn in set(op.dirname(fn) for fn in out_files):
                fn_meta = op.join(dn, 'metadata.yml')
                if op.exists(fn_meta):
                    continue
                with open(fn_meta, 'w') as f:
                    logger.info('Writing out metadata to %s', fn_meta)
                    yaml.dump(rtr.meta, f)

            self.tmax = tmax

            self.processed_files.put(fns)
            self.queue.task_done()
            logger.debug(
                'Saved %d traces from queue in %.1f s',
                len(traces), time() - t_start)


def load_idas_thread(fn):
    logger.debug(
        'Loading %s (thread: %s)', op.basename(fn), threading.get_ident())
    return io.load(fn, format='tdms_idas')


class iDASConvert(object):

    def __init__(
            self, paths, outpath,
            downsample_to=200., record_length=4096, steim=2,
            new_network_code=DEFAULT_NETWORK_CODE,
            new_channel_code=DEFAULT_CHANNEL_CODE,
            channel_selection=None, tmin=None, tmax=None,
            nthreads_loading=8, nthreads_processing=24,
            queue_size=32, processing_batch_size=8, plugins=[]):

        if tmin is not None and tmax is not None:
            assert tmin < tmax
        logger.info('Detecting files...')
        files = []
        for path in paths:
            files.extend(detect_files(path))

        def in_timeframe(fn):
            try:
                fn_tmin = tdms_guess_time(fn)
            except ValueError:
                logger.warning('Could not guess time for %s', op.filename(fn))
                return True

            if tmax is not None and fn_tmin > tmax:
                return False
            if tmin is not None and (fn_tmin + 60.) < tmin:
                return False
            return True

        if tmin is not None or tmax is not None:
            nfiles_before = len(files)
            files = list(filter(in_timeframe, files))
            logger.info(
                'Filtered %d/%d files',
                nfiles_before - len(files), nfiles_before)

        if not files:
            raise OSError('No files selected for conversion.')

        logger.info('Sorting %d files', len(files))

        self.files = sorted(files, key=lambda f: op.basename(f))
        self.bytes_total = sum([op.getsize(f) for f in self.files])
        logger.info('Got %s of data', sizeof_fmt(self.bytes_total))

        self.files_all = self.files.copy()
        self.nfiles = len(self.files_all)
        self.nfiles_processed = 0
        self.t_start = 0.
        self.processing_batch_size = processing_batch_size

        self.outpath = outpath
        channel_selection = None if not channel_selection \
            else re.compile(channel_selection)

        self.before_batch_load = Signal(self)
        self.before_file_read = Signal(self)
        self.finished_batch = Signal(self)
        self.finished = Signal(self)

        self.load_fn_queue = queue.PriorityQueue(maxsize=queue_size)
        self.processing_queue = queue.PriorityQueue(maxsize=queue_size)
        self.save_queue = queue.PriorityQueue()

        self.plugins = plugins
        for plugin in self.plugins:
            logger.info('Activating plugin %s', plugin.__class__.__name__)
            plugin.set_parent(self)

        self.marker_log = MarkerLog()

        # Starting worker threads
        self.load_threads = []
        for ithread in range(nthreads_loading):
            thread = LoadTDMSThread(
                self.load_fn_queue, self.processing_queue,
                self.marker_log)
            thread.name = 'LoadTDMS-%02d' % ithread
            thread.start()
            self.load_threads.append(thread)

        self.processing_thread = ProcessingThread(
            self.processing_queue, self.save_queue,
            downsample_to, new_network_code, new_channel_code,
            channel_selection, tmin, tmax,
            nthreads_processing, processing_batch_size)
        self.processing_thread.start()

        self.save_thread = SaveMSeedThread(
            self.save_queue, self.outpath, self.marker_log,
            record_length, steim)
        self.save_thread.start()

    @property
    def load_queue_size(self):
        return self.load_fn_queue.qsize()

    @property
    def process_queue_size(self):
        return self.processing_queue.qsize()

    @property
    def save_queue_size(self):
        return self.save_queue.qsize()

    @property
    def nfiles_remaining(self):
        return len(self.files)

    @property
    def bytes_loaded(self):
        return sum(thr.bytes_loaded for thr in self.load_threads)

    @property
    def bytes_remaining(self):
        return self.bytes_total - self.bytes_loaded

    @property
    def bytes_processing_rate(self):
        return self.bytes_loaded / self.duration

    @property
    def bytes_input_rate(self):
        return sum(t.bytes_input_rate for t in self.load_threads)

    @property
    def bytes_written(self):
        return self.save_thread.bytes_written

    @property
    def processing_rate(self):
        return self.processing_thread.processing_rate

    @property
    def duration(self):
        return time() - self.t_start

    @property
    def time_remaining(self):
        if not self.bytes_processing_rate:
            return timedelta(seconds=0.)
        s = self.bytes_remaining / self.bytes_processing_rate
        return timedelta(seconds=s)

    @property
    def time_remaining_str(self):
        return str(self.time_remaining)[:-7]

    @property
    def time_head(self):
        return self.save_thread.get_tmax()

    def start(self, checkpt_file=None, log_file=None, marker_file=None):
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s'))
            logger.addHandler(fh)

        if marker_file:
            self.marker_log.set_marker_file(marker_file)

        logger.info('Starting conversion of %d files', self.nfiles)
        ifn = 0
        self.save_thread.set_checkpt_file(checkpt_file)

        self.before_batch_load.dispatch(self.files)
        self.t_start = time()

        while self.files:
            if ifn % self.processing_batch_size == 0:
                self.before_batch_load.dispatch(self.files)

            fn = self.files.pop(0)
            self.before_file_read.dispatch(fn)

            with open(fn, 'rb') as f:
                if not detect_tdms(f.read(512)):
                    logger.warning('Not a tdms file %s', fn)
                    continue
            self.load_fn_queue.put((ifn, fn))

            self.check_processed()
            ifn += 1

        logger.debug('Joining load queue')
        self.load_fn_queue.join()
        for thread in self.load_threads:
            thread.stop.set()
        logger.debug('Joined load queue')

        logger.debug('Joining processing queue')
        self.processing_thread.stop.set()
        self.processing_queue.join()
        logger.debug('Joined processing queue')

        # Ensure it is the last element
        self.save_queue.put((time(), False, None))
        self.save_queue.join()
        logger.debug('Joining save trace queue')

        self.check_processed()
        self.finished.dispatch()

    def check_processed(self):
        proc_fns_queue = self.save_thread.processed_files
        if proc_fns_queue.empty():
            return

        finished_fns = []
        while not proc_fns_queue.empty():
            finished_fns.extend(proc_fns_queue.get_nowait())

        self.nfiles_processed += len(finished_fns)
        self.finished_batch.dispatch(finished_fns)

        logger.info(self.get_status())

    def get_status(self):
        s = self
        return (
            f'Processed {100*s.nfiles_processed/s.nfiles:.1f}%.'
            f' {sizeof_fmt(s.bytes_loaded)}/{sizeof_fmt(s.bytes_total)}'
            f' @ {s.bytes_processing_rate/1e6:.1f} MB/s,'
            f' in {s.bytes_input_rate/1e6:.1f} MB/s,'
            f' proc {s.processing_rate/1e6:.1f} MB/s.'
            f' {sizeof_fmt(s.bytes_written)} written.'
            f' Head is at {tts(s.time_head)}.'
            f' Queues'
            f' L:{s.load_queue_size}'
            f' P:{s.process_queue_size}'
            f' S:{s.save_queue_size}.'
            f' Estimated time remaining {s.time_remaining_str}.')


class iDASConvertConfig(Object):
    nthreads_loading = Int.T(
        default=1,
        help='Number of parallel loaded TDMS files and processed at once.')
    nthreads_processing = Int.T(
        default=8,
        help='Number of threads for processing data.')

    queue_size = Int.T(
        default=32,
        help='Size of the queue holding loaded traces.')
    processing_batch_size = Int.T(
        default=8,
        help='Number of traces processed at once.')

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
        default=DEFAULT_NETWORK_CODE,
        help='Network code for MiniSeeds.')
    new_channel_code = String.T(
        default=DEFAULT_CHANNEL_CODE,
        help='Channel code for MiniSeeds.')

    tmin = Timestamp.T(
        optional=True,
        help='Start time for the conversion.')
    tmax = Timestamp.T(
        optional=True,
        help='End time for the conversion.')

    downsample_to = Int.T(
        default=200.,
        help='Target sample rate for mseed [Hz].')
    record_length = Int.T(
        default=4096,
        help='MiniSeeds record length in bytes. Default 4096 bytes.')
    steim = Int.T(
        default=2,
        help='Which STEIM compression to use. Default 2.')

    plugins = List.T(
        PluginConfig.T(),
        default=[p() for p in PLUGINS_AVAILABLE],
        help='Plugins for the converter')

    def get_converter(self):
        plugins = []
        for plugin_config in self.plugins:
            if plugin_config.enabled:
                plugins.append(plugin_config.get_plugin())

        if self.tmin is not None and self.tmax is None:
            raise AttributeError('tmax is not set')

        if self.tmax is not None and self.tmin is None:
            raise AttributeError('tmin is not set')

        return iDASConvert(
            self.paths, self.outpath,
            self.downsample_to, self.record_length, self.steim,
            self.new_network_code, self.new_channel_code,
            self.channel_selection,
            self.tmin, self.tmax,
            self.nthreads_loading, self.nthreads_processing,
            self.queue_size, self.processing_batch_size,
            plugins)
