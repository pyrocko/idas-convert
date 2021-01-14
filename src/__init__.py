import time
import logging
import numpy as num
import os
import subprocess

from pyrocko import io, trace
from pyrocko.io.tdms_idas import detect as detect_tdms
from pyrocko.util import tts
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('idas_convert')
op = os.path

PADDING = .5
NTHREADS = 8
BATCH_SIZE = 1


def detect_files(path):
    if op.isfile(path):
        return [path]

    return [op.join(root, f)
            for root, dirs, files in os.walk(path)
            for f in files]


def ensure_staged_files(files, stage_bytes, waterlevel=None):
    nbytes = 0
    nbytes_unstaged = 0

    if waterlevel is None:
        waterlevel = stage_bytes / 2

    unstaged_files = []
    for fn in files:
        stats = os.stat(fn)
        size = stats.st_size
        nbytes += size
        if stats.st_blocks == 0:
            nbytes_unstaged += size
            unstaged_files.append(fn)

        if nbytes >= stage_bytes:
            break

    if nbytes_unstaged > waterlevel:
        logger.info('staging additional %d bytes', nbytes)
        for fn in unstaged_files:
            fn_tapes = fn.strip('/projects/ether/')
            logger.debug('staging file %s', fn_tapes)
            subprocess.call(
                args=['stage', '-D', '/archive_FO1/RAW', fn_tapes],
                check=True)
        logger.info('staginged %d bytes', nbytes)

    else:
        logger.info('staging waterlevel ok')


def downsample_data(args):
    trace, deltat, tmin_limit = args
    trace.downsample_to(deltat)

    if tmin_limit \
            and tmin_limit > trace.tmin \
            and tmin_limit < trace.tmax:
        trace.chop(
            tmin=tmin_limit,
            tmax=trace.tmax,
            inplace=True)

    trace.ydata = trace.ydata.astype(num.int32)
    # tmean = (trace.tmax - trace.tmin) / 2
    # mean = num.mean(trace.ydata)
    # max = num.max(trace.ydata)
    # std = num.std(trace.ydata)
    return trace
    # return trace, tmean, mean, max, std


def load_idas(fn):
    logger.info('loading %s', op.basename(fn))
    return io.load(fn, format='tdms_idas')


def convert_tdsm(
        files, outpath,
        downsample_to=1./200, network='ID', record_length=4096,
        stage_ahead=None,
        nthreads=NTHREADS, batch_size=BATCH_SIZE):
    nfiles = len(files)

    tmin_limit = None
    trs_prev = []
    ifn = 0
    while files:
        t0 = time.time()
        trs_new = []
        this_files = []

        while files and len(this_files) < batch_size:
            fn = files.pop(0)
            ifn += 1

            stat = os.stat(fn)
            fn_wait = False

            if stat.st_blocks == 0:
                logger.warning('waiting for file %s', fn)
                fn_wait = time.time()

                while stat.st_blocks == 0:
                    stat = os.stat(fn)
                    time.sleep(1.)

                logger.warning('file %s available. Waited %.2f s',
                               time.time() - fn_wait)

            with open(fn, 'rb') as f:
                if not detect_tdms(f.read(512)):
                    logger.warning('not a tdms file %s', fn)
                    continue

            this_files.append(fn)

        if stage_ahead:
            ensure_staged_files(files, stage_ahead, waterlevel=stage_ahead/2)


        with ThreadPoolExecutor(max_workers=len(this_files)) as executor:
            trs_loaded = list(executor.map(load_idas, this_files))

        tmins = []
        for trs in trs_loaded:
            if not trs:
                logger.warning('loaded empty traces')
                continue
            tmins.append(trs[0].tmin)
            trs_new.extend(trs)

        trs_latest = trs_loaded[num.argmax(tmins)]

        t_load = time.time() - t0
        if not trs_new:
            logger.error('empty input data')
            continue

        if trs_prev:
            deltat = trs_new[0].deltat
            prev_tmax = [tr.tmax for tr in trs_prev]
            if max(tmins) - min(prev_tmax) > 2*deltat:
                logger.warning('gap detected at %s', tts(min(prev_tmax)))
                trs_prev = []

        trs = sorted(trs_prev + trs_new, key=lambda tr: tr.full_id)

        trs_prev = []
        for tr in trs_latest:
            try:
                trs_prev.append(tr.chop(
                    tmin=tr.tmax - PADDING,
                    tmax=tr.tmax,
                    inplace=False))
            except trace.NoData:
                pass

        t = time.time()
        trs = trace.degapper(trs)

        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            trs_ds = list(executor.map(
                downsample_data,
                zip(trs,
                    repeat(downsample_to),
                    repeat(tmin_limit))))
        tmin_limit = max([tr.tmax + tr.deltat for tr in trs_ds])
        t_ds = time.time() - t

        for tr in trs_ds:
            tr.set_network(network)

        t = time.time()
        io.save(
            trs_ds, outpath,
            format='mseed',
            record_length=record_length,
            append=True)
        t_save = time.time() - t

        elapsed = time.time() - t0
        files_left = len(files)
        logger.info(
            'processed %d/%d files: in %.2f s '
            '(FIR: %.2f, IO: %.2f/%.2f, remaining %.2f s)',
            ifn, nfiles, elapsed, t_ds, t_load, t_save,
            elapsed * (files_left / batch_size))


def main():
    import argparse

    def data_size(size):
        mag = dict(k=3, M=6, G=9, T=12, P=15)
        if size is None:
            return None

        size = size.strip()
        v = int(size.strip(''.join(mag.keys())))
        if not v:
            return None

        for suffix, m in mag.items():
            if size.endswith(suffix):
                return v*10**m
        raise ValueError('cannot interpret size %s' % size)

    parser = argparse.ArgumentParser(
        'Convert and downsample iDAS TDMS to MiniSeed for archiving')
    parser.add_argument(
        'paths', type=str, nargs='+',
        help='TDMS paths to convert.')
    parser.add_argument(
        '--downsample', type=int,
        default=200,
        help='Target sample rate for mseed. Default: %(default)s')
    parser.add_argument(
        '--network',
        default='ID', type=lambda s: str(s)[:2],
        help='Network code for MiniSeeds. Default: %(default)s')
    parser.add_argument(
        '--record_length', type=int,
        default=4096,
        help='MiniSeeds record length. Default: %(default)s')
    parser.add_argument(
        '--threads', type=int,
        default=NTHREADS,
        help='Number of threads for processing data. Default: %(default)s')
    parser.add_argument(
        '--batchsize', type=int,
        default=BATCH_SIZE,
        help='Number of parallel loaded TDMS files and processed at once.'
             ' Default: %(default)s')
    parser.add_argument(
        '--outpath', type=str,
        default='%(tmin_year)s%(tmin_month)s%(tmin_day)s/%(network)s.%(station)s_%(tmin_year)s%(tmin_month)s%(tmin_day)s.mseed',  # noqa
        help='Outfile in pyrocko.io.save format. Default: %(default)s')

    parser.add_argument(
        '--stage-ahead', type=data_size, default=None,
        help='Amount of data to stage before conversion in bytes. Suffix with'
             ' M, G, T, P. e.g. 4T.')

    parser.add_argument(
        '--verbose', '-v', action='count',
        default=0,
        help='Verbosity, add mutliple to increase verbosity.')

    args = parser.parse_args()

    log_level = logging.INFO - args.verbose * 10
    log_level = log_level if log_level > logging.DEBUG else logging.DEBUG
    logging.basicConfig(level=log_level)

    paths = args.paths

    if isinstance(paths, str):
        paths = [paths]

    logger.info('detecting files...')
    files = []
    for path in paths:
        files.extend(detect_files(path))

    logger.info('sorting %d files', len(files))
    files = sorted(files, key=lambda f: op.basename(f))

    convert_tdsm(
        files, outpath=args.outpath, downsample_to=1./args.downsample,
        network=args.network, record_length=args.record_length,
        nthreads=args.threads, batch_size=args.batchsize)


if __name__ == '__main__':
    main()
