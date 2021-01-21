import os
import time
import logging
import subprocess

from pyrocko.guts import Bool, Float

from .plugin import Plugin, PluginConfig, register_plugin
from .meta import Path, DataSize
from .utils import sizeof_fmt

logger = logging.getLogger(__name__)
op = os.path


class GFZTapes(Plugin):

    def __init__(self, bytes_stage, waterlevel,
                 path_tapes_mount, path_tapes_prefix,
                 release_files=True, wait_warning_interval=600.):

        self.bytes_stage = bytes_stage
        self.waterlevel = waterlevel

        self.path_tapes_mount = path_tapes_mount
        self.path_tapes_prefix = path_tapes_prefix

        self.release_files = release_files
        self.wait_warning_interval = wait_warning_interval

        self.requested_files = set()
        self._initial_stage = False

    def set_parent(self, parent):
        super().set_parent(parent)

        parent.before_file_read.register(self.wait)
        parent.before_batch_load.register(self.stage)
        parent.finished_batch.register(self.release)

    def _stage_files(self, files):
        if isinstance(files, str):
            files = (files, )

        proc = subprocess.run(
            ['stage', '-D', self.path_tapes_prefix, *files])
        # logger.debug('ran `%s`', ' '.join(proc.args))
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            logger.exception(e)
        return proc

    def _release_files(self, files):
        if isinstance(files, str):
            files = (files, )

        proc = subprocess.run(
            ['release', '-D', self.path_tapes_prefix, *files])
        # logger.debug('ran `%s`', proc.args)
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            logger.exception(e)
        return proc

    def wait(self, path):
        stat = os.stat(path)
        if stat.st_blocks > 0:
            return

        fn = op.basename(path)
        logger.warning('Waiting for file %s', fn)
        fn_wait = time.time()

        fn_tape = op.relpath(op.abspath(path), self.path_tapes_mount)
        logger.debug('Re-requesting file %s', fn_tape)
        self._stage_files(fn_tape)
        warning_interval = self.wait_warning_interval

        while stat.st_blocks == 0:
            stat = os.stat(path)
            duration = time.time() - fn_wait
            if duration > warning_interval:
                logger.warning('Waiting since %.1f s', duration)
                warning_interval += self.wait_warning_interval

            time.sleep(1.)

        logger.info('File available, waited %.2f s', duration)

    def stage(self, remaining_files):
        nbytes = 0
        nbytes_unstaged = 0

        unstaged_files = []

        for fn in remaining_files:
            stats = os.stat(fn)
            nbytes += stats.st_size
            if stats.st_blocks == 0:
                nbytes_unstaged += stats.st_size
                unstaged_files.append(op.abspath(fn))

            if nbytes >= self.bytes_stage:
                break

        nbytes_staged = nbytes - nbytes_unstaged

        if nbytes_staged / nbytes < self.waterlevel \
                or not self._initial_stage:
            fns_tape = set(op.relpath(fn, self.path_tapes_mount)
                           for fn in unstaged_files)

            logger.debug('staging %d files', len(fns_tape))
            self._stage_files(fns_tape)

            self.requested_files |= fns_tape
            logger.info('requested %s', sizeof_fmt(nbytes_unstaged))
            self._initial_stage = True

        else:
            logger.info(
                'staging waterlevel ok (%.1f%%, %s)',
                (nbytes_staged / nbytes) * 100., sizeof_fmt(nbytes_staged))

    def release(self, files):
        if not self.release_files:
            return

        fns_tape = [op.relpath(fn, self.path_tapes_mount) for fn in files]
        logger.debug('Releasing %d files', len(fns_tape))
        self._release_files(fns_tape)


class GFZTapesConfig(PluginConfig):
    bytes_stage = DataSize.T(
        default='1T',
        help='Amount of data to stage before conversion in bytes. Suffix with'
             ' M, G, T, P. e.g. 4T.')
    waterlevel = Float.T(
        default=.6,
        help='Waterlevel before data is staged from tapes, in percent [0-1].')
    wait_warning_interval = Float.T(
        default=600.,
        help='Log when waiting for files at this interval [s].')
    release_files = Bool.T(
        default=True,
        help='Release files after reading.')
    path_tapes_mount = Path.T(
        default='/projects/ether/',
        help='Where the archive is mounted')
    path_tapes_prefix = Path.T(
        default='/archive_FO1/RAW/',
        help='Prefix for stage -D <prefix>.')

    def get_plugin(self):
        assert 0. < self.waterlevel < 1., 'bad waterlevel'
        return GFZTapes(
            self.bytes_stage, self.waterlevel,
            self.path_tapes_mount, self.path_tapes_prefix,
            self.release_files, self.wait_warning_interval)


register_plugin(GFZTapesConfig)
