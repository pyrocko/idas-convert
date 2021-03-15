import logging
import argparse
import logging
import sys
import os
from pathlib import Path

from pyrocko.guts import load
from pyrocko.util import tts

from .idas_convert import iDASConvertConfig

op = os.path
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        'Convert and downsample iDAS TDMS to MiniSeed for archiving')

    parser.add_argument(
        'config', type=str,
        help='Config file path or \'dump_config\'')

    parser.add_argument(
        '--resume', default=False, action='store_true',
        help='Resume from an existing checkpoint file')

    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Verbosity, add mutliple to increase verbosity')

    args = parser.parse_args()

    log_level = logging.INFO - args.verbose * 10
    log_level = log_level if log_level > logging.DEBUG else logging.DEBUG
    logging.getLogger().setLevel(log_level)

    if args.config == 'dump_config':
        print(iDASConvertConfig())
        sys.exit(0)

    config = load(filename=args.config)
    fn_config = Path(args.config)

    config_fn = op.splitext(fn_config.name)[0]
    checkpt_file = op.abspath(fn_config.parent / ('.progress-' + config_fn))
    log_file = op.abspath(fn_config.parent / (config_fn + '.log'))
    marker_file = op.abspath(fn_config.parent / (config_fn + '-marker.txt'))

    if op.exists(checkpt_file) and args.resume:
        with open(checkpt_file, 'r') as f:
            config.tmin = float(f.read())
        logger.info('Using checkpoint file %s, Resuming from %s',
                    checkpt_file, tts(config.tmin)[:19])
    elif op.exists(checkpt_file) and not args.resume:
        raise EnvironmentError(
            'Found existing checkpoint file %s. Use --resume to continue an'
            ' aborted conversion or delete the file.'
            % op.relpath(checkpt_file))
    elif not op.exists(checkpt_file) and args.resume:
        raise EnvironmentError(
            'Could not find the checkpoint file %s. '
            'Nothing to resume, aborting'
            % op.relpath(checkpt_file))

    converter = config.get_converter()

    try:
        converter.start(checkpt_file, log_file, marker_file)
    except Exception as e:
        logger.exception(e)
        raise e

    os.remove(checkpt_file)


if __name__ == '__main__':
    main()
