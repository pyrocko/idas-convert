
import argparse
import logging
import sys
import os.path as op

from pyrocko.guts import load

from .idas_convert import iDASConvertConfig


def main():
    parser = argparse.ArgumentParser(
        'Convert and downsample iDAS TDMS to MiniSeed for archiving')

    parser.add_argument(
        'config', type=str,
        help='Config file or \'dump_config\'.')

    parser.add_argument(
        '--verbose', '-v', action='count',
        default=0,
        help='Verbosity, add mutliple to increase verbosity.')

    args = parser.parse_args()

    log_level = logging.INFO - args.verbose * 10
    log_level = log_level if log_level > logging.DEBUG else logging.DEBUG
    logging.getLogger().setLevel(log_level)

    if args.config == 'dump_config':
        print(iDASConvertConfig())
        sys.exit(0)

    config = load(filename=args.config)
    converter = config.get_converter()
    converter.start()


if __name__ == '__main__':
    main()
