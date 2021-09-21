# iDAS TDMS Converter

Convert and downsample distribute acoustic sensing (DAS) data acquired by Silixa iDAS to seismological data formats. Main purpose is to quickly convert and downsample massive amounts of high-resolution TDMS data to MiniSeed.

To handle the massive amount of data generated by DAS interrogators, the conversion tool is leveraging parallel IO and signal-processing processing. On production systems a throughput of 200 MB/s can be archived while converting and downsampling (from 1 kHz to 200 Hz).

The signal processing routines are based on the Pyrocko, mature and well tested seismological framework.

## Installation

```sh
python3 setup.py install
```

## Documentation

Find the online documentation at https://pyrocko.org/idas_convert/docs/.

## Usage

Dump a config file

```sh
idas_convert dump_config
```

### Example Config
```yaml
--- !idas.iDASConvertConfig
# Loading TDMS in parallel and process
nthreads_loading: 1
nthreads_processing: 8
queue_size: 32
processing_batch_size: 8

# Threads used for downsampling the data
nthreads: 8

# Input paths
paths:
- /home/isken/src/idas-convert

# Out path, see pyrocko.io for details
outpath: '%(tmin_year)s%(tmin_month)s%(tmin_day)s/%(network)s.%(station)s_%(tmin_year)s%(tmin_month)s%(tmin_day)s.mseed'

# Overwrite mseed meta information
new_network_code: ID
new_channel_code: HSF

downsample_to: 200.0

# MiniSeed record length
record_length: 4096
# MiniSeed STEIM compression
steim: 2

plugins:

# A plugin handling the communication with the GFZ tage file system
- !idas_convert.gfz_tapes.GFZTapesConfig
  enabled: false
  bytes_stage: 1T
  waterlevel: 0.6
  wait_warning_interval: 600.0
  release_files: true
  path_tapes_mount: /projects/ether/
  path_tapes_prefix: /archive_FO1/RAW/

# A Telegram bot to keep up-to-date with the process
- !idas_convert.telegram_bot.TelegramBotConfig
  enabled: false
  # Telegram API Token
  token: 9e98b8c0567149eb861838a1d770be7d
  # Telegram Chat ID
  chat_id: -1237123123
  # A status message will be dispatched every 3600 s
  status_interval: 3600.0
```

### Start Conversion

```sh
idas_convert my_config.yml
```

See `idas_convert -h` for more options.
