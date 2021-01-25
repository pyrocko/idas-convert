# iDAS TDMS Converter

this suite converts and downsamples Silixa iDAS TDMS data files to MiniSeed.

## Installation

```sh
python3 setup.py install
```

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
  token: Telegram Token
  chat_id: Telegram Chat ID, e.g. -456413218
  status_interval: 3600.0
```

### Start Conversion

```sh
idas_convert my_config.yml
```

See `idas_convert -h` for more options.
