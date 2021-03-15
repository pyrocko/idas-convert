Setup
=============

Installation
------------

Python 3 requirements:

* `pyrocko <https://pyrocko.org>`_
* `NumPy <https://numpy.org>`_
* telebot (optional)

Installation using pip

.. code-block :: bash

   pip3 install git+https://git.pyrocko.org/pyrocko/idas-convert


Or installation from source:

.. code-block :: bash

   git clone https://git.pyrocko.org/pyrocko/idas-convert.git
   cd idas-convert
   python3 setup.py install

Check out the help with:

.. code-block :: bash

   idas_convert -h


Quickstart
----------

Before starting the conversion edit the config file. See details of the YAML config in the :doc:`config`.

.. code-block :: bash

   idas_convert dump_config > my_config.yml
   idas_convert my_config.yml

In case of an abort or error the conversion can be resumed with:

.. code-block :: bash

   idas_convert --resume my_config.yml


Configuration
-------------

The conversion tools are configured with a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file. This file is then _executed_ by ``idas_convert`` CLI program.

.. code-block :: YAML

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

    # Start time (optional)
    # tmin: 2021-05-03 00:00:00.00
    # End time (optional)
    # tmax: 2021-05-06 00:00:00.00

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

Plugins
-------

The following plugins can be configured in the ``plugins`` list in the YAML file.

Telegram Bot
^^^^^^^^^^^^

A Telegram bot can be configured to keep up-to-date with the processing progress.
This plugin is forwarding the log levels ``INFO`` and ``WARNING`` to the chat bot, simply add the bot to a Telegram chat group.

Details about the Telegram ``token`` and ``chat_id`` can be found `here <https://core.telegram.org/bots>`_.

.. code-block :: YAML

    - !idas_convert.telegram_bot.TelegramBotConfig
    enabled: false
    # Telegram API Token
    token: 9e98b8c0567149eb861838a1d770be7d
    # Telegram Chat ID
    chat_id: -1237123123
    status_interval: 3600.0


GFZ Tape Interaction
^^^^^^^^^^^^^^^^^^^^

The `GFZ German Research Centre for Geosciences <https://gfz-potsdam.de>`_ maintains a tape storage system, details about the system `here <https://www.golem.de/news/bandlaufwerke-als-backupmedium-ein-bisschen-tetris-spielen-1906-141575.html>`_ (in German).
This plugin enables seamless inteaction with the SAMFS RPC call procedures, needed to stage and release the files from the hot-storage.

.. code-block :: YAML

    - !idas_convert.gfz_tapes.GFZTapesConfig
    enabled: false
    bytes_stage: 1T
    waterlevel: 0.6
    wait_warning_interval: 600.0
    release_files: true
    path_tapes_mount: /projects/ether/
    path_tapes_prefix: /archive_FO1/RAW/
