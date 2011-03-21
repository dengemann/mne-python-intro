==========================================
MEG and EEG processing with MNE and Python
==========================================

**Authors:** A. Gramfort and M. Hämäläinen

.. role:: input(strong)


Introduction
==============

What is the MNE Python Package?
-------------------------------

    - A pure Python package to **make your life easier** when interacting **with data**
    - Philosophy : **Keep things super simple** !
    - Software engineering: **tests**, **coverage** analysis, **code quality** control (PEP8)

What you're not supposed to do with MNE Python
----------------------------------------------

    - **Process raw files**: In short everything you do with *mne_process_raw* (filtering, computing SSP vectors, marking bad channels, downsampling etc.)
    - **Forward modeling**
    - **Advanced visualization** done with *mne_browse_raw* and  *mne_analyze* GUIs

What you can do with MNE Python
----------------------------------------------

    - **Epoching**: Define epochs, baseline correction etc.
    - **Averaging** to get Evoked data
    - **Time-frequency** analysis (induced power, phase lock value)
    - **Linear inverse solvers** (dSPM, MNE)
    - **Non-parametric statistics** in time, space and frequency
    - **Scripting** (batch and cluster computing)

Why Python?
-----------

    - Full **control of the memory** you use for your analysis
    - **Parallel** processing
    - **Memoizing** functions
    - **Packaging**, **software engineering tools** are shared across disciplines (not only scientific computing)
    - Runs on **all systems** (Linux, Mac and Windows)
    - **Free** (not my number 1 argument)

Installation of the required materials
---------------------------------------

The Python scientific computing environment: **Numpy** and **Scipy**. Numpy provides data structures (array,
matrices) and Scipy provides algorithms (linear algebra, signal processing, etc.)

Get the code
^^^^^^^^^^^^

  https://github.com/mne-tools/mne-python (open version control system and BSD License)

From raw data to evoked data
============================

Now, launch ipython (Advanced Python shell)::

  $ ipython

First, load the mne package:

    >>> import mne

Load some data (package comes with an easy access to the MNE sample data):

.. doctest::

    >>> from mne.datasets import sample
    >>> data_path = sample.data_path()
    >>> raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    >>> print raw_fname
    ./MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif

Read a raw file:

    >>> raw = mne.fiff.Raw(raw_fname) # doctest:+ELLIPSIS
    Opening raw data ...
    Ready.
    >>> print raw
    Raw (n_channels x n_times : 376 x 41700)

Read and plot a segment of raw data

    >>> start, stop = raw.time_to_index(100, 115) # 100 s to 115 s data segment
    >>> data, times = raw[:, start:stop]
    Reading 15015 ... 17266  =     99.998 ...   114.989 secs...  [done]
    >>> print data.shape
    (376, 2252)
    >>> print times.shape
    (2252,)
    >>> data, times = raw[2:20:3, start:stop] # take some Magnetometers

.. figure:: images/plot_read_raw_data.png
    :alt: Raw data

Extracts events triggers:

    >>> events = mne.find_events(raw)
    Reading 6450 ... 48149  =     42.956 ...   320.665 secs...  [done]
    >>> print events[:5]
    [[6994    0    2]
     [7086    0    3]
     [7192    0    1]
     [7304    0    4]
     [7413    0    2]]

Define and read epochs:

    >>> event_id = 1
    >>> tmin = -0.2
    >>> tmax = 0.5
    >>> exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more
    >>> picks = mne.fiff.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude=exclude)
    >>> epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=False)
    4 projection items activated
    The projection vectors do not apply to these channels
    72 matching events found
    >>> print epochs
    Epochs (n_epochs : 72, tmin : -0.2 (s), tmax : 0.5 (s), baseline : (None, 0))

Compute evoked responses by averaging and plot it:

    >>> evoked = epochs.average() # doctest: +ELLIPSIS
    Reading ...
    >>> print evoked
    Evoked (comment : Evoked data, time : [-0.199795, 0.492828], n_epochs : 72, n_channels x n_times : 364 x 105)
    >>> from mne.viz import plot_evoked
    >>> plot_evoked(evoked)

.. figure:: images/plot_read_epochs.png
    :alt: Evoked data

.. topic:: Exercise

  1. Extract the max value of each epoch

  >>> max_in_each_epoch = [e.max() for e in epochs] # doctest:+ELLIPSIS
  Reading ...
  >>> print max_in_each_epoch[:4]
  [2.6751692973693302e-05, 3.5135456261958446e-05, 2.0282791755715339e-05, 2.2940160602805886e-05]

Some screen shots
=================

.. figure:: images/plot_topography.png
    :alt: 2D toprography
    
    2D toprography

.. figure:: images/plot_time_frequency.png
    :alt: Time Frequency

    Time frequency decomposition of one sensor

.. figure:: images/plot_cluster_1samp_test_time_frequency.png
    :alt: Cluster level stat in time Frequency decomposition

    Cluster level stat in time Frequency decomposition

.. figure:: images/cluster_full_layout_c0-c1.png
    :alt: Topography of cluster level stat in time

    Topography of cluster level stat in time

.. figure:: images/plot_cluster_stats_evoked.png
    :alt: Statistics on evoked data

    Statistics on evoked data


Some more ? go to www.martinos.org/mne (soon public)
====================================================

