===================================================
Intro to MEG and EEG processing with MNE and Python
===================================================

**Authors:** A. Gramfort and M. Hämäläinen

.. role:: input(strong)


Introduction
==============

MNE Python: The project vision
------------------------------

    - A package to **make your life easier** when interacting **with MEG/EEG data**
    - KISS principle : **Keep it super simple** !
    - Robust software with good engineering: **tests**, **coverage** analysis, **code quality** control
    - **Open** project: very permissive BSD license, open version control system to facilitate contributions
    - The project should be maintained by a **community of labs**

What you're not supposed to do with MNE Python
----------------------------------------------

    - **Process raw files**: In short everything you do with *mne_process_raw* (filtering, computing SSP vectors, downsampling etc.)
    - **Forward modeling**: BEM computation and mesh creation (done with FreeSurfer)
    - **Raw data visualization** done with *mne_browse_raw*
    - **MNE source estimates visualization** done *mne_analyze*

What you can do with MNE Python
----------------------------------------------

    - **Epoching**: Define epochs, baseline correction etc.
    - **Averaging** to get Evoked data
    - **Linear inverse solvers** (dSPM, MNE)
    - **Time-frequency** analysis with Morlet wavelets (induced power, phase lock value) also in the source space
    - **Non-parametric statistics** in time, space and frequency
    - **Scripting** (batch and parallel computing)

.. note:: Packaged based on the FIF file format from Neuromag but can work with CTF and 4D after conversion to FIF.

Why Python?
-----------

    - Full **control of the memory** you use for your analysis
    - **Parallel** processing
    - **Memoizing** functions
    - **Packaging**, **software engineering tools** are shared across disciplines (not only scientific computing)
    - Runs on **all systems** (Linux, Mac and Windows)
    - **Free**
    - Python is a real language that allows to design **clean and powerful APIs**

Installation of the required materials
---------------------------------------

The Python scientific computing environment: **Numpy**, **Scipy**, also the **scikit-learn** (optional).
Numpy provides data structures (array,
matrices) and Scipy provides algorithms (linear algebra, signal processing, etc.). For parallel computing
it uses `joblib`_ shipped with the `scikit-learn`_ .

.. _joblib: http://http://packages.python.org/joblib/
.. _scikit-learn: http://http://scikit-learn.sourceforge.net/


Get the code
^^^^^^^^^^^^

  https://github.com/mne-tools/mne-python

From raw data to evoked data
============================

.. _ipython: http://ipython.scipy.org/

Now, launch `ipython`_ (Advanced Python shell)::

  $ ipython -pylab -wthread

First, load the mne package:

    >>> import mne

Access raw data
---------------

.. doctest::

    >>> from mne.datasets import sample
    >>> data_path = sample.data_path()
    >>> raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    >>> print raw_fname
    ./MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif

.. note:: The MNE sample dataset should be downloaded automatically but be patient (> 600MB)

Read data from file:

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

Save a segment of 150s of raw data (MEG only):

    >>> picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, stim=True)
    >>> raw.save('sample_audvis_meg_raw.fif', tmin=0, tmax=150, picks=picks)

Define and read epochs
----------------------

First extract events:

    >>> events = mne.find_events(raw, stim_channel='STI 014')
    Reading 6450 ... 48149  =     42.956 ...   320.665 secs...  [done]
    >>> print events[:5]
    [[6994    0    2]
     [7086    0    3]
     [7192    0    1]
     [7304    0    4]
     [7413    0    2]]

Define epochs parameters:

    >>> event_id = 1
    >>> tmin = -0.2
    >>> tmax = 0.5

Exclude some channels (bads + 2 more):

    >>> exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']

Pick the good channels:

    >>> picks = mne.fiff.pick_types(raw.info, meg=True, eeg=True, eog=True, stim=False, exclude=exclude)

Read epochs:

    >>> epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=(None, 0),
                            preload=False, reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
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

Inverse modeling: MNE and dSPM on evoked and raw data
=====================================================

Import the required functions:

    >>> from mne.minimum_norm import apply_inverse, read_inverse_operator

Read the inverse operator:

    >>> fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
    >>> inverse_operator = read_inverse_operator(fname_inv)

Define the inverse parameters:

    >>> snr = 3.0
    >>> lambda2 = 1.0 / snr ** 2
    >>> dSPM = True

Compute the inverse solution:

    >>> stc = apply_inverse(evoked, inverse_operator, lambda2, dSPM)

Save the source time courses to disk:

    >>> stc.save('mne_dSPM_inverse')

Now, let's compute dSPM on a raw file within a label:

    >>> fname_label = data_path + '/MEG/sample/labels/Aud-lh.label'
    >>> label = mne.read_label(fname_label)

Compute inverse solution during the first 15s:

    >>> start, stop = raw.time_to_index(0, 15)  # read the first 15s of data
    >>> stc = apply_inverse_raw(raw, inverse_operator, lambda2, dSPM, label, start, stop)

Save result in stc files:

    >>> stc.save('mne_dSPM_raw_inverse_Aud')

What else can I do?
===================

    - morph stc from one brain to another for group studies
    - estimate power in the source space
    - estimate noise covariance matrix from Raw and Epochs
    - detect heart beat QRS component
    - detect eye blinks and EOG artifacts

What comes next?
================

    - sparse solvers
    - coherence measures
    - anything you want to contribute for the community !

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


Want to know more ? Go to `martinos.org/mne`_
=================================================

.. _martinos.org/mne: http://www.martinos.org/mne

