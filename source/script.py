###############################################################################
# Load / Import what you need

import numpy as np # Numpy for array and matrices
import pylab as pl # Pylab for plotting (matlab like syntax)

# Import MNE modules
import mne
from mne.datasets import sample
from mne.viz import plot_evoked

# get path to raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
print raw_fname

# Read raw data (eq. mne_setup_raw.m)
raw = mne.fiff.Raw(raw_fname)
print raw

# Read data segment
start, stop = raw.time_to_index(100, 115) # 100 s to 115 s data segment
data, times = raw[2:20:3, start:stop] # take Magnetometers
print data.shape
print times.shape

# Plot
pl.plot(times, data.T)
pl.xlabel('Time (s)')
pl.ylabel('MEG (T)')

# Find events in data
events = mne.find_events(raw, stim_channel='STI 014')
print events[:5]

# Define Epochs
event_id, tmin, tmax = 1, -0.2, 0.5
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude=exclude)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=False)
print epochs

# Average Epochs -> Evoked data
evoked = epochs.average()
print evoked

# Plot Evoked data
plot_evoked(evoked)

# Python syntaxic sugar :

# Get the max of all epochs
max_in_each_epoch = [e.max() for e in epochs] # doctest:+ELLIPSIS

# Get average of Epochs if max is lower than 3.5e-5
evoked_data = [e for e in epochs if np.max(np.abs(e)) < 3.5e-5]


# You can also do TF, stats, dSPM