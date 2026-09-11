import numpy as np


def psth(spk_times, dt, nt, frac = 1):
    """
    spk_times: list of repeats, with each repeat containing spike times (s)
    dt: duration of one frame (s)
    nt: number of stimulus frames in the repeat
    frac: upsample bins for spike times by frac

    cents: bin centers (s)
    rate: bin firing rate (Hz)
    """

    n_reps = len(spk_times) # number of repeats

    trial_dur = nt*dt #seconds

    n_bins = int(trial_dur/dt*frac)

    counts = np.zeros((n_reps,n_bins))

    for i, repeat in enumerate(spk_times):
        counts[i], bins = np.histogram(repeat, bins=n_bins, range=(0,trial_dur)) #counts for every individual trial

    total_counts = np.sum(counts,0) # sum counts along trial dimension
    cents = np.diff(bins)+bins[:-1] # bin centers, not sure if this is best practice
    rate = total_counts/n_reps/dt

    return cents, rate