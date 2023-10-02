import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data

raw_train = mne.io.read_raw_edf("sleep-cassette/SC4012E0-PSG.edf", stim_channel='Event marker',
                                        misc=['Temp rectal'])
annot_train = mne.read_annotations("sleep-cassette/SC4012EC-Hypnogram.edf")

annotation_desc_2_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

# keep last 30-min wake events before sleep and first 30-min wake events after
# sleep and redefine annotations on raw data
annot_train.crop(annot_train[1]["onset"] - 30 * 60, annot_train[-2]["onset"] + 30 * 60)
raw_train.set_annotations(annot_train, emit_warning=False)

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0
)

# create a new event_id that unifies stages 3 and 4
event_id = {
    "Alerta": 1,
    "Sonolência": 2,
    "Estágio de sono 2": 3,
    "Estágio de sono NREM": 4,
    "Estágio de sono REM": 5,
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


# plot events
fig = mne.viz.plot_events(
    events_train,
    event_id=event_id,
    sfreq=raw_train.info["sfreq"],
    first_samp=events_train[0, 0],
    show=False
)

# Set x-axis and y-axis labels
fig.axes[0].set_xlabel('Tempo (s)')
fig.axes[0].set_ylabel('ID do Evento')

# Convert x-axis ticks to seconds
x_ticks_original = fig.axes[0].get_xticks()
x_ticks_portuguese = [f"{int(t)} s" for t in x_ticks_original]
fig.axes[0].set_xticklabels(x_ticks_portuguese)

plt.show()