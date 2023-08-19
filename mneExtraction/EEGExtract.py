import mne
from pdb import set_trace as pause

class EEGClass:
    def __init__(self, patientFile, patientHypnogram):
        raw_train = mne.io.read_raw_edf(patientFile, stim_channel='Event marker',
                                        misc=['Temp rectal'])
        annot_train = mne.read_annotations(patientHypnogram)

        raw_train.set_annotations(annot_train, emit_warning=False)

        annotation_desc_2_event_id = {'Sleep stage W': 0,
                                    'Sleep stage 1': 1}
        
        annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                        annot_train[-2]['onset'] + 30 * 60)
        raw_train.set_annotations(annot_train, emit_warning=False)

        events_train, _ = mne.events_from_annotations(
            raw_train, event_id=annotation_desc_2_event_id, chunk_duration=1.0)

        event_id = {'Sleep stage W': 0,
                    'Sleep stage 1': 1}
        tmax = 1.0 - 1.0 / raw_train.info["sfreq"]
        self.epochs = mne.Epochs(raw=raw_train, events=events_train,
                                event_id=event_id, tmin=0., tmax=tmax, baseline=None)