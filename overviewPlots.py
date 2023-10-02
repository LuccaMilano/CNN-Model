import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from pdb import set_trace as pause
from Wavelet.haar import *

raw_train = mne.io.read_raw_edf("sleep-cassette/SC4012E0-PSG.edf", stim_channel='Event marker',
                                        misc=['Temp rectal'])
annot_train = mne.read_annotations("sleep-cassette/SC4012EC-Hypnogram.edf")

raw_train.set_annotations(annot_train, emit_warning=False)

annotation_desc_2_event_id = {'Sleep stage W': 0,
                            'Sleep stage 1': 1}

annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                annot_train[-2]['onset'] + 30 * 60)
raw_train.set_annotations(annot_train, emit_warning=False)

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0)

event_id = {'Sleep stage W': 0,
            'Sleep stage 1': 1}
tmax = 30.0 - 1.0 / raw_train.info["sfreq"]
epochs = mne.Epochs(raw=raw_train, events=events_train,
                        event_id=event_id, tmin=0., tmax=tmax, baseline=None)

eventsArray = epochs.events[:, 2]
dataArray = epochs.get_data()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

plt.figure(figsize=(12, 2))

plt.plot(dataArray[0][1], label='raw', color='blue')

plt.tick_params(top='off', bottom='off', left='off', right='off')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticks([])
plt.savefig("raw.pdf",bbox_inches='tight')

# Splitting the data into the desired portions
data_segment_1 = dataArray[0][1][:dataArray.shape[2] // 30]
data_segment_2 = dataArray[0][1][dataArray.shape[2] // 30:]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

plt.figure(figsize=(12, 2))

# Plot the first 1/30 segment in a different color
plt.plot(data_segment_1, color='orange')

x_offset = len(data_segment_1)
plt.plot(range(x_offset, x_offset + len(data_segment_2)), data_segment_2, color='blue')


plt.xlim(0, 0.33*(len(dataArray[0][1])))
plt.tick_params(top='off', bottom='off', left='off', right='off')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticks([])
plt.savefig("segment.pdf",bbox_inches='tight')


N = 128
v = dataArray[0][1][:128]
#v = loadtxt('firstEpochEEG.txt')

f = 1/sqrt(2)
H = Hl(N,5,f)

assert isclose(sum(H.T - linalg.inv(H)), 0), 'H is not orthogonal'

w = dot(H,v)

linf = N//2; lsup = N  ; D1 = w[linf:lsup]; wD1 = zeros_like(v); wD1[linf:lsup] = D1; #print(linf, lsup)
linf //= 2 ; lsup //= 2; D2 = w[linf:lsup]; wD2 = zeros_like(v); wD2[linf:lsup] = D2; #print(linf, lsup)
linf //= 2 ; lsup //= 2; D3 = w[linf:lsup]; wD3 = zeros_like(v); wD3[linf:lsup] = D3; #print(linf, lsup)
linf //= 2 ; lsup //= 2; D4 = w[linf:lsup]; wD4 = zeros_like(v); wD4[linf:lsup] = D4; #print(linf, lsup)
linf //= 2 ; lsup //= 2; D5 = w[linf:lsup]; wD5 = zeros_like(v); wD5[linf:lsup] = D5; #print(linf, lsup)
lsup = linf; linf = 0  ; C5 = w[linf:lsup]; wC5 = zeros_like(v); wC5[linf:lsup] = C5; #print(linf, lsup)

vGamma = dot(H.T, wD1)
vBeta = dot(H.T, wD2)
vAlpha = dot(H.T, wD3)
vTheta = dot(H.T, wD4)
vDelta = dot(H.T, wC5+wD5)

vmin = 1.1*min(hstack([v, vGamma, vBeta, vAlpha, vTheta, vDelta]))
vmax = 1.1*max(hstack([v, vGamma, vBeta, vAlpha, vTheta, vDelta]))

#vmin *= 1e6
#vmax *= 1e6
#print(vmin, vmax)

v *= 1e6
vGamma *= 1e6
vAlpha *= 1e6
vBeta *= 1e6
vTheta *= 1e6
vDelta *= 1e6

vmin = -70
vmax = 70

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

plt.figure(figsize=(12, 1))

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 5), sharex=False, sharey=True)

sumWaves = vGamma + vBeta + vAlpha + vTheta + vDelta

axes[0].plot(vGamma, label=r'$\gamma$', color='orange')
axes[0].set_ylim([vmin, vmax])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].get_xaxis().set_ticks([])
axes[0].get_yaxis().set_ticks([])

axes[1].plot(vBeta, label=r'$\beta$', color='green')
axes[1].set_ylim([vmin, vmax])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].get_xaxis().set_ticks([])

axes[2].plot(vAlpha, label=r'$\alpha$', color='red')
axes[2].set_ylim([vmin, vmax])
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].spines['left'].set_visible(False)
axes[2].get_xaxis().set_ticks([])

axes[3].plot(vTheta, label=r'$\theta$', color='purple')
axes[3].set_ylim([vmin, vmax])
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].spines['bottom'].set_visible(False)
axes[3].spines['left'].set_visible(False)
axes[3].get_xaxis().set_ticks([])

axes[4].plot(vDelta, label=r'$\delta$', color='brown')
axes[4].set_ylim([vmin, vmax])
axes[4].spines['top'].set_visible(False)
axes[4].spines['right'].set_visible(False)
axes[4].spines['bottom'].set_visible(False)
axes[4].spines['left'].set_visible(False)


assert isclose(sum(v-(vGamma+vBeta+vAlpha+vTheta+vDelta)),0), 'The sum of the components is the raw signal.'

plt.tight_layout()
plt.savefig("subbands.pdf",bbox_inches='tight')

# Splitting the data into the desired portions
data_segment_1 = dataArray[60][1][:dataArray.shape[2] // 30]
data_segment_2 = dataArray[60][1][dataArray.shape[2] // 30:]

data_segment_1 *= 1e6
data_segment_2 *= 1e6

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

plt.figure(figsize=(12, 2))

# Plot the first 1/30 segment in a different color
plt.plot(data_segment_1, color='orange')

x_offset = len(data_segment_1)
plt.plot(range(x_offset, x_offset + len(data_segment_2)), data_segment_2, color='blue')

plt.ylabel('Tensão (µV)')
plt.xlabel('Tempo (cs)')
plt.xlim(0, 0.33*(len(dataArray[0][1])))
plt.savefig("drowsy6.pdf",bbox_inches='tight')

# Splitting the data into the desired portions
data_segment_1 = dataArray[59][1][:dataArray.shape[2] // 30]
data_segment_2 = dataArray[59][1][dataArray.shape[2] // 30:]

data_segment_1 *= 1e6
data_segment_2 *= 1e6

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

plt.figure(figsize=(12, 2))

# Plot the first 1/30 segment in a different color
plt.plot(data_segment_1,  color='orange')

x_offset = len(data_segment_1)
plt.plot(range(x_offset, x_offset + len(data_segment_2)), data_segment_2, color='blue')

plt.ylabel('Tensão (µV)')
plt.xlabel('Tempo (cs)')
plt.xlim(0, 0.33*(len(dataArray[0][1])))
plt.savefig("awake6.pdf",bbox_inches='tight')