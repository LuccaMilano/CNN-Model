from numpy import *
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

from pdb import set_trace as pause

from haar import *
import os
import sys

N = 128
v = loadtxt(os.path.join(os.path.dirname(sys.argv[0])+'/firstEpochEEG.txt'))
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


fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6), sharex=True, sharey=True)

sumWaves = vGamma + vBeta + vAlpha + vTheta + vDelta
axes[0].plot(sumWaves, label='raw', color='blue')
axes[0].set_ylim([vmin, vmax])
axes[0].set_ylabel('Tensão (µV)')

axes[1].plot(vGamma, label=r'$\gamma$', color='orange')
axes[1].set_ylim([vmin, vmax])
axes[1].set_ylabel('Tensão (µV)')
#axes[0].legend(frameon=False)
#plt.show()
axes[2].plot(vBeta, label=r'$\beta$', color='green')
axes[2].set_ylim([vmin, vmax])
axes[2].set_ylabel('Tensão (µV)')
#axes[1].legend()
#plt.show()
axes[3].plot(vAlpha, label=r'$\alpha$', color='red')
axes[3].set_ylim([vmin, vmax])
axes[3].set_ylabel('Tensão (µV)')
#axes[2].legend()
#plt.show()
axes[4].plot(vTheta, label=r'$\theta$', color='purple')
axes[4].set_ylim([vmin, vmax])
axes[4].set_ylabel('Tensão (µV)')
#axes[3].legend()
#plt.show()
axes[5].plot(vDelta, label=r'$\delta$', color='brown')
axes[5].set_ylim([vmin, vmax])
axes[5].set_ylabel('Tensão (µV)')
#axes[4].legend()
axes[5].set_xlabel('Tempo (cs)')

#plt.show()
# sumWaves = vGamma + vBeta + vAlpha + vTheta + vDelta
# plt.plot(sumWaves, label='sumWaves')


#axes[2, 1].plot(v, label='Raw')
#axes[2, 1].legend()
assert isclose(sum(v-(vGamma+vBeta+vAlpha+vTheta+vDelta)),0), 'The sum of the components is the raw signal.'

plt.tight_layout()
plt.show()
plt.savefig("TWD.pdf",bbox_inches='tight')

#pause()


