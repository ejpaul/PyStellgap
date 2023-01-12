import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd.vmec import Vmec
import sys

"""
This script produces three figures:

spectrum_*.pdf - Plots the frequency as a function of radius for each continuum
mode. The label and color of the markers are determined by the mode numbers.

hist_full_*.png - Plots a histogram of the frequencies of all continuum modes,
including all radii.

hist_target_*.png - Plots a histogram of the frequencies of all continuum modes
centered at s_target within a window of s_width.

Additional lines are overlaid on each plot to indicate the theoretically predicted
gap location based on a cylindrical model. 
"""

plotname = '' # This string is appended to the end of the figure names
s_target = 0.5 # Sets the central frequency of the hist_target_*.png histogram
s_width = 0.1 # Sets the frequency with of the hist_target_*.png histogram
freqmin = 0 # Minimum frequency to plot
freqmax = 1500 # Maximum frequency to plot
markevery = 1 # If > 1, every markevery radial points will be included in figures.
nproc = 64 # Number of processors the Stellgap calculation was performed with

colors = plt.cm.jet(np.linspace(0, 1, 10))

NUM_COLORS = 30
cm = plt.get_cmap('gist_rainbow')

ion_profile = np.loadtxt('ion_profile')
vA_prof = ion_profile[:,3]
iota_prof = ion_profile[:,2]
s_prof = ion_profile[:,0]

vmec = Vmec('../../wout_QH_bootstrap.nc')
epsilon = 2.5/vmec.wout.aspect
R_full = vmec.wout.rmnc[0,:]
phi = vmec.wout.phi
s_full = phi/phi[-1]

R_prof = np.interp(s_prof,s_full,R_full)

f_prof = np.abs(1e-3*iota_prof*vA_prof/(2*R_prof*2*np.pi))
deltaf_prof = np.abs(1e-3*epsilon*vA_prof*iota_prof/(2*R_prof*2*np.pi))

f_prof_m_1 = np.abs(1e-3*(np.abs(iota_prof))*vA_prof/(2*R_prof*2*np.pi))
f_prof_m_2 = np.abs(1e-3*(np.abs(iota_prof)*2)*vA_prof/(2*R_prof*2*np.pi))
f_prof_n_1_m_2 = np.abs(1e-3*(np.abs(iota_prof)*2-vmec.wout.nfp)*vA_prof/(2*R_prof*2*np.pi))
f_prof_n_1 = np.abs(1e-3*(-vmec.wout.nfp)*vA_prof/(2*R_prof*2*np.pi))
f_prof_n_1_m_1 = np.abs(1e-3*(np.abs(iota_prof)-vmec.wout.nfp)*vA_prof/(2*R_prof*2*np.pi))

f_prof_m_1_target = np.interp(s_target,s_prof,f_prof_m_1)
f_prof_m_2_target = np.interp(s_target,s_prof,f_prof_m_2)
f_prof_n_1_m_2_target = np.interp(s_target,s_prof,f_prof_n_1_m_2)
f_prof_n_1_target = np.interp(s_target,s_prof,f_prof_n_1)
f_prof_n_1_m_1_target = np.interp(s_target,s_prof,f_prof_n_1_m_1)

fig = plt.figure(1)
plt.xlabel(r'$\sqrt{s}$')
plt.ylabel('Freq [kHz]')
plt.plot(np.sqrt(s_prof),f_prof,label=r'$f_{\delta m=1}$')
plt.plot(np.sqrt(s_prof),2*f_prof,label=r'$f_{\delta m=2}$')
plt.plot(np.sqrt(s_prof),f_prof_n_1_m_2,label=r'$f_{\delta m = 2, \delta n = 1}$')
plt.plot(np.sqrt(s_prof),f_prof_n_1_m_1,label=r'$f_{\delta m = 1, \delta n = 1}$')
plt.plot(np.sqrt(s_prof),f_prof_n_1,label=r'$f_{\delta n = 1}$')
ax = plt.gca()
marker_list = []
linestyle_list = []
for i in range(NUM_COLORS):
    if (i%4 == 0):
        marker_list.append('1')
        linestyle_list.append('--')
    elif (i%4 == 1):
        marker_list.append('|')
        linestyle_list.append('-')
    elif (i%4 == 2):
        marker_list.append('+')
        linestyle_list.append(':')
    elif (i%4 == 3):
        marker_list.append('X')
        linestyle_list.append('-.')
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)],marker=marker_list)

bins = np.linspace(freqmin,freqmax,50)
freq_all = np.zeros((0,))
freq_target = np.zeros((0,))
m_all = np.zeros((0,))
n_all = np.zeros((0,))
s_all = np.zeros((0,))
alfven_spec_filename = 'alfven_spec'
for i in range(nproc):
    alfven = np.loadtxt(alfven_spec_filename+str(i),skiprows=1)
    s = alfven[:,0]
    alfr = alfven[:,1]
    beta = alfven[:,3]
    m = alfven[:,4]
    n = alfven[:,5]

    freq = np.abs(alfr[beta!=0]/beta[beta!=0])
    s = s[beta!=0]
    m = m[beta!=0]
    n = n[beta!=0]

    s = s[freq>=0]
    freq = freq[freq>=0]
    m = m[freq>=0]
    n = n[freq>=0]

    freq_all = np.hstack((freq_all,np.sqrt(freq)))
    freq_target = np.hstack((freq_target,np.sqrt(freq[np.abs(s-s_target)<=s_width])))
    m_all = np.hstack((m_all,m))
    n_all = np.hstack((n_all,n))
    s_all = np.hstack((s_all,s))

freq = freq_all
m = m_all
n = n_all
s = s_all

m_list = range(0,int(np.max(m))+1)
n_list = np.arange(int(np.min(n)),int(np.max(n))+1,vmec.wout.nfp)
s_list = np.sort(np.unique(s))[0:-1:markevery]

for i in range(len(n)):
    if np.any(n[i]==n_list) == False:
        print('n = ',n[i])
    if np.any(m[i]==m_list) == False:
        print('m == ',m[i])

for this_m in m_list:
    for this_n in n_list:
        this_freq = freq[(m==this_m)*(n==this_n)*(freq >= freqmin)*(freq <= freqmax)*np.isin(s,s_list)]
        this_s = s[(m==this_m)*(n==this_n)*(freq >= freqmin)*(freq <= freqmax)*np.isin(s,s_list)]
        if (len(this_freq)>0):
            plt.plot(np.sqrt(this_s),this_freq,label='m= '+str(this_m)+', n= '+str(this_n),linestyle=None,markersize=3,lw=0)

print('Maximum frequency: ',np.max(freq))

plt.figure(1)
plt.ylim([freqmin,freqmax])
lgd = plt.legend(bbox_to_anchor=(1.01,1.01),ncol=2)
plt.savefig('spectrum'+plotname+'.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(3)
plt.hist(freq_all,bins=bins)
plt.xlabel('Freq [kHz]')
plt.axvline(np.min(f_prof_n_1_m_2),label=r'$f_{\delta m = 2, \delta n = 1}$',color='red')
plt.axvline(np.max(f_prof_n_1_m_2),color='red')
plt.axvline(np.min(f_prof_n_1),label=r'$f_{\delta n = 1}$',color='green')
plt.axvline(np.max(f_prof_n_1),color='green')
plt.axvline(np.min(f_prof_m_2),label=r'$f_{\delta m=2}$',color='black')
plt.axvline(np.max(f_prof_m_2),color='black')
plt.axvline(np.min(f_prof_m_1),label=r'$f_{\delta m=1}$',color='orange')
plt.axvline(np.max(f_prof_m_1),color='orange')
plt.axvline(np.min(f_prof_n_1_m_1),label=r'$f_{\delta m = 1, \delta n = 1}$',color='purple')
plt.axvline(np.max(f_prof_n_1_m_1),color='purple')
plt.legend()
plt.xlim([0,1000])
plt.savefig('hist_full'+plotname+'.png')

plt.figure(2)
plt.hist(freq_target,bins=bins)
plt.axvline(f_prof_n_1_m_2_target,label=r'$f_{\delta m = 2, \delta n = 1}$',color='red')
plt.axvline(f_prof_n_1_target,label=r'$f_{\delta n = 1}$',color='green')
plt.axvline(f_prof_m_2_target,label=r'$f_{\delta m=2}$',color='black')
plt.axvline(f_prof_m_1_target,label=r'$f_{\delta m=1}$',color='orange')
plt.axvline(f_prof_n_1_m_1_target,label=r'$f_{\delta m = 1, \delta n = 1}$',color='purple')
plt.legend()
plt.xlabel('Freq [kHz]')
plt.title('Histogram near $s = $'+str(s_target))
plt.savefig('hist_target'+plotname+'.png')
