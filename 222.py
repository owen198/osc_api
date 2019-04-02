'''
Created on 2019年3月13日

@author: I28564
'''



import osc_grafana as osc  # include define lib for osc in grafana
import get_env_variable as env_var  # include define lib for get env variable
target_name = 'Amplitude'
resp_item ={
        'target': target_name,
        'datapoints': [1,2,3,4,5]    # data
}
x = [5]

if len(x) != 0:
        # Compute and plot the spectrogram.
        Fs = 8192.0  # rate
        Ts = 1.0/Fs # interval
        ff = 5  # frequency of the signal

        n = len(x) # length of the signal
        k = np.arange(n)
        T = n/Fs
        X, Ys = wavelet_comp_inv(x,Fs,Level=4,wavName='db5')
        print(len(Ys))
        Y = Ys[3]
        wavelet = []
        for i in range(0,X.shape[0]):
            wavelet.append([float(Y[i]), float(X[i])])

        resp_item['datapoints'] = wavelet
        resp.append(resp_item)

else:
        resp_item['datapoints'] = []
        
print(resp_item)