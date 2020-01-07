#-*-coding:utf-8 -*-
# For flask implementation
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import datetime
import time
import json
import traceback
import requests
import re
import os
import copy
import scipy.fftpack
import configparser
import difflib

#import sys
#import platform

import numpy as np
#import pandas as pd

from scipy import signal

from influxdb import InfluxDBClient

import osc_grafana as osc  # include define lib for osc in grafana
import get_env_variable as env_var  # include define lib for get env variable


"""
Get data from influxdb to compute.
measurement: cpu_v1


api compute for wave transform.
request from grafana.
return data for grafana panel.
"""



app = Flask(__name__)


### set connect ###

# # read config file
config = configparser.ConfigParser()
config.read('config.ini')
host = config['influxdb']['host']
port = config['influxdb']['port']
database = config['influxdb']['database']
username = config['influxdb']['username']
password = config['influxdb']['password']

client = InfluxDBClient(host, port, username, password, database)   # connect influxdb


# get cf environment variable
#obj = env_var.get_influxdb_info()   # get cf environment variable
#client = InfluxDBClient(obj['host'], obj['port'], obj['username'], obj['password'], obj['database'])   # connect influxdb


# measurement = 'cpu_v1'

###  ###



@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    # response.headers['Content-Type'] = 'application/json'

    return response



@app.route("/", methods=['GET','POST'])
def test_1():
    """
    for simple json.
    official page: should return 200 ok. Used for "Test connection" on the datasource config page.
    """
    str_msg = 'test simple json with /'

    return jsonify({'msg': str_msg}), 200



@app.route("/annotations", methods=['GET','POST'])
def test_2():
    """
    for simple json
    """
    str_msg = 'test simple json with /annotations'

    return jsonify({'msg': str_msg}), 200




@app.route("/search", methods=['GET','POST'])
def test_3():
    """
    for simple json
    
    controller for firehose write in this function.
    """
    str_msg = 'test simple json with /search'

    #return jsonify({'msg': str_msg}), 200
    return jsonify({'msg_1': 'fft', 'msg_2': 'ceps', 'msg_3': 'wavelet', 'msg_4': 'envelope'}), 200



@app.route("/query", methods=['GET','POST'])
def test_4():

    # retrieve post JSON object
    jsonobj = request.get_json(silent=True)
    print(jsonobj)
    
    target_obj = jsonobj['targets'][0]['target']
    date_obj = jsonobj['range']['from']
    date_from = jsonobj['range']['from']
    date_to = jsonobj['range']['to']
    
    date_from = datetime.datetime.strptime(date_from, '%Y-%m-%dT%H:%M:%S.%fZ')
    date_to = datetime.datetime.strptime(date_to, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    print('date_obj=' + date_obj)
    print('date_from=', date_from)
    print('date_to=', date_to)


    # get bin file from s3 API
    # url = 'http://s3-api-envelop.fomos.csc.com.tw/query'
    url = os.getenv("S3_URL")


    json_body = {
        "timezone": "browser",
        "panelId": 2,
        "dashboardId": 56,
        "range": {
            "from": date_obj,
            "to": "2100-03-09T07:13:44.138Z",
            "raw": {
                "from": "now-6h",
                "to": "now"
            }
        },
        "rangeRaw": {
            "from": "now-6h",
            "to": "now"
        },
        "interval": "15s",
        "intervalMs": 15000,
        "targets": [
            {
                "target": target_obj,
                "refId": "A",
                "type": "timeserie"
            }
        ],
        "maxDataPoints": 1260,
        "scopedVars": {
            "__interval": {
                "text": "15s",
                "value": "15s"
            },
            "__interval_ms": {
                "text": 15000,
                "value": 15000
            }
        }
    }
    
    headers = {'Content-Type': 'application/json'}
    s3_api_resp = requests.post(url, headers=headers, data=json.dumps(json_body))
    s3_api_list = s3_api_resp.json()[0]['datapoints']
    
    # query bin file by time range
    raw_list = [row[0] for row in s3_api_list]
    time_list = [row[1] for row in s3_api_list]
    
    print('bin file from:', datetime.datetime.fromtimestamp(int(time_list[0]//1000)).strftime('%c'))
    print('bin file to:', datetime.datetime.fromtimestamp(int(time_list[-1]//1000)).strftime('%c'))
    print('bin file 1st element:', time_list[0])
    print('bin file last element:', time_list[-1], type(time_list[-1]))
    
    # Query bin file
    query_bin_from = combine_s3_query_string(date_from)
    query_bin_to = combine_s3_query_string(date_to)
    index_from = time_list.index(min(time_list, key=lambda timestamp: abs(timestamp - query_bin_from)))
    index_to = time_list.index(min(time_list, key=lambda timestamp: abs(timestamp - query_bin_to)))

    print('query_from and query_to:', index_from, index_to)
    raw_list = raw_list[index_from:index_to]
    resp = osc_envelope(raw_list)

    print('/query')
    return jsonify(resp), 200

def combine_s3_query_string(input_dt):
    epoch_second = input_dt.strftime('%s')
    milisecond = input_dt.microsecond / 1000
    query_string = str(int(epoch_second) * 1000 + milisecond)
    return float(query_string)


def osc_envelope(x):

    print('signal processing start')
    
    target_name = 'Amplitude'
    resp = []
    resp_item = {
        'target': target_name,
        'datapoints': []    # data
    }

    if len(x) != 0:
        # Compute and plot the spectrogram.
        Fs = 8192.0  # rate
        Ts = 1.0/Fs # interval
        ff = 5  # frequency of the signal

        n = len(x) # length of the signal
        k = np.arange(n)
        T = n/Fs
        x = x[0:2**16] if n > 60000 else x 
        X, Y = envelope_spectrum(x,Fs)


        envelope = []
        #for i in range(0,X.shape[0]):
        for i in range(0,int(100.0/X[1])):
            envelope.append([float(Y[i]), float(X[i])])

        resp_item['datapoints'] = envelope
        resp.append(resp_item)
    
    else:
        resp_item['datapoints'] = []

    print('signal processing end')

    return resp

def cepstrum(signal,sample_rate=8192,cut_time=0.01,suppression=0.1):
    """
    ***calculate signal cepstrum, parameters:
       cut_time: time not considered before
       suppression: the factor for amplitudes not in top rankings are suppressed by
    ****return:
        x & y coordinates for plotting       
    ****example:
    import numpy as np
    filename = u'FM7 號精軋機小齒輪箱 輸出軸操作側軸承振動-Rolling 波形資料.txt'
    with open(filename) as f: content = f.readlines()
    sample_rate = float(content[1].split(':')[1])
    signal = [float(s) for s in content[2:]]
    t = np.linspace(0.0, len(signal)*1.0/sample_rate, len(signal))
    del content
    xcoords, ycoords = cepstrum(signal,sample_rate)
    #plot
    import matplotlib.pyplot as plt
    plt.plot(xcoords, ycoords,'-b')
    plt.title('real_cepstrum (improved)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()    
    """
    import copy
    import numpy as np
    import scipy.fftpack   
    import sys
    
    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = 1.0 / sample_rate
    x = np.linspace(0.0, N*T, N)

    #fft
    yf = scipy.fftpack.fft(signal)
    #only show half signal
    ceps0 = np.fft.ifft(np.log(np.abs(yf)+sys.float_info.min)).real[0:N//2]
    #post processing
    #cut_time = 0.01
    not_considered_index_before = [index for index,xx in enumerate(x) if xx<0.01][-1:][0] + 1
    #calculate RMS or threshold
    thr1 = ceps0[not_considered_index_before:].std() * 3
    thr2 = np.sort(ceps0[not_considered_index_before:])[::-1][50] 
    thr = thr2 if thr2 > thr1 else thr1
    #multiply ahead elements by suppression
    ceps = ceps0.copy()
    ceps[:not_considered_index_before]
    ceps[:not_considered_index_before] = ceps[:not_considered_index_before] * suppression

    #remove first element
    ceps = ceps[1:]
    #pick up small elements & multiply suppression
    whs = [index for index,ce in enumerate(ceps) if abs(ce)<thr]
    ceps[whs]=ceps[whs]*suppression
    
    return x[1:N//2], ceps
    pass
    
def envelope_spectrum(signal,sample_rate=8192,bfft=True):
    """
    ****calculate signal envelope, parameters:
        bfft: whether to perform Fourier transform for the envelope signal
    ****return:
        x & y coordinates for plotting
    ****example:
    from scipy.signal import chirp
    import numpy as np
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
    xcoords, ycoords = envelope_spectrum(signal,fs,False)
    xcoordsf, ycoordsA = envelope_spectrum(signal,fs,True)
    #plot
    import matplotlib.pyplot as plt
    
    plt.subplot(211)
    plt.plot(xcoords, ycoords,'-r',label='envelope')
    plt.plot(t, signal,'-b',label='signal')
    plt.title('signal & envelope')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(212)
    plt.plot(xcoordsf, ycoordsA,'-b')
    plt.title('envelope spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.show()    
    """
    import copy
    import numpy as np
    import scipy.fftpack
    from scipy.signal import hilbert
    
    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = 1.0 / sample_rate
    x = np.linspace(0.0, N*T, N)
    
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    if not bfft: return x,amplitude_envelope
    
    yf = abs(scipy.fftpack.fft(amplitude_envelope))[:amplitude_envelope.shape[0]//2] * 2.0 / amplitude_envelope.shape[0]
    tt = np.linspace(0.0, 1.0/(2.0*T), amplitude_envelope.shape[0]/2)    
    return tt, yf
    pass
    
def wavelet_comp_inv(signal,sample_rate=8192,Level=3,wavName='db12'):
    """
    ****calculate wavelet components and perform reconstruction for one specific component, parameters:
        wavName: wavelet family
        Level: decomposition levels
    ****return:
        x & [y_lvN, y_lvN-1, ..., 1] coordinates for plotting
    ****example:
    import numpy as np
    filename = u'FM7 號精軋機小齒輪箱 輸出軸操作側軸承振動-Rolling 波形資料.txt'
    with open(filename) as f: content = f.readlines()
    sample_rate = float(content[1].split(':')[1])
    signal = [float(s) for s in content[2:]]
    t = np.linspace(0.0, len(signal)*1.0/sample_rate, len(signal))
    del content
    
    wavName='db12'
    x, ys = wavelet_comp_inv(signal,sample_rate=8192,Level=3,wavName=wavName)

    #plot
    import matplotlib.pyplot as plt
    
    plt.subplot(411)
    plt.plot(t, signal,'-r',label='signal')
    plt.title('original signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(412)
    plt.plot(t, signal,'-r',label='signal')
    plt.plot(t, ys[0],'-b',label='Level {} {}'.format(1,wavName))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.subplot(413)
    plt.plot(t, signal,'-r',label='signal')
    plt.plot(t, ys[1],'-b',label='Level {} {}'.format(2,wavName))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.subplot(414)
    plt.plot(t, signal,'-r',label='signal')
    plt.plot(t, ys[2],'-b',label='Level {} {}'.format(3,wavName))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.show()    
    """
    import copy
    import numpy as np
    import pywt
    import copy
    
    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = 1.0 / sample_rate
    x = np.linspace(0.0, N*T, N)
    
    #wavelet for db12
    LV = Level    
    db12 = pywt.Wavelet(wavName)
    print('dwt_max_level: %d' % (pywt.dwt_max_level(len(x), db12)))
    yf = pywt.wavedec(signal, db12, mode='symmetric', level=LV)
    
    rys = [] * 1
    for i in range(len(yf)-1,0,-1):
        #copy data for editing & reconstruction
        yf1 = copy.deepcopy(yf)
        index = list(set(range(len(yf)-1,-1,-1)) - set([i]))
        print('ignore indices:')
        print(index)
        for ind in index: yf1[-ind] = np.zeros_like(yf1[-ind])
        ry = pywt.waverec(yf1, wavName)
        rys.append(ry)
    
    rys.reverse()
    
    return x, rys


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(host='0.0.0.0', port=5500, debug=True)
    
    # Careful with the debug mode..

