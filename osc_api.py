#-*-coding:utf-8 -*-
# For flask implementation
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import datetime
import time
import json
import traceback
import requests
import re
import copy
import scipy.fftpack
import configparser

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
#     """
#     for simple json (return type: (json) time series)

#     plotly simple json osc api write in this function.
#     """
#     print("_DEBUG_Enter test_4()!")
#     obj_req = request.get_json(silent=True)	# get post json
#     print (obj_req)
#     if obj_req == None:
#         return jsonify({
#             'EXEC_DESC': '62',
#             'message': 'request body is wrong'
#         }), 400


#     req_param = osc.get_param_list(obj_req['targets'][0]['target']) # (dict) parse request param

#     # check '_type' is exist
#     if ('_type' not in req_param) or ('measurement' not in req_param):
#         return jsonify({
#             'EXEC_DESC': '61',
#             'message': 'lose wave transformation method.'
#         }), 400


#     query_list = osc.get_param_constraint(req_param)    # (list) transform key, value to constraint string
#     query_constraint = osc.combine_constraint(query_list)   # (string) AND constraing list

#     time_start = osc.trans_time_value(obj_req['range']['from']) # start time
#     time_end = osc.trans_time_value(obj_req['range']['to']) # end time

#     measurement = req_param['measurement'][0]
#     print('req_param:')
#     print(req_param)
#     tagValue = req_param['tagValue'][0]
#     if 'wavelet'==req_param['_type']:
        
#         global wavelet_ID
#         wavelet_ID = req_param['wavelet_ID'][0]
#         print('wavelet_ID:')
#         print(wavelet_ID)
    
#     print ("measurement: " + measurement)
#     query = osc.get_query_string(measurement, tagValue, query_constraint, time_start, time_end)    # get query string
#     print("Query string: " + query)
#     #print(sys.platform)
#     #print(platform.platform())


    # retrieve post JSON object
    jsonobj = request.get_json(silent=True)
    print(jsonobj)
    target_obj = jsonobj['targets'][0]['target']
    date_obj = jsonobj['range']['from']
    
    date_from = jsonobj['range']['from']
    date_to = jsonobj['range']['to']
    #date_obj = date_obj.split('T')[0]
    
    DATE = datetime.datetime.strptime(date_obj, '%Y-%m-%dT%H:%M:%S.%fZ')
    DATE = DATE + datetime.timedelta(hours=8)
    DATE = DATE.strftime('%Y-%m-%d')

    EQU_ID = target_obj.split('@')[0]
    FEATURE = target_obj.split('@')[1]
    TYPE = target_obj.split('@')[2]
    
    #print('Datatime='+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('EQU_ID=' + EQU_ID)
    print('Feature=' + FEATURE)
    print('Type=' + TYPE)
    print('Query Date=' + DATE)


    url = 'http://s3-api-fft.fomos.csc.com.tw/query'

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
    resp = requests.post(url, headers=headers, data=json.dumps(json_body))
    resp_list = resp.json()[0]['datapoints']
    resp_list = [row[0] for row in resp_list]
    resp = osc_fft(resp_list)

    epoch_second = datetime.datetime.strptime(date_from, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%s')
    milisecond = '{:03.0f}'.format(datetime.datetime.strptime(date_from, '%Y-%m-%dT%H:%M:%S.%fZ').microsecond / 1000.0)
    query_string = epoch_second+milisecond
    
    time_list = [row[1] for row in resp_list]
    l = [time_list.index(i) for i in time_list if query_string in str(i)]
    print('query from'+time_list[l[0]])
    
    # route to different osc api
#     if req_param['_type'][0] == 'fft':
#         print("Transfer to FFT!")
#         resp = osc_fft(resp_list)
#     elif req_param['_type'][0] == 'envelope':
#         resp = osc_envelope(query)
#     elif req_param['_type'][0] == 'ceps':
#         print("Transfer to CEPS!")        
#         resp = osc_ceps(query)
#     elif req_param['_type'][0] == 'wavelet':
#         print("Transfer to wavelet!")
#         #global wavelet_ID
#         wavelet_ID = int (req_param['wavelet_ID'][0])
#         resp = osc_wavelet(query,wavelet_ID)
#         print('wavelet_ID:')
#         print(wavelet_ID)        
#     else:
#         resp = []


    print(resp)
    print('/query')
    return jsonify(resp), 200

def osc_fft(x):
#     """
#     wave transform: fft

#     @param  query: (string) query string for influxdb.
#     @return resp: (list) fft trans result in grafana timeseries format.
#     """
#     # grafana simple json response format
    target_name = 'Amplitude'
    resp = []
    resp_item = {
        'target': target_name,
        'datapoints': []    # data
    }

#     print('query:')
#     print(query)
#     result = client.query(query)    # query influxdb
    
#     result = result.raw # trans query result to json
       

#     x = []
#     if 'series' in result:
#         items = result['series'][0]['values']   # data array with data in influxdb
                     
#         for item in items:
#             x.append(item[1])
    
#     client.close()

    
    if len(x) != 0:
        # Compute and plot the spectrogram.
        Fs = 8192.0  # rate
        Ts = 1.0/Fs # interval
        ff = 5  # frequency of the signal

        n = len(x) # length of the signal
        k = np.arange(n)
        T = n/Fs
        frq = k/T # two sides frequency range
        frq = frq[range(int(n/2))] # one side frequency ralsge
        Y = np.fft.fft(x)/n*2 # fft computing and normalization
        Y = Y[range(int(n/2))]


        fft = []
        for i in range(int(n/2)):
            fft.append([float(abs(Y[i])), float(frq[i])])

        resp_item['datapoints'] = fft
        resp.append(resp_item)
    
    else:
        resp_item['datapoints'] = []


    return resp
    
def osc_ceps(query):
#     """
#     wave transform: ceps

#     @param  query: (string) query string for influxdb.
#     @return resp: (list) fft trans result in grafana timeseries format.
#     """
#     # grafana simple json response format
    target_name = 'Amplitude'
    resp = []
    resp_item = {
        'target': target_name,
        'datapoints': []    # data
    }


#     result = client.query(query)    # query influxdb
#     result = result.raw # trans query result to json


#     x = []
#     if 'series' in result:
#         items = result['series'][0]['values']   # data array with data in influxdb
        
#         for item in items:
#             x.append(item[1])
    
#     client.close()


    if len(x) != 0:
        # Compute and plot the spectrogram.
        Fs = 8192.0  # rate
        Ts = 1.0/Fs # interval
        ff = 5  # frequency of the signal

        n = len(x) # length of the signal
        k = np.arange(n)
        T = n/Fs
        X, Y = cepstrum(x,Fs)

        ceps = []
        for i in range(0,X.shape[0]):
            ceps.append([abs(float(Y[i])), float(X[i])])

        resp_item['datapoints'] = ceps
        resp.append(resp_item)

    else:
        resp_item['datapoints'] = []
        
    return resp
    
def osc_wavelet(query, wavelet_ID):
    """
    wave transform: ceps

    @param  query: (string) query string for influxdb.
    @return resp: (list) fft trans result in grafana timeseries format.
    """
    # grafana simple json response format
    target_name = 'Amplitude'
    resp = []
    resp_item = {
        'target': target_name,
        'datapoints': []    # data
    }


    result = client.query(query)    # query influxdb
    result = result.raw # trans query result to json


    x = []
    if 'series' in result:
        items = result['series'][0]['values']   # data array with data in influxdb
        
        for item in items:
            x.append(item[1])
    
    client.close()


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
        #wavelet_ID = req_param['wavelet_ID'][0]
        print('wavelet_ID:')
        print(wavelet_ID)
        #wavelet_ID=int(3)
        Y = Ys[wavelet_ID]
        wavelet = []
        for i in range(0,X.shape[0]):
            wavelet.append([float(Y[i]), float(X[i])])

        resp_item['datapoints'] = wavelet
        resp.append(resp_item)

    else:
        resp_item['datapoints'] = []
        
    return resp
    
def osc_envelope(query):
    """
    wave transform: envelope (temp)

    @param  query: (string) query string for influxdb.
    @return resp: (list) fft trans result in grafana timeseries format.
    """
    # grafana simple json response format
    target_name = 'Amplitude'
    resp = []
    resp_item = {
        'target': target_name,
        'datapoints': []    # data
    }


    result = client.query(query)    # query influxdb
    result = result.raw # trans query result to json


    x = []
    if 'series' in result:
        items = result['series'][0]['values']   # data array with data in influxdb
        
        for item in items:
            x.append(item[1])
    
    client.close()


    if len(x) != 0:
        # Compute and plot the spectrogram.
        Fs = 8192.0  # rate
        Ts = 1.0/Fs # interval
        ff = 5  # frequency of the signal

        n = len(x) # length of the signal
        k = np.arange(n)
        T = n/Fs
        X, Y = envelope_spectrum(x,Fs)


        envelope = []
        #for i in range(0,X.shape[0]):
        for i in range(0,int(100.0/X[1])):
            envelope.append([float(Y[i]), float(X[i])])

        resp_item['datapoints'] = envelope
        resp.append(resp_item)
    
    else:
        resp_item['datapoints'] = []


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

