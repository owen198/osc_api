'''
Created on 2019年1月23日

@author: I28564
'''

if __name__ == '__main__':
    pass
measurement = 'KKKKKK'
tagValue = 'JJJJJ'
time_start='0'
time_end='100'
wavelet_ID='1'
query_constraint='query_constraint'
query_from = 'select v from "' + measurement + '" where p=' + "'" + tagValue + "' and wavelet_ID='" + wavelet_ID + "'"  
query_limit = ' limit 24576'    # limit of raw data
query_time = "and time>= now()-5s"

    # combine query string
query = query_from + query_time
if query_constraint != '':
        query =  query + ' AND ' + query_constraint
    
query = query + query_limit

print(query)
print(measurement)
print(tagValue)
