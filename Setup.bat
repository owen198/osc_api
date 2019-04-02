cf push -f manifest.yml --no-start
cf set-env api-csc-fft-history SrpName dashboard
cf set-env api-csc-fft-history group_name g_grafana_dashboard
cf set-env api-csc-fft-history org_id 7cd7fb10-476e-40cd-bd02-d29df729308b
cf set-env api-csc-fft-history org_name csc
cf set-env api-csc-fft-history influxdb_service_name CSC-realtime
cf set-env api-csc-fft-history schema_name grafana_dashboard
cf set-env api-csc-fft-history sso-url
cf set-env api-csc-fft-history menu_logo http://dashboard-csc-srp.fomos.csc.com.tw/api/images/getImage?imageId=1
cf set-env api-csc-fft-history login_logo
cf bind-service api-csc-fft-history influxdb -c "{\"group\":\"g_grafana_dashboard\"}"
cf start api-csc-fft-history
pause