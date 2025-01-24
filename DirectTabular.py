import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

data= pd.read_csv("Nongfu.csv")



#5月是训练集，剩下的是测试集
#May is the training set and the rest is the test set
train_end_point=744
test_start_point=744

#预测时间长度
#set the forecast horizon
forecasthorizon=24


train_data = TimeSeriesDataFrame.from_data_frame(data.iloc[0:train_end_point,:],id_column="item_id",timestamp_column="timestamp")
predictor = TimeSeriesPredictor(prediction_length=forecasthorizon,path="autogluon-m4-hourly",target="target",eval_metric="MASE")


predictor.fit(train_data,presets="best_quality",num_val_windows=10)



#执行forecast_group_num次预测
#Perform forecast_group_num sub-predictions
#6月1日到8月20日一共1944个数据点
#A total of 1944 data points from June 1 through August 20
forecast_group_num=int(1944/forecasthorizon)   
train_data1=TimeSeriesDataFrame.from_data_frame(data.iloc[test_start_point-train_end_point:test_start_point,:],id_column="item_id",timestamp_column="timestamp")
predictions = predictor.predict(train_data1,model="DirectTabular")

for j in range(1,forecast_group_num):
    print(j)
    train_data1= TimeSeriesDataFrame.from_data_frame(data.iloc[test_start_point-train_end_point+forecasthorizon*j:test_start_point+forecasthorizon*j,:],id_column="item_id",timestamp_column="timestamp")
    predictions1 = predictor.predict(train_data1,model="DirectTabular")
 
    predictions= pd.concat([predictions,predictions1], axis=0)

#测试集的预测值就是predictions
#The forecast values of the test set are the predictions

forecastvalues=predictions



