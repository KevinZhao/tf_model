import talib 
import pandas as pd
import numpy as np

from Talib_calc import *

def fix_data(path):
    tmp = pd.read_csv(path, encoding="gbk", engine='python')
    tmp.rename(columns={'Unnamed: 0':'trading_time'}, inplace=True)
    tmp['trading_point'] = pd.to_datetime(tmp.trading_time)
    del tmp['trading_time']
    tmp.set_index(tmp.trading_point, inplace=True)
    return tmp.tail(800)

def High_2_Low(tmp, freq):
    # 分别处理bar数据
    tmp_open = tmp['open'].resample(freq).ohlc()
    tmp_open = tmp_open['open'].dropna()

    tmp_high = tmp['high'].resample(freq).ohlc()
    tmp_high = tmp_high['high'].dropna()

    tmp_low = tmp['low'].resample(freq).ohlc()
    tmp_low = tmp_low['low'].dropna()

    tmp_close = tmp['close'].resample(freq).ohlc()
    tmp_close = tmp_close['close'].dropna()

    tmp_price = pd.concat([tmp_open, tmp_high, tmp_low, tmp_close], axis=1)
    
    # 处理成交量
    tmp_volume = tmp['volume'].resample(freq).sum()
    tmp_volume.dropna(inplace=True)
    
    return pd.concat([tmp_price, tmp_volume], axis=1)

def load_data(path):

    tmp = fix_data(path)

    tmp_1d = High_2_Low(tmp, '1d')
    rolling = 88
    targets = tmp_1d
    #两日后盈亏
    targets['returns'] =  targets['close'].shift(-2) / targets['close'] - 1.0
    #88日平均值 + 0.5 * 盈亏平均值
    targets['upper_boundary']= targets.returns.rolling(rolling).mean() + 0.5 * targets.returns.rolling(rolling).std()
    targets['lower_boundary']= targets.returns.rolling(rolling).mean() - 0.5 * targets.returns.rolling(rolling).std()

    targets.dropna(inplace=True)
    targets['labels'] = 1
    targets.loc[targets['returns']>=targets['upper_boundary'], 'labels'] = 2
    targets.loc[targets['returns']<=targets['lower_boundary'], 'labels'] = 0
    
    # factors 1d 数据合成
    tmp_1d = High_2_Low(tmp, '1d')
    factors = get_factors(tmp_1d.index, tmp_1d.open.values, tmp_1d.close.values, tmp_1d.high.values, tmp_1d.low.values, tmp_1d.volume.values, rolling = 26, drop=True)
    factors = factors.loc[:targets.index[-1]]
    
    tmp_factors_1 = factors.iloc[:12]
    targets = targets.loc[tmp_factors_1.index[-1]:]

    gather_list = np.arange(factors.shape[0])[11:]
    
    inputs = np.array(factors).reshape(-1, 1, factors.shape[1])

    def dense_to_one_hot(labels_dense):
        """标签 转换one hot 编码
        输入labels_dense 必须为非负数
        2016-11-21
        """
        num_classes = len(np.unique(labels_dense)) # np.unique 去掉重复函数
        raws_labels = labels_dense.shape[0]
        index_offset = np.arange(raws_labels) * num_classes
        labels_one_hot = np.zeros((raws_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot  

    targets = dense_to_one_hot(targets['labels'])
    targets = np.expand_dims(targets, axis=1)

    return inputs, targets, gather_list