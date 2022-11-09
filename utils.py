import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report,cohen_kappa_score
from scipy.stats import pearsonr,zscore
import tensorflow as tf
from tensorflow.keras import Model,Sequential, losses, optimizers, metrics, layers, initializers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,LearningRateScheduler,ReduceLROnPlateau
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
import seaborn as sns
import os,sys
import datacube
from datacube.utils.geometry import CRS,Geometry
from datacube.utils import geometry
from pyproj import Proj, transform
import fiona
from fiona.crs import from_epsg
import rasterio.features
import xarray as xr
from shapely.geometry import shape
from dea_tools.plotting import rgb

def geometry_mask(geoms, geobox, touches=False, invertion=False):
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=touches,
                                           invert=invertion)

def calculate_index(data, index):

    """
    Optical Indices Computation

    :param xarray: datacube_object
    :param string: you want to compute
    
    """
     
    if index.lower() == 'ndvi':
        B08 = data.B08.astype('float16')
        B04 = data.B04.astype('float16')
        return (B08 - B04) / (B08 + B04)
    if index.lower() == 'ndwi':
        B03 = data.B08.astype('float16')
        B08 = data.B04.astype('float16')
        return (B03 - B08) / (B08 + B03)
    if index.lower() == 'psri':
        B02 = data.B02.astype('float16')
        B06 = data.B06.astype('float16')
        B08 = data.B08.astype('float16')
        return (B04 - B02) / B06
    if index.lower() == 'savi':
        B08 = data.B08.astype('float16')
        B04 = data.B04.astype('float16')
        L = 0.428;
        return ((B08 - B04) / (B08 + B04 + L)) * (1.0 + L)
    else:
        return None
    
def cloud_data(data, index,fill_val=np.nan):
    
    """
    Cloud Masking Computation
    
    :param xarray: datacube_object
    :param index: you want to compute the cloud mask
    :param float: masking value (default:np.nan)
    
    """
    return xr.where((data.SCL>=4) & (data.SCL<=6), data[index.lower()], fill_val)

def getData(dc,product,geom,geom_buffer,startDate,endDate,bands=['B02','B03','B04','B08','SCL'],cloud_lim_to_keep=0.2):
    
    """
    return an xarray of the data you want

    :param string: product_name
    :param geom: rasterized geometry
    :param geom: buffered_rasterized geometry
    :param string: initial_date
    :param string: final_date
    :param list[bands]: list of bands to be returned from DC
    """

    
    query = {
        'geopolygon': geom_buffer,
        'time': (startDate,endDate),
        'product': product
        }
    
    data = dc.load(output_crs="EPSG:3857",measurements=bands,resolution=(-10,10),**query,dask_chunks={})
    if len(data) == 0:
        return -1
    
    #ndvi calculation and masking of clouds only this index
    data['ndvi'] = calculate_index(data,'ndvi')
    
    data['ndvi'] = cloud_data(data,'ndvi')
    mask = geometry_mask([geom], data.geobox, invertion=True)
    
    #masked data of ndvi based on input geometry
    masked_data = data.where(mask)
    
    #keep only "clearsky" time instances (real values of higher than a threshold)
    pixels_lim = int(mask.sum()*cloud_lim_to_keep)
    d = xr.where((masked_data.SCL>=4) & (masked_data.SCL<=6) & (masked_data.ndvi>0), masked_data,np.nan)
    to_keep = d.dropna(dim='time',thresh=pixels_lim).time

    data = data.sel(time=to_keep)
    masked_data = masked_data.sel(time=to_keep)
    
    tt =  np.array([np.datetime64(datetime.strptime(str(k),'%Y-%m-%dT%H:%M:%S.%f000').strftime('%Y-%m-%d'))for k in data.time.values])
    data = data.assign_coords(time=('time',tt))
    masked_data = masked_data.assign_coords(time=('time',tt))
    #drop duplicates
    tt = np.unique(tt,return_index=True)[1]
    data = data.isel(time=tt)
    masked_data = masked_data.isel(time=tt)

    data = data.load()
    masked_data = masked_data.load()
    
    return data,masked_data


def datacube_parcel(case,gdf,d_start,d_end,path,buffer=100,figsize=(12,8)):
    
    
    gdf_f = gdf.iloc[case:case+1].copy()
    home_directory = '/home/eouser'
    # os.chdir(home_directory)


    dc = datacube.Datacube(app="test",config=os.path.join(home_directory,"datacube.conf"))
    all_optical_bands = ['B02','B03','B04','B08','SCL']
    product= 's2_preprocessed_lithuania'

    # parcel ids to examine. This list can be retrieved with a simple query from the db as following:
    ### select ids,geom from parcel where parcel.declaration_crop_id != parcel.prediction_crop_id order by parcel.confidence desc


    # open file and get geometry
    ds = fiona.open(path)
    crs = geometry.CRS(ds.crs_wkt)
    f = ds[case]
    
    # get attributes
    unique_id = f['properties']['parcel_id']
    area = f['properties']['Area']
    # region = f['properties']['region']
    feature_geom = f['geometry']
    geom = Geometry(feature_geom,crs)


    geom_buffer = geom.buffer(buffer)  
    bounds = shape(feature_geom).bounds
    s2,s2_mask = getData(dc,product,geom,geom_buffer,d_start,d_end,bands=all_optical_bands)
    dates_str = np.array([str(d).split('T')[0] for d in s2_mask.time.values])

    # sar = getData_sar(product_sar,geom,d_start,d_end)
    # sar_dates_str = np.array([str(d).split('T')[0] for d in sar.time.values])

    pseudo_col_1 = ["B04", "B03", "B02"]
    pseudo_col_2 = ["ndvi"]
    col_n = 6

    # check if an additional row is needed for the plot
    if len(dates_str)%col_n==0:
        row_n = len(dates_str)//col_n
    else:
        row_n = len(dates_str)//col_n + 1

    fig1, ax1 = plt.subplots(row_n,col_n, figsize=figsize)

    n = 0 

    stop = False

    for i in range(row_n):
        for j in range(col_n):

            gdf_f.geometry.to_crs("EPSG:3857").plot(ax=ax1[i][j],facecolor='none',edgecolor='red',linewidth=3)
            s2[pseudo_col_1].isel(time=n).to_array().plot.imshow(ax=ax1[i][j], robust=True, add_labels=False)

            ax1[i][j].set_title(dates_str[n],fontsize=15)
            ax1[i][j].axis('off')

            n += 1
            if n==len(dates_str):
                stop = True
                break
        if stop:
            break

    #         fig.delaxes(ax[-1][-1])
    fig1.suptitle('ID: {} \n Area: {} hec'.format(unique_id,area),fontsize=21,y=1.000001)
    plt.tight_layout()




def plot_parcel(dates,mowing_events_photo,y,sar_vv,insar_vv,y_when):   
    
    fig, ax = plt.subplots(figsize=(20,5))
    fig.subplots_adjust(right=0.75)
    
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.1))
    
    p1, = ax.plot(dates,y,label='NDVI',marker='s',lw=2,ls='--',ms=7,color='darkgreen',zorder=-1)
    p2, = twin1.plot(dates,sar_vv*(-30),label='SAR_VV',marker='d',lw=1,ms=7,color='firebrick')
    p3, = twin2.plot(dates,insar_vv,label='INSAR_VV',marker='x',lw=1,ms=7,color='navy')

    for d in mowing_events_photo:
        plt.axvspan(d[0],d[1],ymin=0,ymax=1,color='orange',hatch="//",alpha=0.5)

           
    events = np.where(y_when==1)[0]
    for n in events:
        plt.vlines(dates[n],ymin=0,ymax=1,color='green',lw=8,alpha=0.7,label='model')
        
    ax.set_xlim(dates[0],dates[-1])
    ax.set_ylim(0, 1)
    twin1.set_ylim(-18, -6)
    twin2.set_ylim(0, 1)
    ax.set_xlabel(" ")
    ax.set_ylabel("NDVI")
    twin1.set_ylabel("Sigma0 VV (dB)")
    twin2.set_ylabel("Coherence VV")

    ax.tick_params(axis='both', which='major', labelsize=22)
    twin1.tick_params(axis='y', which='major', labelsize=20)
    twin2.tick_params(axis='y', which='major', labelsize=20)

    ax.set_ylabel(r'$NDVI$',fontsize=26)
    twin1.set_ylabel(r'$\sigma_{0} \ VV \ (dB)$',fontsize=24)
    twin2.set_ylabel('$coherence \ VV$',fontsize=24)

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())


    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    dtFmt = mdates.DateFormatter('%b-%Y') # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt) 
    # show every 12th tick on x axes
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45,fontsize=25,fontweight='medium')

    plt.grid(ls='--')
    plt.tight_layout()
    plt.show()
    
    
def add_cnn_block_1D(x_inp,filters,kernel_size=3,padding="same",strides=1,r=True):
    

    x = layers.Conv1D(filters,kernel_size,padding=padding, strides=strides,
                      kernel_initializer=initializers.glorot_normal())(x_inp)
    x = layers.Activation('relu')(x)

    return x

def attention_seq(query_value, scale):

    query, value = query_value
    score = tf.matmul(query, value, transpose_b=True) # (batch, timestamp, 1)
    score = scale*score # scale with a fixed number (it can be finetuned or learned during train)
    score = tf.nn.softmax(score, axis=1) # softmax on timestamp axis
    score = score*query # (batch, timestamp, feat)
    return score

    

def Conv_RNN_mowing_model_when(n_vars,n_timesteps,lstm_units):

    inputs = list([])
    k = list([])

    var = [str(k+1) for k in range(n_vars)]
    for v in var:
        x_inp = layers.Input(shape=(n_timesteps,1),name='{}_input'.format(v))
        inputs.append(x_inp)
        x = add_cnn_block_1D(x_inp,filters=16,r=False)
        x = add_cnn_block_1D(x,filters=16,r=False)
        x = layers.Dropout(0.2)(x)
#         x = layers.MaxPooling1D(pool_size=2,strides=None)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal())(x)
        x = layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal())(x)
        k.append(x)

    m = layers.Concatenate()(k)
    m = layers.RepeatVector(lstm_units)(m)
    
    m_2 = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu'))(m)
    m_2 = layers.RepeatVector(n_timesteps)(m_2)
    m_2 = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu',return_sequences=True))(m_2)
 
    m_2 = layers.TimeDistributed(layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal()))(m_2)
    m_2 = layers.TimeDistributed(layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal()))(m_2)
    out_2 = layers.TimeDistributed(layers.Dense(1,activation='sigmoid',kernel_initializer=initializers.glorot_normal()),name='when_out')(m_2)
    
    return Model(inputs=inputs, outputs=out_2)


def Conv_RNN_mowing_model_when_s2(n_timesteps,lstm_units):


    x_inp = layers.Input(shape=(n_timesteps,1),name='{}_input'.format('ndvi'))
    x = add_cnn_block_1D(x_inp,filters=16,r=False)
    x = add_cnn_block_1D(x,filters=16,r=False)
    x = layers.Dropout(0.2)(x)
#         x = layers.MaxPooling1D(pool_size=2,strides=None)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal())(x)
    x = layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal())(x)

    m = layers.RepeatVector(lstm_units)(x)
    
    m_2 = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu'))(m)
    m_2 = layers.RepeatVector(n_timesteps)(m_2)
    m_2 = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu',return_sequences=True))(m_2)
 
    m_2 = layers.TimeDistributed(layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal()))(m_2)
    m_2 = layers.TimeDistributed(layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal()))(m_2)
    out_2 = layers.TimeDistributed(layers.Dense(1,activation='sigmoid',kernel_initializer=initializers.glorot_normal()),name='when_out')(m_2)
    
    return Model(inputs=x_inp, outputs=out_2)


                                             
def Conv_RNN_mowing_model_when_attention(n_vars,n_timesteps,lstm_units):
    inputs = list([])
    k = list([])

    var = [str(k+1) for k in range(n_vars)]
    for v in var:
        x_inp = layers.Input(shape=(n_timesteps,1),name='{}_input'.format(v))
        inputs.append(x_inp)
        x = add_cnn_block_1D(x_inp,filters=16,r=False)
        x = add_cnn_block_1D(x,filters=16,r=False)
        x = layers.Dropout(0.2)(x)
#         x = layers.MaxPooling1D(pool_size=2,strides=None)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal())(x)
        x = layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal())(x)
        k.append(x)

    m = layers.Concatenate()(k)
    m = layers.RepeatVector(lstm_units)(m)
    
#     seq,state_1,state_2,_,_ = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu',return_sequences=True,return_state=True))(m)
#     att = tf.keras.layers.Lambda(attention_seq, arguments={'scale': 0.01})([seq, tf.expand_dims(tf.concat((state_1,state_2),axis=1),1)])
#     m_2 = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu',return_sequences=True))(att)

    seq,state,_ = layers.LSTM(lstm_units, activation='relu',return_sequences=True,return_state=True)(m)
    att = tf.keras.layers.Lambda(attention_seq, arguments={'scale': 0.01})([seq, tf.expand_dims(state,1)])
    m_2 = layers.LSTM(lstm_units, activation='relu',return_sequences=True)(att)
    
    m_2 = layers.TimeDistributed(layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal()))(m_2)
    m_2 = layers.TimeDistributed(layers.Dense(64,activation='relu',kernel_initializer=initializers.glorot_normal()))(m_2)
    out_2 = layers.TimeDistributed(layers.Dense(1,activation='sigmoid',kernel_initializer=initializers.glorot_normal()),name='when_out')(m_2)
    
    return Model(inputs=inputs, outputs=out_2)    




def f1_when(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)



def results_refinement(preds_clf,pixel_id,proximity,dates):

    results_mowing_a = {}

    for n,case in enumerate(pixel_id):
        events_ii = np.where(preds_clf[n]==1)[0]
        events_i = []
        for i in events_ii:
            events_i.append((dates[i],dates[i]+timedelta(days=6)))
        results_mowing_a[case] = events_i

    results_mowing = {}

    for case in results_mowing_a.keys():
        case_i = results_mowing_a[case]
        e = []
        mow_n = len(case_i)
        for j in range(mow_n):
            e_i = (case_i[j][0],case_i[j][1])
            if (j>0):
                if (e_i[0].dayofyear-e[-1][0].dayofyear<proximity):
                    continue
                else:
                    e.append(e_i)
            else:
                e.append(e_i)
        results_mowing[case] = e

    return results_mowing

def mowing_performance(results_mowing,ground_truth,tolerance,pixel_lvl=True):
    
    
    pixel_id = sorted(list(results_mowing.keys()))
    
    
    predictions_mowing = {}
    for c in pixel_id:
        if pixel_lvl:
            c_1 = '-'.join(c.split('-')[:-1])
        else:
            c_1 = c
        case_i = results_mowing[c]
        try:
            if len(case_i)==0:
                predictions_mowing[c] = np.nan
                continue
            if len(case_i)==1:
                dd = pd.date_range(case_i[0][0],case_i[0][1])
    #             predictions_fusion[c] = dd[round(len(dd)/2)]
                predictions_mowing[c] = case_i[0][0]
            else:
                l = []
                for case_sub in case_i:
                    dd = pd.date_range(case_sub[0],case_sub[1])
    #                 l.append(dd[round(len(dd)/2)])
                    l.append(case_sub[0])
                predictions_mowing[c] = l[np.argmin([abs((l_i-ground_truth[c_1]).days) for l_i in l])]
        except:
            predictions_mowing[c] = np.nan
            continue
    
    score_mowing = []
    x_y_mowing = []
    for c in pixel_id:
        if pixel_lvl:
            c_1 = '-'.join(c.split('-')[:-1])
        else:
            c_1 = c
        try:
            score_mowing.append((predictions_mowing[c]-ground_truth[c_1]).days)
            if np.abs((predictions_mowing[c]-ground_truth[c_1]).days) < tolerance:
                x_y_mowing.append((ground_truth[c_1].dayofyear,predictions_mowing[c].dayofyear))
        except:
            score_mowing.append(np.nan)
    score_mowing = np.array(score_mowing)
    x_mowing = [x[0] for x in x_y_mowing]
    y_mowing = [x[1] for x in x_y_mowing]
    
    recall_mowing = {}
    recall_mowing = score_mowing[np.abs(score_mowing)<tolerance].shape[0]/len(pixel_id)

    correct_mowing = 0
    all_mowing = 0 
    for n,case in enumerate(pixel_id):
        if np.abs(score_mowing[n])<tolerance:
            correct_mowing += 1
        all_mowing += len(results_mowing[case])
    precision_mowing = correct_mowing/all_mowing      

    support = int(len(predictions_mowing))
    recall = round(recall_mowing,3)
    precision = round(precision_mowing,3)
    if (recall+precision)!=0:
        f1 = round((2*recall*precision)/(recall+precision),3)
    else:
        f1 = np.nan
    me = round(score_mowing[np.abs(score_mowing)<tolerance].mean(),3)
    mae = round(np.abs(score_mowing[np.abs(score_mowing)<tolerance]).mean(),3)
    r2 = round(np.corrcoef(x_mowing, y_mowing)[0,1],3)

    print('support: {}'.format(support))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))
    print('f1_score: {}'.format(f1))
    print('ME: {}'.format(me))
    print('MAE: {}'.format(mae))
    print('R^2: {}'.format(r2))
    
    m = [support,recall,precision,f1,me,mae,r2]
    
    return m

def results_aggr(results_dict,ground_truth,perc_tolerance=0.5,tolerance=6):

    pixel_id = np.array(sorted(list(results_dict.keys())))
    parcel_id = np.array(['-'.join(c.split('-')[:-1]) for c in pixel_id])
    
    results_mowing = {c:[] for c in sorted(set(parcel_id))}
    for c in pixel_id:
        parcel_i = '-'.join(c.split('-')[:-1])
        results_mowing[parcel_i].extend(results_dict[c])
    results_mowing = {c_key:[(c[0].month,c) for c in results_mowing[c_key]] for c_key in results_mowing.keys()}

    rr = {}
    for c in sorted(set(parcel_id)):

        cc = list(Counter(results_mowing[c]).items())
        len_pixels = len(pixel_id[np.where(parcel_id==c)[0]])
        df_i = pd.DataFrame(columns=['month','tmstmp','vals'],index=np.arange(len(cc)))
        for i in range(len(cc)):
            df_i.loc[i,'month'] = cc[i][0][0]
            df_i.loc[i,'tmstmp'] = cc[i][0][1]
            df_i.loc[i,'vals'] = cc[i][1]

        p = []
        try:
            sums = df_i.groupby('month').sum()
            most_commons = df_i.groupby('month')['tmstmp'].agg(lambda x:x.value_counts().index[0])
            for i in range(len(sums)):
                if sums.iloc[i].vals>=int(perc_tolerance*len_pixels):
                    p.append(most_commons.iloc[i])
        except:
            pass 
        rr[c] = sorted(p)
        
    results_mowing = rr

    predictions = {}
    for c in sorted(set(parcel_id)):
        case_i = results_mowing[c]
        try:
            if len(case_i)==0:
                predictions[c] = np.nan
                continue
            if len(case_i)==1:
                dd = pd.date_range(case_i[0][0],case_i[0][1])
    #             predictions_fusion_s2[c] = dd[round(len(dd)/2)]
                predictions[c] = case_i[0][0]
            else:
                l = []
                for case_sub in case_i:
                    dd = pd.date_range(case_sub[0],case_sub[1])
    #                 l.append(dd[round(len(dd)/2)])
                    l.append(case_sub[0])
                predictions[c] = l[np.argmin([abs((l_i-ground_truth[c]).days) for l_i in l])]
        except:
            predictions[c] = np.nan
            continue

    score = []
    x_y = []
    for c in sorted(set(parcel_id)):
        try:
            score.append((predictions[c]-ground_truth[c]).days)
            if np.abs((predictions[c]-ground_truth[c]).days) < tolerance:
                x_y.append((ground_truth[c].dayofyear,predictions[c].dayofyear))
        except:
            score.append(np.nan)
    score = np.array(score)
    x = [x[0] for x in x_y]
    y = [x[1] for x in x_y]


    recall = {}
    recall = score[np.abs(score)<tolerance].shape[0]/len(set(parcel_id))

    correct = 0
    all = 0 

    for n,case in enumerate(sorted(set(parcel_id))):
        if np.abs(score[n])<tolerance:
            correct += 1
        all += len(results_mowing[case])

    precision = correct/all      

    support = int(len(predictions))
    recall = round(recall,3)
    precision = round(precision,3)
    if (recall+precision)!=0:
        f1 = round((2*recall*precision)/(recall+precision),3)
    else:
        f1 = np.nan
    me = round(score[np.abs(score)<tolerance].mean(),3)
    mae = round(np.abs(score[np.abs(score)<tolerance]).mean(),3)
    r2 = round(np.corrcoef(x, y)[0,1],3)

    print('support: {}'.format(support))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))
    print('f1_score: {}'.format(f1))
    print('ME: {}'.format(me))
    print('MAE: {}'.format(mae))
    print('R^2: {}'.format(r2))
   
    m = [support,recall,precision,f1,me,mae,r2]
    
    return results_mowing,m