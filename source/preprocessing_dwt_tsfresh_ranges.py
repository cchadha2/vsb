import pandas as pd
import numpy as np
import pywt
import glob
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, peak_prominences
from tsfresh import extract_features
from functools import reduce


def train_sampling(df_sig, df_meta, output_path, output_meta, seed=1001):
    """
    Under sample negative samples (that do not have any PD) at a 1:1 ratio with positive samples.

    df_sig: train signals (pandas.DataFrame)
    df_meta: train metadata (pandas.DataFrame)
    output_path: path to save parquet file of sine suppressed train or test signals (str)
    output_meta: path to save metadata of sine suppressed train or test signals (str)

    :returns train_sig_sampled (pandas.DataFrame)
    """

    # Note: Could use SMOTE to handle undersampling negative samples while
    # oversampling positive samples
    positive_samples = df_meta[df_meta['target'] == 1]
    num_negative_samples = positive_samples.shape[0]
    negative_samples = df_meta[df_meta['target'] == 0].sample(
        num_negative_samples, random_state=seed)

    train_sampled = pd.concat([positive_samples, negative_samples])
    train_sampled.to_csv(output_meta)

    samples = train_sampled['signal_id'].values.tolist()
    samples = [str(x) for x in samples]

    train_sig_sampled = df_sig[samples]
    train_sig_sampled.to_parquet(output_path)

    return train_sig_sampled


def suppress_sine(df_sig, cutoff=2500, measurements=800000,
                  time_length=0.02, filter_order=4):
    """
    Suppress sinusoidal component (50Hz) of signals using butterworth filter. 2.5kHz high-pass 4th order filter applied
    from the left and from the right (scipy.signal.filtfilt) worked well empirically.

    input_df: train or test parquet file loaded into dataframe (pandas.DataFrame)
    output_path: path to save parquet file of sine suppressed train or test signals (str)

    :returns filtered_sig_df (pandas.DataFrame)
    """

    sampling_rate = measurements / time_length
    nyquist = sampling_rate * 0.5
    wn = cutoff / nyquist

    b, a = butter(filter_order, wn, btype='highpass')

    filtered_sig = filtfilt(b, a, df_sig.values, axis=0)
    filtered_sig_df = pd.DataFrame(
        data=filtered_sig, columns=df_sig.columns.tolist())

    return filtered_sig_df


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise_signal(df_sig, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(df_sig, wavelet, mode="per")

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute
    # Deviation not Median Absolute Deviation
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(df_sig)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def tsfresh_formatting(df_sig, df_meta):

    amplitude = pd.DataFrame(df_sig.stack().reset_index().drop(
        'level_0', axis=1).rename(columns={'level_1': 'signal_id'}))
    amplitude['signal_id'] = amplitude['signal_id'].astype('int')

    signal_id = pd.DataFrame(df_meta['signal_id']).reset_index(drop=True)
    tsfresh_df = pd.merge(signal_id, amplitude, how='inner').rename(
        columns={0: 'amplitude'})

    return tsfresh_df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def main():

    train_meta = pd.read_csv('../output/train_sampled.csv')
    train_meta = train_meta.drop(['id_measurement', 'phase'], axis=1)
    print('Meta data shape: {}'.format(train_meta.shape))
    version = "1.2"

    fc = {
          'mean': None,
          'number_peaks': [{'n': 100}],
          'first_location_of_maximum': None,
          'range_count': [{'min': 10, 'max': 15},
                          {'min': -15, 'max': -10}],
          }

    dfs = pd.read_csv('../output/train_sig_2500Hz_4th_order_dwt_tsfresh_format.csv', chunksize=800000)
    train_chunk = 1

    processed_train = pd.DataFrame()

    for df in dfs:

        print('Train chunk: {}'.format(train_chunk))

        ranges = 1
        stepsize = 200000

        df_chunk = pd.DataFrame()

        for rows in range(0,800000,stepsize):
            df_range = df[rows:rows+stepsize]

            df_range = reduce_mem_usage(df_range)

            tsfresh_df = extract_features(
                df_range,
                column_id='signal_id',
                column_value='amplitude',
                default_fc_parameters=fc)
            print('Extracted tsfresh features')
            print(tsfresh_df.head())

            tsfresh_df.columns = [column + '_' + str(ranges) for column in tsfresh_df.columns.tolist()]

            peak_indices = [find_peaks(df_range[df_range['signal_id'] == signal] .reset_index(
                drop=True)['amplitude']) for signal in df_range.signal_id.unique()]
            peak_width = np.asarray([peak_widths(df_range[df_range['signal_id'] == signal].reset_index(drop=True)['amplitude'],
                                                 peak_indices[col_num][0]) for col_num, signal in enumerate(zip(df_range.signal_id.unique()))])
            peak_height = np.asarray([peak_prominences(df_range[df_range['signal_id'] == signal].reset_index(drop=True)[
                                     'amplitude'], peak_indices[col_num][0]) for col_num, signal in enumerate(zip(df_range.signal_id.unique()))])

            tsfresh_df['max_peak_width_' + str(ranges)] = np.asarray(
                [np.max(peak_width[:, 0][x]) for x in range(peak_width.shape[0])])
            tsfresh_df['min_peak_width_' + str(ranges)] = np.asarray(
                [np.min(peak_width[:, 0][x]) for x in range(peak_width.shape[0])])
            tsfresh_df['mean_peak_width_' + str(ranges)] = np.asarray(
                [np.mean(peak_width[:, 0][x]) for x in range(peak_width.shape[0])])
            tsfresh_df['max_peak_height_' + str(ranges)] = np.asarray(
                [np.max(peak_height[:, 0][x]) for x in range(peak_height.shape[0])])
            tsfresh_df['min_peak_height_' + str(ranges)] = np.asarray(
                [np.min(peak_height[:, 0][x]) for x in range(peak_height.shape[0])])
            tsfresh_df['mean_peak_height_' + str(ranges)] = np.asarray(
                [np.mean(peak_height[:, 0][x]) for x in range(peak_height.shape[0])])

            df_range = pd.merge(tsfresh_df.reset_index().rename(columns={'id': 'signal_id'}), train_meta, on='signal_id')


            df_range = df_range.rename(columns={'signal_id': 'signal_id_' + str(ranges),
                                                'target': 'target_' + str(ranges)})

            df_chunk = pd.concat([df_chunk, df_range], axis=1)

            ranges += 1


        df_chunk = df_chunk.rename(columns={'signal_id_1': 'signal_id','target_1': 'target'})\
                           .drop(['signal_id_2', 'signal_id_3', 'signal_id_4', 'target_2', 'target_3', 'target_4'],
                                 axis=1)

        processed_train = pd.concat([processed_train, df_chunk])
        print(processed_train.head())



        train_chunk += 1

    processed_train.to_csv('../data/processed_train_2500Hz_' + version + '.csv', index_label=False)
    print('Saved processed training data')


if __name__ == '__main__':
    main()
