import numpy as np
import pandas as pd
import pywt
import glob
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, peak_prominences
from tsfresh import extract_features


# TODO: Include logging

def suppress_sine(df_sig, cutoff=2500, measurements=800000,
                  time_length=0.02, filter_order=4):
    """
    Suppress sinusoidal component (50Hz) of signals using butterworth filter. 2.5kHz high-pass 4th order filter applied
    from the left and from the right (scipy.signal.filtfilt) worked well empirically.

    input_df: train or test parquet file loaded into dataframe (pandas.DataFrame)

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
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
              .format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def main():

    # TODO: Put read and write functions into try-except statements
    test = pd.read_csv('../data/metadata_test.csv')
    # test_sig = pd.read_parquet('../data/test.parquet')
    test_column_start = 8712
    test_column_end = 29048
    chunks = 30
    stepsize = int(np.ceil((test_column_end + 1 - test_column_start) / chunks))
    version = "1.1"

    fc = {'symmetry_looking': [{'r': 1}],
          'mean': None,
          'number_peaks': [{'n': 100}],
          'first_location_of_maximum': None,
          'range_count': [{'min': 10, 'max': 15},
                          {'min': -15, 'max': -10}],
          }

    test_chunk = 1

    for column in range(test_column_start + stepsize *
                        (test_chunk - 1), test_column_end + 1, stepsize):
        print('Starting chunk processing {}'.format(test_chunk))
        signals = [str(cols) for cols in range(
            column, column + stepsize) if cols <= test_column_end]
        # print('Number of signals being processed: {}'.format(len(signals)))
        # test_sig_chunk = test_sig[signals]
        # print('Shape of test chunk: {}'.format(test_sig_chunk.shape))
        test_meta = test[test['signal_id'].isin(signals)]
        print('Shape of test_meta chunk: {}'.format(test_meta.shape))
        print('Loaded test data')

        df = pd.read_parquet(
            '../output/test_chunks/2500Hz_1.0/test_sig_2500hz_4th_order_dwt_' +
            str(test_chunk) +
            '.parquet')

        df = tsfresh_formatting(df, test_meta)
        # df.to_csv('../output/test_sig_2500Hz_4th_order_dwt_tsfresh_format_' + str(test_chunk) + '.csv', index=False)
        print('Converted to flat format')
        print('DataFrame shape: {}'.format(df.shape))

        df = reduce_mem_usage(df)
        print('DataFrame shape: {}'.format(df.shape))

        tsfresh_df = extract_features(
            df,
            column_id='signal_id',
            column_value='amplitude',
            default_fc_parameters=fc)
        # df.to_csv('../output/test_sig_2500Hz_with_ts_features_' + str(test_chunk) + '.csv', index=False)
        print('Extracted tsfresh features')

        peak_indices = [find_peaks(df[df['signal_id'] == signal] .reset_index(
            drop=True)['amplitude']) for signal in df.signal_id.unique()]
        peak_width = np.asarray([peak_widths(df[df['signal_id'] == signal].reset_index(drop=True)['amplitude'],
                                             peak_indices[col_num][0]) for col_num, signal in
                                 enumerate(zip(df.signal_id.unique()))])
        peak_height = np.asarray([peak_prominences(df[df['signal_id'] == signal].reset_index(drop=True)['amplitude'],
                                                   peak_indices[col_num][0]) for col_num, signal in
                                  enumerate(zip(df.signal_id.unique()))])

        tsfresh_df['max_peak_width'] = np.asarray(
            [np.max(peak_width[:, 0][x]) for x in range(peak_width.shape[0])])
        tsfresh_df['min_peak_width'] = np.asarray(
            [np.min(peak_width[:, 0][x]) for x in range(peak_width.shape[0])])
        tsfresh_df['mean_peak_width'] = np.asarray(
            [np.mean(peak_width[:, 0][x]) for x in range(peak_width.shape[0])])
        tsfresh_df['max_peak_height'] = np.asarray(
            [np.max(peak_height[:, 0][x]) for x in range(peak_height.shape[0])])
        tsfresh_df['min_peak_height'] = np.asarray(
            [np.min(peak_height[:, 0][x]) for x in range(peak_height.shape[0])])
        tsfresh_df['mean_peak_height'] = np.asarray(
            [np.mean(peak_height[:, 0][x]) for x in range(peak_height.shape[0])])

        pd.merge(
            tsfresh_df.reset_index().rename(
                columns={
                    'id': 'signal_id'}),
            test_meta,
            on='signal_id').to_csv(
                '../output/test_chunks/2500Hz_1.1/processed_test_2500Hz_chunk_' +
                str(test_chunk) +
                '.csv',
            index=False)
        del df, tsfresh_df, test_meta
        print('Saved processed chunk')
        print('_____________________')

        test_chunk += 1

    test_chunks = glob.glob(
        "../output/test_chunks/2500Hz_1.1/processed_test_2500Hz_chunk_*.csv")
    processed_test = pd.concat((pd.read_csv(processed_chunk)
                                for processed_chunk in test_chunks))
    processed_test = processed_test.sort_values('signal_id')
    processed_test = processed_test.drop(['id_measurement', 'phase'], axis=1)
    processed_test.to_csv('../data/processed_test_2500Hz_' +
                          version + '.csv', index_label=False)


if __name__ == '__main__':
    main()
