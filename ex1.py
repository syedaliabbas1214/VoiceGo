from time import  time
from time import sleep
from datetime import datetime
import sounddevice as sd
import os
import argparse
import psutil as ps
import uuid
import redis
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import zipfile
import psutil

#%% #parser
parser = argparse.ArgumentParser()
parser.add_argument('--device', type = int, default=1)
parser.add_argument('--host', type=str, default="redis-19958.c246.us-east-1-4.ec2.cloud.redislabs.com")
parser.add_argument('--port', type=int, default=19958)
parser.add_argument('--user', type=str, default="default")
parser.add_argument('--password', type=str, default="ixQZhyT2TmuE12NAxb7MyQINpRiFYVYx")
parser.add_argument('--delete', type=int, default=0)  # debug
parser.add_argument('--verbose', type=int, default=0)  # debug
parser.add_argument('--device', type = int, default=1)

#%% #redis
args = parser.parse_args()
# Recording parameters
DEVICE = args.device
CHANNELS = 1
DTYPE = 'int16'
AUDIO_FILE_LENGTH_IN_S = 1

REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USER = args.user
REDIS_PASSWORD = args.password

redis_client = redis.Redis(host = REDIS_HOST, port = REDIS_PORT, username = REDIS_USER, password = REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

mac_address = hex(uuid.getnode())
mac_battery = f'{mac_address}:battery'
mac_power = f'{mac_address}:power'
mac_pluged_seconds = f'{mac_address}:mac_pluged_seconds'

# Create a timeseries named 'integers'
mac_battery = str(hex(uuid.getnode())) + ":battery"   # creating time series mac:battery
#print("mac battery: ", mac_battery)
try:
    redis_client.ts().create(mac_battery)
except redis.ResponseError:
    # Ignore error if the timeseries already exists
    pass


mac_power = str(hex(uuid.getnode())) + ":power"   # creating time series mac:power
#print("mac power: ", mac_power)
try:
    redis_client.ts().create(mac_power)
except redis.ResponseError:
    # Ignore error if the timeseries already exists
    pass


mac_pluged_seconds = str(hex(uuid.getnode())) + ":mac_pluged_seconds"   # creating time series mac:pluged_seconds
#print("mac pluged_seconds: ", mac_pluged_seconds)
try:
    redis_client.ts().create(mac_pluged_seconds)
    #redis_client.ts().createrule(mac_power, mac_pluged_seconds, aggregation_type='sum', bucket_size_msec=bucket_duration_in_ms)   # laying out rule to compute aggregation

    redis_client.ts().createrule(mac_power, mac_pluged_seconds, aggregation_type='sum', bucket_size_msec=3000)   # laying out rule to compute aggregation
except redis.ResponseError:
    # Ignore error if the timeseries already exists
    pass

battery_retention = int(5 * (2**20 / 1.6) * 1000)  # 3276800000 ms
power_retention = int(5 * (2**20 / 1.6) * 1000)
power_plugged_seconds_retention = int((2**20 / 1.6) * 24 * 60 * 60 * 1000)  # 5.6623104e13 ms

#create retention window
redis_client.ts().alter(mac_battery, retention_msec=battery_retention)
redis_client.ts().alter(mac_power, retention_msec=power_retention)
redis_client.ts().alter(mac_pluged_seconds, retention_msec=power_plugged_seconds_retention)

# delete previous timeseries
if args.delete == 1:
    redis_client.delete(mac_battery)
    redis_client.delete(mac_power)
    redis_client.delete(mac_pluged_seconds)
#%% #parameters

LABELS = ['go', 'stop']
OUTPUT_FOLDER = 'audio_records'

IS_SILENCE_ARGS = {
    'downsampling_rate' : 16000,
    'frame_length_in_s' : 0.016,
    'dbFSthres' : -120,
    'duration_thres' : 0.08
}

# Parameters for mfccs function
PREPROCESSING_MFCCS_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.016,
    'frame_step_in_s': 0.012,
    'num_mel_bins' : 10,
    'lower_frequency': 20,
    'upper_frequency': 8000,
    'num_mfccs_coefficients':40
}


bucket_duration_in_ms = 24 * 60 * 60 * 1000  # 24h


#%% #functions

def get_audio_from_numpy(filename):
    filename = tf.convert_to_tensor(filename, dtype=tf.float32)
    filename = 2 * ((filename + 32768) / (32767 + 32768)) - 1  # CORRECT normalization between -1 and 1
    filename = tf.squeeze(filename)

    return filename

def get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s):
    audio_padded = get_audio_from_numpy(filename)

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    spectrogram = stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram

def get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency):
    spectrogram= get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s)

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    num_spectrogram_bins = frame_length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=downsampling_rate,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

    return log_mel_spectrogram

def get_mfccs(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients):
    log_mel_spectrogram= get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency)

    mfccs =  tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    mfccs = mfccs[:,:num_coefficients]

    return mfccs

def is_silence(filename, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres):

    spectrogram = get_spectrogram(
        filename,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)

    non_silence = energy > dbFSthres
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s
    
    if non_silence_duration > duration_thres:
        return 0
    else:
        return 1

def callback(filename, frames, call_back, status):

    global store_information
    
    store_audio = is_silence(filename=filename,
                             downsampling_rate=IS_SILENCE_ARGS['downsampling_rate'],
                             frame_length_in_s=IS_SILENCE_ARGS['frame_length_in_s'],
                             dbFSthres=IS_SILENCE_ARGS['dbFSthres'],
                             duration_thres=IS_SILENCE_ARGS['duration_thres'])

    if not store_audio:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        mfccs = get_mfccs(filename=filename,
                          downsampling_rate=PREPROCESSING_MFCCS_ARGS['downsampling_rate'],
                          frame_length_in_s=PREPROCESSING_MFCCS_ARGS['frame_length_in_s'],
                          frame_step_in_s=PREPROCESSING_MFCCS_ARGS['frame_step_in_s'],
                          num_mel_bins=PREPROCESSING_MFCCS_ARGS['num_mel_bins'],
                          lower_frequency=PREPROCESSING_MFCCS_ARGS['lower_frequency'],
                          upper_frequency=PREPROCESSING_MFCCS_ARGS['upper_frequency'],
                          num_coefficients=PREPROCESSING_MFCCS_ARGS['num_mfccs_coefficients'])
        

        SHAPE=mfccs.shape
        mfccs.set_shape(SHAPE)
        mfccs = tf.expand_dims(mfccs, 0)
        mfccs = tf.expand_dims(mfccs, -1)
        mfccs = tf.image.resize(mfccs, [32, 32])

        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        top_index = np.argmax(output[0])
        predicted_probability = output[0][top_index]
        predicted_keywords = LABELS[top_index]

        print(f'Prediction: {predicted_keywords} with magnitude: {predicted_probability}')
        if(predicted_probability > 0.95):
            # print(f'Prediction: {predicted_keywords} with magnitude: {predicted_probability}')
            if (predicted_keywords == 'go'):
                store_information = True
                timestamp_ms = int(time() * 1000) 
                battery = psutil.sensors_battery().percent
                power = int(psutil.sensors_battery().power_plugged)
                redis_client.ts().add(mac_battery, timestamp_ms, battery)
                redis_client.ts().add(mac_power, timestamp_ms, power)
                
                print("Started to add in Redis")
            elif (predicted_keywords == 'stop'):
                store_information = False
                print("Stopped to add in Redis")
    


                
#%% #import model
model_path = os.path.join('C:/Users/beorn/Downloads/Telegram Desktop/', 'model03.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

store_information = False

#%% #while
with sd.InputStream(device=DEVICE,
                    channels=CHANNELS,
                    samplerate=IS_SILENCE_ARGS['downsampling_rate'],
                    dtype=DTYPE,
                    blocksize=IS_SILENCE_ARGS['downsampling_rate'] * AUDIO_FILE_LENGTH_IN_S,
                    callback=callback):

    while True:
        key = input()
        if key in ['Q', 'q']:
            print('Stopping the script')
            break
    time.sleep(1)
    

