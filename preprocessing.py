import tensorflow as tf
import tensorflow_io as tfio


LABELSB = ['go','stop']

def get_audio_and_label(filename):
    audio_binary = tf.io.read_file(filename)
    audio, sampling_rate = tf.audio.decode_wav(audio_binary) 

    path_parts = tf.strings.split(filename, '/')
    path_end = path_parts[-1]
    file_parts = tf.strings.split(path_end, '_')
    label = file_parts[0]

    audio = tf.squeeze(audio)
    zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
    audio_padded = tf.concat([audio, zero_padding], axis=0)

    return audio_padded, sampling_rate, label


def get_spectrogram(filename, downsampling_rate, frame_length_in_s,frame_step_in_s,num_mel_bin, lower_frequency, upper_frequency,num_coefficients):
    # TODO: Write your code here
    audio_padded, sampling_rate, label = get_audio_and_label(filename)
    
    if downsampling_rate != sampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    # frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_length = int(frame_length_in_s * downsampling_rate)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    spectrogram = stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)
    
    return spectrogram, downsampling_rate, label


def get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bin, lower_frequency, upper_frequency,num_coefficients):

    
    spectrogram, downsampling_rate, label = get_spectrogram(filename,downsampling_rate,frame_length_in_s,frame_step_in_s,num_mel_bin, lower_frequency, upper_frequency,num_coefficients)
    # num_spectrogram_bins = tf.shape(spectrogram)[1] # frame_length // 2 + 1
    # print("num_spectrogram_bins: ", num_spectrogram_bins)
    # print("spectrogram: ", spectrogram.shape)
    frame_length = int(downsampling_rate * frame_length_in_s)
    num_spectrogram_bin = frame_length//2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=num_mel_bin,
      num_spectrogram_bins=num_spectrogram_bin,
      sample_rate=downsampling_rate,
      lower_edge_hertz=lower_frequency,
      upper_edge_hertz=upper_frequency
     )
    # print(spectrogram.take(1).shape)
    #print(linear_to_mel_weight_matrix.shape)
    # print(linear_to_mel_weight_matrix.get_shape().as_list())
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]
   
    return mfccs,downsampling_rate, label


