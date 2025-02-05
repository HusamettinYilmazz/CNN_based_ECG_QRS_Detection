import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import matplotlib
import pywt
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir(r"/home/husammm/Desktop/Courses/Python/MLCourse/uni/ECG_QRS")
# path = input("please enter your data path inculde the image name")
path100 = '/home/husammm/Desktop/Courses/Python/MLCourse/uni/ECG_QRS/database/100'
path101 = '/home/husammm/Desktop/Courses/Python/MLCourse/uni/ECG_QRS/database/101'
path102 = '/home/husammm/Desktop/Courses/Python/MLCourse/uni/ECG_QRS/database/102'

## Creat directory
def create_dir(path):
    ## Create a directory
    if not os.path.exists(path):
        os.makedirs(path)

## square wave
def square_wave(t, frequency, amplitude = 1.0):
    return amplitude * (1 + np.sign(np.sin(2*np.pi*frequency*t))) / 2

## FS
def fourier_coefficients(t, signal, num_terms):
    coeffients = []

    for n in range(num_terms):
        an = 2*np.trapz(signal*np.cos(2*np.pi*n*t),t)
        bn = 2*np.trapz(signal*np.sin(2*np.pi*n*t),t)
        coeffients.append((an, bn))

    return coeffients

def reconstruct_signal(t, coeffients):
    signal = np.zeros_like(t)
    individual_terms = []

    for n, (an, bn) in enumerate(coeffients):
        term = an * np.cos(2*np.pi*n*t) + bn*np.sin(2*np.pi*n*t)
        individual_terms.append(term)

        signal += term
    
    return signal, individual_terms

## FT
def plot_time_series(signal, time_points, title):
    plt.plot(time_points,signal)
    plt.title(title)
    plt.xlabel('Time (sec)')
    plt.ylabel('Amp')
    plt.grid(True)

def discrete_fourier_transform(signal, sampling_frequency):
    N = len(signal)
    k = np.arange(N)
    n = k.reshape(N,1)
    W = np.exp(-2j * np.pi * k * n / N)
    frequencies = k * sampling_frequency / N
    return frequencies, np.dot(W,signal)

def plot_discrete_fourier_transform(signal, sampling_frequency, title):
    frequencies, fourier_transform = discrete_fourier_transform(signal, sampling_frequency)

    positive_freq_indices = (frequencies >= 0) & (frequencies <= 500)
    plt.plot(frequencies[positive_freq_indices], np.abs(fourier_transform[positive_freq_indices]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

##### Fourier Series #####
def Fourier_series(times, ECG, num_terms):
    coefficients = fourier_coefficients(times, ECG, num_terms)
    reconstructed_signal, individual_terms = reconstruct_signal(times,coefficients)
    return reconstructed_signal, individual_terms

##### Fourier Transform #####
def Fourier_transform(record):
    samplingFrequency = record.fs
    return samplingFrequency

##### Wavelet Transform #####
def Wavelet_transform(ECG1, ECG2, levels):
    wavelet = 'db4'
    level = levels # 360 - 180, 180 - 90, 90 - 45, 45 - 0

    signal_length = len(ECG1)

    coeffs1 = pywt.wavedec(ECG1,wavelet,level=level)
    coeffs2 = pywt.wavedec(ECG2,wavelet,level=level)

    time_values = [np.linspace(0,signal_length, len(coef), endpoint=False) for coef in coeffs1]
    timesRealECG = np.arange(len(ECG1),dtype=float)
    return coeffs1, coeffs2, time_values, timesRealECG

##### Convert Wavelet to image #####
def waveletImage(path, levels):
    #load an ECG record
    record = wfdb.rdrecord(path, sampto= 15000)
    data = record.p_signal
    channel1 = data[:, 0]

    #compute the lenth of the orginial signal 
    signal_length = len(channel1)

    # print(pywt.wavelist(kind= 'discrete'))
    #Define wavelets
    wavelets = pywt.wavelist(kind = 'discrete')
    # wavelets = ['bior1.1', 'bior1.3', 'bior1.5',
    # 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8',
    # 'bior3.1', 'bior3.3', 'bior3.5','bior3.7',]

    wavelet_level = levels

    #create an array to store the data
    ecgDataset = np.zeros(((wavelet_level+1) * len(wavelets), signal_length))
    # print(len(ecgDataset))

    #store the real ecg data in a array
    channel1 = channel1 - np.mean(channel1)
    minValue = min(channel1)
    maxValue = max(channel1)
    channel1 = (channel1 - minValue) / (maxValue - minValue)
    ecgDataset[0] = channel1

    counterwaveletassignmet = 0
    for i, wavelet in enumerate(wavelets):
        #applay the wavelet transform
        coeffs1 = pywt.wavedec(channel1, wavelet, level=wavelet_level)

        #compute the time value for each coefficient level
        time_value = [np.linspace(0, signal_length, len(coef),
                                endpoint= False, dtype=int)for coef in coeffs1]
        
        # print(len(time_value[:][0]), len(time_value[0][:]),
        #     max(time_value[:][0]) ,min(time_value[0][:]))

        for j, coef in enumerate(coeffs1):
            coef = coef - np.mean(coef)
            #normalize the wavelet coefficients
            minValue = min(coef)
            maxValue = max(coef)
            coef = (coef - minValue) / (maxValue - minValue)
            ecgDataset[counterwaveletassignmet][time_value[j]] = coef
            counterwaveletassignmet = counterwaveletassignmet + 1

    #visualize of the wavelet picture
    stepsize = 500
    iterationnumber = 20
    imagenumber = int(signal_length / stepsize)
    # print(len(ecgDataset[0,:]), len(ecgDataset[:,0]))

    #The rest of your code for visualization
    #plt.figure(figsize=(12, 12))
    cmap = "viridis"
    counter = 0
    # print(len(signal_length))
    images = []
    for i in range(0, signal_length - stepsize, iterationnumber):
        newmatrix = np.zeros(((wavelet_level+1) * len(wavelets), iterationnumber), dtype=float)
        newmatrix[:, :] = 255*ecgDataset[:, i:i + iterationnumber]
        images.append(newmatrix)
        
    return images

##### FS_visualize #####
def visalize_FS(times, ECG, individual_terms, reconstructed_signal):
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(times, ECG, label='ECG1')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.legend()

    plt.subplot(3,1,2)

    for n,term in enumerate(individual_terms):
        plt.plot(times, term)

    plt.title('Individual Signal')
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(times, reconstructed_signal, label='Our ECG')
    plt.title('Reconstructed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.legend()

    plt.tight_layout()
    plt.show()

##### FT_visualize #####
def visalize_FT(times, ECG, samplingFrequency):
    plt.figure(figsize=(8,6))

    plt.subplot(2,1,1)
    plot_time_series(ECG, times, 'Times Series Data')

    plt.subplot(2,1,2)
    plot_discrete_fourier_transform(ECG, samplingFrequency,'DFT Coefficients')

    plt.tight_layout()
    plt.show()

##### WV_visualize #####
def visualize_wavelet(timesRealECG, time_values, ECG1, ECG2, coeffs1, coeffs2):
    
    # visualize the wavelet coefficient

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(timesRealECG,ECG1, label='Real ECG1')

    for i, coef in enumerate(coeffs1):
        plt.plot(time_values[i],coef + i*3,label=f'Level{i}')

    plt.title('Wavelet Coeffients - ECG1')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(timesRealECG,ECG2, label='Real ECG1')

    for i, coef in enumerate(coeffs2):
        plt.plot(time_values[i],coef + i*3,label=f'Level{i}')

    plt.title('Wavelet Coeffients - ECG2')
    plt.legend()

    plt.tight_layout()
    plt.show()

##### WAV_IMG_visualize #####
def visualize_IMGwavelet(images, levels):
    cmap = "viridis"
    for i in images:
        # print(newmatrix)
        plt.figure(figsize=(8, 8))
        plt.imshow(i, cmap=cmap)

        plt.tight_layout()
        plt.show()

##### Save images #####
def save_images(save_path, images, signal_num):
    create_dir(save_path)
    idx = 1
    for i in images:
        tmp_image_name = f"{signal_num}_{idx}.jpg"
        image_path = os.path.join(save_path, tmp_image_name)
        cv2.imwrite(image_path, i)
        idx += 1

## annotation
def annotation(images100, images101, images102):
    lenth = len(images100) + len(images101) + len(images102)
    annot = [1] * lenth 
    for i in range(lenth):
        if i > 2 * lenth / 3:
            annot[i] = 0
    final_images = np.concatenate((images100, images101, images102), axis=0)
    return final_images, annot

## split_data
def split_data(final_images, annot, split=0.2):
    ## Split the data
    split_size = int(len(final_images) * split)
    x_train, x_test, t_train, t_test = train_test_split(final_images, annot, test_size=split_size, random_state=42, shuffle=True) 

    return x_train, x_test, t_train, t_test
