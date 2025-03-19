
import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
import pygame
from pygame.locals import *
import sys


import usb.core
import usb.backend.libusb1

backend = usb.backend.libusb1.get_backend(
    find_library=lambda x: r"C:\Users\leoni\Downloads\pat\MinGW64\dll\libusb-1.0.dll"
)
# Get all USB devices using the modern API
usb_devices = usb.core.find(backend=backend, find_all=True)

# Define your audio interface Vendor ID and Product ID
audio_interface_vendor_id = 0x08bb  # Replace with your audio interface's Vendor ID
audio_interface_product_id = 0x2902  # Replace with your audio interface's Product ID

# Iterate over the devices found
for dev in usb_devices:
    if dev.idVendor == audio_interface_vendor_id and dev.idProduct == audio_interface_product_id:
        print("Audio interface is connected!")



# Set the sample rate and duration
samplerate = 48000
duration = 10  # Duration of the recording in seconds

# Define the number of channels
input_channels = 1
output_channels = 2

# Define the block size and latency
blocksize = 1024
latency = 0.05

# Input and output device indices
input_device = 1  # Example: Microsoft Sound Mapper - Input
output_device = 4  # Example: Microsoft Sound Mapper - Output


# Constants for visualization
screen_width = 800
screen_height = 600
max_display_samples = 100  # Number of samples to display in heatmap
update_interval = 20  # Update visualization every N callbacks


# Global variables
detail_coeffs_queue = []
approx_coeffs_queue = []
dft_magnitude_queue = []
callback_counter = 0


#Convolution
def manual_convolve(signal, filter):
    filter_len = len(filter)
    signal_len = len(signal)
    result_len = signal_len + filter_len - 1
    
    # Initialize the result array with zeros
    result = np.zeros(result_len)
    
    # Reverse the filter
    reversed_filter = filter[::-1]
    
    # Perform convolution
    for i in range(result_len):
        sum_val = 0
        for j in range(filter_len):
            if i - j >= 0 and i - j < signal_len:
                sum_val += signal[i - j] * reversed_filter[j]
        result[i] = sum_val
    
    return result

#Downsampling
def downsample(signal):
    return signal[::2]

#Discrete wavelet transform
def dwt(signal, wavelet='haar'):
    if wavelet == 'haar' or wavelet == 'db1':
        lp_filter = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Low-pass filter
        hp_filter = np.array([1/np.sqrt(2), -1/np.sqrt(2)]) # High-pass filter
    else:
        raise ValueError("Unsupported wavelet type.")
    
    def convolve_and_downsample(signal, filter):
        convolved = manual_convolve(signal, filter)
        return downsample(convolved)  # Downsample by taking every second sample
    
    approx_coeffs = convolve_and_downsample(signal, lp_filter)
    detail_coeffs = convolve_and_downsample(signal, hp_filter)
    
    return approx_coeffs, detail_coeffs

def calcular_tdf(signal):
    N = len(signal)
    tdf = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            tdf[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return tdf

# Distortion function
def distortion(input_signal, gain=50.0, threshold=45.0):
    processed_signal = input_signal * gain
    processed_signal = np.where(processed_signal > threshold, threshold, processed_signal)
    processed_signal = np.where(processed_signal < -threshold, -threshold, processed_signal)
    return processed_signal

# Helper function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 6-Band EQ function
def eq_6_band(input_signal, fs, gains):
    bands = [(20, 60), (60, 250), (250, 500), (500, 2000), (2000, 4000), (4000, 16000)]
    output_signal = np.zeros_like(input_signal)

    for (low, high), gain in zip(bands, gains):
        band_signal = bandpass_filter(input_signal, low, high, fs)
        band_signal *= 10**(gain / 20)  # Convert dB gain to linear scale
        output_signal += band_signal

    return output_signal

# Example gains for the 6 bands (in dB)
gains = [5, 5, 8, 7, 5, 10]  # Example gain values for each band

# Revised delay effect function
def delay_effect(input_signal, fs, delay_time=0.7, feedback=0.9, mix=0.6):
    delay_samples = int(fs * delay_time)
    buffer = np.zeros(delay_samples)
    output_signal = np.zeros_like(input_signal)
    
    for n in range(len(input_signal)):
        delayed_sample = buffer[n % delay_samples]
        output_signal[n] = input_signal[n] * (1 - mix) + delayed_sample * mix
        buffer[n % delay_samples] = input_signal[n] + delayed_sample * feedback
    
    return output_signal

def smoothstep(x):
    y = (6 * (x ** 5)) - (15 * (x ** 4)) + (10 * (x ** 3))
    return y

def Smoothing(insig):
    signal = insig.copy()
    low = np.min(signal)
    high = np.max(signal)
    normsignal = (signal - low) / (high - low)
    smnormal = smoothstep(normsignal)
    smoothed = (smnormal * (high - low)) + low
    return smoothed

def compressor(signal, bot=-1.0, top=1.0):
    diff = top - bot
    marr = max(list(signal))
    minarr = min(list(signal))
    diffar = marr - minarr
    signal = (((signal - minarr) * diff) / diffar) + bot
    return signal

def NoiseFilter(insig, Thresh=0.1):
    noiseless = np.where(np.abs(insig) < Thresh, 0, insig)
    return noiseless

# Create a callback function to process the audio data
def callback(indata, outdata, frames, time, status):
    global callback_counter
    if status:
        print(status)
        pass

    # Clean = indata


    #precomp = NoiseFilter(indata, 0.00009)
    #compressed_Signal = compressor(precomp)
    #Bitcrush = NoiseFilter(compressed_Signal, 0.7)


    Toob = compressor(indata)
    distorted_signal = distortion(Toob, gain=10, threshold=1)
    DistoEq = eq_6_band(distorted_signal, samplerate, gains)
    
        
    # Twangy = NoiseFilter(indata, 0.0001)
    # Tweedy = compressor(Twangy)
    # smoothed = Smoothing(Tweedy)
    
    # Copy the processed signal to the output
    
    #print(indata)
    processed_signal = indata[:, 0]

    if callback_counter % update_interval == 0:
        if visualization_mode == 'DWT':
            # Perform DWT
            approx_coeffs, detail_coeffs = dwt(processed_signal)

            if len(detail_coeffs_queue) >= max_display_samples:
                detail_coeffs_queue.pop(0)
            if len(approx_coeffs_queue) >= max_display_samples:
                approx_coeffs_queue.pop(0)

            detail_coeffs_queue.append(detail_coeffs)
            approx_coeffs_queue.append(approx_coeffs)

        elif visualization_mode == 'DFT':
            # Perform DFT
            tdf = calcular_tdf(processed_signal)
            tdf_magnitude = np.abs(tdf)

            if len(dft_magnitude_queue) >= max_display_samples:
                dft_magnitude_queue.pop(0)

            dft_magnitude_queue.append(tdf_magnitude)

    callback_counter += 1
    outdata[:] = DistoEq
    

def visualize_dwt():
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Real-Time DWT Visualization')
    font = pygame.font.SysFont('Arial', 18)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))

        if len(detail_coeffs_queue) > 0:
            detail_matrix = np.array(detail_coeffs_queue)
            detail_surface = pygame.surfarray.make_surface(np.abs(detail_matrix.T) * 255)
            screen.blit(pygame.transform.scale(detail_surface, (screen_width, screen_height//2)), (0, 0))

        if len(approx_coeffs_queue) > 0:
            approx_matrix = np.array(approx_coeffs_queue)
            approx_surface = pygame.surfarray.make_surface(np.abs(approx_matrix.T) * 255)
            screen.blit(pygame.transform.scale(approx_surface, (screen_width, screen_height//2)), (0, screen_height//2))

        # Draw axes labels for detail coefficients
        screen.blit(font.render('Time', True, (255, 255, 255)), (screen_width // 2, screen_height // 4 - 20))
        screen.blit(font.render('Frequency', True, (255, 255, 255)), (10, screen_height // 4 - 10))

        # Draw axes labels for approximation coefficients
        screen.blit(font.render('Time', True, (255, 255, 255)), (screen_width // 2, 3 * screen_height // 4 - 20))
        screen.blit(font.render('Frequency', True, (255, 255, 255)), (10, 3 * screen_height // 4 - 10))

        # Draw line to separate sections
        pygame.draw.line(screen, (255, 255, 255), (0, screen_height//2), (screen_width, screen_height//2), 1)

        pygame.display.flip()
        clock.tick(30)

def visualize_dft():
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Real-Time DFT Visualization')
    font = pygame.font.SysFont('Arial', 18)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))

        if len(dft_magnitude_queue) > 0:
            dft_matrix = np.array(dft_magnitude_queue)
            dft_surface = pygame.surfarray.make_surface(np.abs(dft_matrix.T) * 255)
            screen.blit(pygame.transform.scale(dft_surface, (screen_width, screen_height)), (0, 0))

        # Draw axes labels for DFT magnitudes
        screen.blit(font.render('Time', True, (255, 255, 255)), (screen_width // 2, screen_height - 20))
        screen.blit(font.render('Frequency', True, (255, 255, 255)), (10, screen_height // 2 - 10))

        pygame.display.flip()
        clock.tick(30)

def open_audio_stream():
    try:
        with sd.Stream(device=(input_device, output_device),
                       samplerate=samplerate,
                       blocksize=blocksize,
                       channels=(input_channels, output_channels),
                       callback=callback):
            print(f"Running stream with input device {input_device} and output device {output_device} at {samplerate} Hz")
            if visualization_mode == 'DWT':
                visualize_dwt()
            elif visualization_mode == 'DFT':
                visualize_dft()
    except Exception as e:
        print(f"Failed to open stream -> {e}")

    print("Done!")

# Ask user for visualization mode
visualization_mode = "DWT"#input("Select visualization mode (DWT/DFT): ").strip().upper()

if visualization_mode not in ['DWT', 'DFT']:
    print("Invalid mode selected. Exiting.")
    sys.exit()

# Run the audio stream with the selected visualization
open_audio_stream()
