import numpy as np
import matplotlib.pyplot as plt

def dwt(signal, wavelet):
    if wavelet == 'haar' or wavelet == 'db1':
        lp_filter = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Low-pass filter
        hp_filter = np.array([1/np.sqrt(2), -1/np.sqrt(2)]) # High-pass filter
    else:
        raise ValueError("Unsupported wavelet type.")
    
    def convolve_and_downsample(signal, filter):
        convolved = np.convolve(signal, filter, mode='full')
        return convolved[::2]  # Downsample by taking every second sample
    
    approx_coeffs = convolve_and_downsample(signal, lp_filter)
    detail_coeffs = convolve_and_downsample(signal, hp_filter)
    
    return approx_coeffs, detail_coeffs

def multi_level_dwt(signal, wavelet, levels):
    detail_coeffs_list = []
    approx_coeffs_list = []
    current_signal = signal
    for level in range(levels):
        approx_coeffs, detail_coeffs = dwt(current_signal, wavelet)
        detail_coeffs_list.append(detail_coeffs)  # Save detail coefficients for heat-map
        approx_coeffs_list.append(approx_coeffs)  # Save approximation coefficients
        current_signal = approx_coeffs  # Use approximation coefficients for next level
    return approx_coeffs_list, detail_coeffs_list

# Create a high-resolution sample signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(1000)

# Perform multi-level DWT
levels = 6  # Increase the number of levels for a more detailed heat-map
approx_coeffs, detail_coeffs = multi_level_dwt(signal, 'haar', levels)

# Create matrices for the heat-maps
detail_coeffs_matrix = np.zeros((levels, len(detail_coeffs[0])))
approx_coeffs_matrix = np.zeros((levels, len(approx_coeffs[0])))

# Fill the matrices with the detail and approximation coefficients at each level
for i in range(levels):
    detail_coeffs_matrix[i, :len(detail_coeffs[i])] = detail_coeffs[i]
    approx_coeffs_matrix[i, :len(approx_coeffs[i])] = approx_coeffs[i]

# Plot the combined heat-maps
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title('Original Signal')

plt.subplot(3, 1, 2)
plt.imshow(np.abs(detail_coeffs_matrix), aspect='auto', cmap='PRGn', interpolation='nearest')
plt.title('DWT Detail Coefficients (High-Pass) Heat-Map')
plt.xlabel('Time')
plt.ylabel('Level')
plt.colorbar(label='Magnitude')

plt.subplot(3, 1, 3)
plt.imshow(np.abs(approx_coeffs_matrix), aspect='auto', cmap='PRGn', interpolation='nearest')
plt.title('DWT Approximation Coefficients (Low-Pass) Heat-Map')
plt.xlabel('Time')
plt.ylabel('Level')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()


