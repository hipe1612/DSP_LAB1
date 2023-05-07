import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

Input_1kHz_15kHz =[

+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
-0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
-0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
]

Impulse_response = [
  -0.0018225230, -0.0015879294, +0.0000000000, +0.0036977508, +0.0080754303, +0.0085302217, -0.0000000000, -0.0173976984,
  -0.0341458607, -0.0333591565, +0.0000000000, +0.0676308395, +0.1522061835, +0.2229246956, +0.2504960933, +0.2229246956,
  +0.1522061835, +0.0676308395, +0.0000000000, -0.0333591565, -0.0341458607, -0.0173976984, -0.0000000000, +0.0085302217,
  +0.0080754303, +0.0036977508, +0.0000000000, -0.0015879294, -0.0018225230
]

# Convert to frequency domain
frequency = fft.fft(Impulse_response)

def plot_time():
    plt.plot(Impulse_response)
    plt.title("Impulse Response")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

# Real and imaginary components
real = np.real(frequency)
imaginary = np.imag(frequency)

# Magnitude and phase components
magnitude = np.abs(frequency)
phase = np.angle(frequency)

def plot_frequency():
    # Plotting
    plt.figure(figsize=(12, 6))

    # Real component
    plt.subplot(221)
    plt.plot(real)
    plt.title('Real Component')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    # Imaginary component
    plt.subplot(222)
    plt.plot(imaginary)
    plt.title('Imaginary Component')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    # Magnitude component
    plt.subplot(223)
    plt.plot(magnitude)
    plt.title('Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    # Phase component
    plt.subplot(224)
    plt.plot(phase)
    plt.title('Phase')
    plt.xlabel('Frequency')
    plt.ylabel('Phase (radians)')

    plt.tight_layout()
    plt.show()

#==============================================================================
# Pad the input signal or impulse response with zeros to match their lengths
n = max(len(Input_1kHz_15kHz), len(Impulse_response))
Input_1kHz_15kHz_padded = np.pad(Input_1kHz_15kHz, (0, n - len(Input_1kHz_15kHz)), mode='constant')
Impulse_response_padded = np.pad(Impulse_response, (0, n - len(Impulse_response)), mode='constant')

# Convert input signal and impulse response to frequency domain
input_freq = fft.fft(Input_1kHz_15kHz_padded)
impulse_freq = fft.fft(Impulse_response_padded)

# Output using frequency domain multiplication
output_freq_mult = input_freq * impulse_freq

# Convert output back to time domain using inverse FFT
output_time = fft.ifft(output_freq_mult)

def plot_output():
    # Plotting the output using time convolution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(Input_1kHz_15kHz, label='Input Signal')
    plt.plot(Impulse_response, label='Impulse Response')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plotting the output using frequency domain multiplication
    plt.subplot(2, 1, 2)
    plt.plot(np.real(output_time), label='Output')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Convert output to frequency domain

# Pad the input signal or impulse response with zeros to match their lengths
n = max(len(Input_1kHz_15kHz), len(Impulse_response))
Input_1kHz_15kHz_padded = np.pad(Input_1kHz_15kHz, (0, n - len(Input_1kHz_15kHz)), mode='constant')
Impulse_response_padded = np.pad(Impulse_response, (0, n - len(Impulse_response)), mode='constant')

# Convert input signal and impulse response to frequency domain
input_freq = fft.fft(Input_1kHz_15kHz_padded)
impulse_freq = fft.fft(Impulse_response_padded)

# Output using frequency domain multiplication
output_freq_mult = input_freq * impulse_freq

# Convert output back to time domain using inverse FFT
output_time = fft.ifft(output_freq_mult)

# Convert output to frequency domain
output_freq = fft.fft(output_time)

# Frequency axis
dt = 1  # Assuming the sampling rate is 1
freq = fft.fftfreq(n, dt)

# Plotting real, imaginary, magnitude, and phase of the output frequency domain
def plot_output_frequency():
    plt.figure(figsize=(9, 7))
    plt.subplot(4, 1, 1)
    plt.plot(freq, np.real(output_freq))
    plt.xlabel('Frequency')
    plt.ylabel('Real')

    plt.subplot(4, 1, 2)
    plt.plot(freq, np.imag(output_freq))
    plt.xlabel('Frequency')
    plt.ylabel('Imaginary')

    plt.subplot(4, 1, 3)
    plt.plot(freq, np.abs(output_freq))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.subplot(4, 1, 4)
    plt.plot(freq, np.angle(output_freq))
    plt.xlabel('Frequency')
    plt.ylabel('Phase')

    plt.tight_layout()
    plt.show()

#==============================================================================
def signals_init():
    while True:
        print("1 - plot Impulse Response")
        print("2 - plot input over frequency")
        print("3 - plot output using time convolution/frequency multiplication")
        print("4 - plot output over frequency")
        print("5 - exit")

        choice = int(input("choice: "))
        match choice:
            case 1:
                plot_time()
            case 2:
                plot_frequency()
            case 3:
                plot_output()
            case 4:
                plot_output_frequency()
            case 5:
                break
            case _:
                print("invalid choice!")


if __name__ == "__main__":
    signals_init()
