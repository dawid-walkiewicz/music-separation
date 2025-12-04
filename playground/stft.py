import torch
import torchaudio
import matplotlib.pyplot as plt

SAMPLE_SPEECH = r"C:\\Users\\matic\\PycharmProjects\\music-separation\\musdb18-wav\\train\\A Classic Education - NightOwl\\mixture.wav"

# ===== SELECT DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()
    _, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_matrix(mat, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.set_title(title)
    im = ax.imshow(mat, origin="lower", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax)


# ===== Load audio (CPU)
waveform, sr = torchaudio.load(SAMPLE_SPEECH)

# ===== SETTINGS — one middle fragment
chunk_seconds = 10
start_time = 80          # sekunda startu
end_time = start_time + chunk_seconds

start_sample = int(start_time * sr)
end_sample = int(end_time * sr)

# ===== Extract fragment and move to GPU
chunk = waveform[:, start_sample:end_sample].to(device)
print("Chunk shape:", chunk.shape, "| device:", chunk.device)

# ===== STFT parameters
n_fft = 2048
hop_length = 512
win_length = 2048
window = torch.hann_window(win_length).to(device)

# ===== STFT on GPU
stft_gpu = torch.stft(
    input=chunk,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    return_complex=True
)

# ===== Magnitude + Phase (still on GPU)
magnitude_gpu = stft_gpu.abs()
phase_gpu = stft_gpu.angle()

# ===== Move to CPU only for plotting
chunk_cpu = chunk.cpu()
magnitude = magnitude_gpu[0].cpu().numpy()
phase = phase_gpu[0].cpu().numpy()

# ===== Plot results
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plot_waveform(chunk_cpu, sr, "Waveform of selected fragment", ax=axs[0])
plot_matrix(magnitude, "STFT Magnitude", ax=axs[1])
plot_matrix(phase, "STFT Phase", ax=axs[2])
plt.tight_layout()
plt.show()
