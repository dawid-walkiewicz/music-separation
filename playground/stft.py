import math
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# Ścieżka do przykładowego pliku (możesz zmienić na inny plik z twojego dysku)
SAMPLE_SPEECH = r"C:\Users\matic\PycharmProjects\music-separation\musdb18-wav\train\A Classic Education - NightOwl\mixture.wav"

# Parametry STFT
n_fft = 2048
hop_length = 512
win_length = 2048

# Wybierz device: CUDA jeśli dostępne, w przeciwnym razie CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używany device: {device}")

# Helper: załaduj plik lub wygeneruj sygnał testowy
def load_or_generate(path, max_seconds=6):
    try:
        waveform, sr = torchaudio.load(path)
        print(f"Wczytano: {path} (sr={sr}, channels={waveform.shape[0]}, samples={waveform.shape[1]})")
    except Exception as e:
        print(f"Nie można wczytać pliku {path}: {e}\nGeneruję sygnał testowy (sinusoida 440Hz, 5s).")
        sr = 44100
        t = torch.linspace(0, 5, steps=5 * sr)
        waveform = 0.5 * torch.sin(2 * math.pi * 440 * t)
        waveform = waveform.unsqueeze(0)
    # Konwertuj do mono jeśli stereo
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Ogranicz do maksymalnie max_seconds
    max_samples = min(waveform.shape[1], int(max_seconds * sr))
    waveform = waveform[:, :max_samples]
    # Przenieś waveform na device (CUDA lub CPU)
    waveform = waveform.to(device)
    return waveform, sr


def compute_log_power_spectrogram(waveform, sr, n_fft=2048, hop_length=512, win_length=None, eps=1e-10):
    if win_length is None:
        win_length = n_fft
    # Utwórz okno na tym samym device co waveform
    window = torch.hann_window(win_length, device=waveform.device)
    # STFT (wynik: [channel, freq_bins, time_frames], complex)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    magnitude = stft.abs()
    power = magnitude ** 2
    # Przelicz na decybele (dB), zabezpiecz przed log(0)
    log_power = 10.0 * torch.log10(power + eps)
    return log_power


def plot_and_show(waveform, sr, log_power, hop_length):
    # Przenieś do CPU i konwertuj do numpy do rysowania
    lp = log_power[0].detach().cpu().numpy()
    # osie czas/freq
    freq_bins, time_frames = lp.shape
    freqs = np.linspace(0, sr / 2, freq_bins)
    times = np.arange(time_frames) * (hop_length / sr)

    # Waveform for plotting (CPU)
    waveform_np = waveform[0].detach().cpu().numpy()
    time_wave = np.linspace(0, waveform_np.shape[0] / sr, waveform_np.shape[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    ax1.plot(time_wave, waveform_np, linewidth=0.6)
    ax1.set_title('Waveform')
    ax1.set_xlim(0, time_wave[-1])
    ax1.set_xlabel('Time (s)')

    # Spectrogram
    im = ax2.imshow(lp, origin='lower', aspect='auto', cmap='magma', extent=[times[0], times[-1] if len(times)>0 else 0, freqs[0], freqs[-1]])
    ax2.set_title('Log Power Spectrogram (dB)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    cbar = fig.colorbar(im, ax=ax2, format='%+2.0f dB')
    cbar.set_label('dB')

    plt.tight_layout()
    # Nie zapisujemy pliku — tylko wyświetlamy
    try:
        plt.show()
    except Exception as e:
        print(f"Nie udało się wyświetlić wykresu: {e}")


if __name__ == '__main__':
    waveform, sr = load_or_generate(SAMPLE_SPEECH, max_seconds=6)
    log_power = compute_log_power_spectrogram(waveform, sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    plot_and_show(waveform, sr, log_power, hop_length)
