# Trening separatora audio (folder `traning/`)

Ten folder zawiera kompletny, prosty pipeline do trenowania modelu rozdzielającego ścieżki dźwiękowe (source separation) z nagrania muzycznego. Trenujemy lekki model 1D U-Net na zbiorze MUSDB18 (wersja `.stem.mp4`) do separacji czterech źródeł: vocals, drums, bass, other.

Spis treści:
- Co robi ten kod (wysoki poziom)
- Składniki i jak działają (data → model → trening → checkpointy)
- Jak uruchomić trening (GPU/CPU, parametry CLI)
- Wyniki treningu i do czego służą
- Wydajność i typowe problemy (MemoryError itp.)
- Ograniczenia i dalsze kierunki

---

## Co robi ten kod (wysoki poziom)

- Losowo pobiera krótkie segmenty (domyślnie 6 s) z utworów MUSDB18.
- Tworzy „miks” przez sumę 4 źródeł i parę danych wejście/wyjście: (mixture → poszczególne źródła).
- Trenuje 1D U-Net, który przewiduje maski czasowe dla każdego źródła; maski nakłada na sygnał wejściowy.
- Liczy stratę L1 między przewidywanymi źródłami a referencjami.
- Wspiera GPU (CUDA) i AMP (mixed precision) dla szybkości.
- Zapisuje checkpointy co N kroków i pozwala wznawiać trening.

Efekt: Wytrenowany checkpoint modelu, który można użyć do separacji nowych nagrań (patrz „Wyniki i do czego służą”).

---

## Składniki i jak działają

Pliki:
- `data.py` – loader danych MUSDB18, losowe segmenty
- `model.py` – prosty 1D U-Net + funkcja `apply_masks`
- `train.py` – pętla treningowa z GPU/AMP, logami i checkpointami

Kontrakty danych (kształty tensorów):
- `mixture` ma kształt `(C, L)`, gdzie C=1 (mono), L = liczba próbek w segmencie.
- `targets` ma kształt `(S, C, L)`, gdzie S=4 (vocals, drums, bass, other).
- W batchu: `mixture` jest `(B, 1, L)`, `targets` `(B, 4, 1, L)`.

### `data.py` (MUSDB losowe segmenty)
- Używa `musdb` i `stempeg`, czyta źródła utworu (vocals, drums, bass, other).
- Każdy przykład to losowy start w utworze i wycinek o długości `segment_seconds`.
- Normalizuje dane do układu `(C, L)` (kanały, długość) i konwertuje na `float32`.
- Wariant mono: kanały stereo są uśredniane do jednego kanału (C=1) – to upraszcza model i zmniejsza RAM/VRAM.
- Miks = suma czterech źródeł w domenie czasu: `mixture = sum(sources)`.

### `model.py` (1D U-Net + maski)
- Architektura encoder–decoder z 1D conv, `ConvTranspose1d` i skip-connections.
- Wyjście modelu: maski o wymiarze `(B, S, Lm)` w zakresie `[0,1]` (sigmoid).
- `apply_masks(mixture, masks)` przycina do wspólnej długości i zwraca `(B, S, 1, Lc)` jako przewidywane źródła.
- Podczas dekodowania przycinamy tensory skipów do tej samej długości (czasem różnią się o 1 próbkę po down/up-samplingu).

### `train.py` (trening na GPU/CPU, checkpointy)
- Urządzenie: automatycznie wykrywa CUDA i włącza AMP na GPU.
- Strata: średnia L1 pomiędzy `preds` i `targets` (dopasowane długością do `preds`).
- Checkpointy: zapisywane do `workdir/checkpoints/step_XXXXXX.pt` oraz finalny `final_step_XXXXXX.pt`.
- Wznawianie: `--resume auto` znajdzie ostatni checkpoint w folderze; możesz też podać ścieżkę.

Struktura checkpointu (`.pt`):
- `model` – `state_dict()` modelu,
- `optimizer` – stan optymalizatora (AdamW),
- `step`, `epoch` – licznik postępu,
- `scaler` – stan GradScaler (AMP), jeśli GPU,
- `extra` – pole na metadane.

---

## Jak uruchomić trening

Wymagania (podstawowe): `musdb`, `stempeg`, `demucs`, `numpy`. PyTorch instaluj zgodnie z Twoją wersją CUDA.

1) Zainstaluj zależności projektu (bez PyTorch):
```bat
pip install -r requirements.txt
```
2) Zainstaluj PyTorch dla Twojej karty/CUDA (Windows, przykłady – sprawdź https://pytorch.org/get-started/locally/):
- CUDA 12.1:
```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
- CPU-only:
```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
3) Upewnij się, że dane MUSDB18 są w `./musdb18/train` i `./musdb18/test` (pliki `.stem.mp4`).

4) Uruchom trening (przykład bezpieczny pamięciowo):
```bat
python -m training.train --data_root .\musdb18 --workdir .\runs\unet1d ^
  --batch_size 2 --segment_seconds 2.0 --num_workers 0 ^
  --max_steps 2000 --ckpt_every 200 --log_every 20
```
Wyjaśnienie parametrów:
- `--data_root` – ścieżka do MUSDB18.
- `--workdir` – gdzie zapisać checkpointy i logi.
- `--batch_size` – zwiększaj stopniowo na GPU; jeśli OOM, zmniejsz.
- `--segment_seconds` – długość wycinka; dłuższe = więcej kontekstu, ale większe zużycie pamięci.
- `--num_workers` – na Windows często 0..4; jeśli są błędy, zacznij od 0.
- `--max_steps` – ile kroków treningu.
- `--ckpt_every` – co ile kroków zapisać checkpoint.
- `--log_every` – co ile kroków wypisać średnią stratę.
- `--resume` – ścieżka do checkpointu albo `auto`.

Wznowienie z ostatniego checkpointu:
```bat
python -m traning.train --data_root .\musdb18 --workdir .\runs\unet1d --resume auto
```

---

## Wyniki treningu i do czego służą

Co powstaje:
- Folder `workdir/checkpoints/` z plikami `step_XXXXXX.pt` i `final_step_XXXXXX.pt`.
- Każdy checkpoint zawiera stan modelu, optymalizatora i (na GPU) skaler AMP.

Do czego służą:
- To są wagi wytrenowanego separatora. Możesz ich użyć do:
  - kontynuacji treningu (fine-tuning),
  - ewaluacji na zbiorze testowym,
  - separacji nowych plików audio (inferencja).

Przykładowa inferencja w Pythonie (opcjonalnie):

```python
import torch
import numpy as np
from models.unet1d.model import UNet1D, apply_masks

# Załaduj model
model = UNet1D(n_sources=4)
ckpt = torch.load(r"runs\unet1d\checkpoints\final_step_002000.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# Wczytaj mono waveform (44100 Hz) jako numpy (1, L)
# (np. przez soundfile/librosa/torchaudio – zrób resampling do 44100 Hz i uśrednij kanały do mono)
wave = ...  # ndarray shape (L,) lub (1, L)
if wave.ndim == 1:
    wave = wave[None, :]

x = torch.from_numpy(wave.astype(np.float32))[None, ...]  # (B=1, C=1, L)
with torch.no_grad():
    masks = model(x)  # (1, 4, Lm)
    sources = apply_masks(x, masks)  # (1, 4, 1, Lc)

# sources[0, i, 0] to waveform i-tego źródła; zapisz do WAV jak wolisz
```
Uwaga: Ten projekt nie zawiera jeszcze dedykowanego skryptu inferencji na plikach (można go łatwo dodać). Jeśli chcesz, doinstaluj `torchaudio`/`soundfile` i zrób mały skrypt wczytujący WAV/MP3, resamplujący do 44.1 kHz i zapisujący 4 ścieżki.

---

## Wydajność i typowe problemy

- MemoryError (RAM) lub OOM (VRAM):
  - Zmniejsz `--batch_size` (nawet do 1).
  - Skróć `--segment_seconds` (np. 1.0–2.0 s).
  - Ustaw `--num_workers 0` na Windows dla stabilności.
  - Upewnij się, że używasz AMP na GPU (włączone automatycznie).

- Różnice długości o 1 próbkę przy skip-connections: obsłużone automatycznym przycinaniem w modelu i przy liczeniu straty.

- `musdb` nie widzi plików:
  - Sprawdź strukturę folderów: `musdb18/train/*.stem.mp4` i `musdb18/test/*.stem.mp4`.
  - Zainstaluj `ffmpeg` (potrzebny do `stempeg`).

- Loader jest mono (C=1). Rozszerzenie do stereo wymaga zmian w modelu i pamięci.

---

## Ograniczenia i dalsze kierunki

- Prosty, czasowy model 1D – dla lepszej jakości rozważ:
  - architekturę opartą o STFT/ISTFT (maski w dziedzinie częstotliwości),
  - Conv-TasNet, Demucs-arch, hybrid time–freq.
- Brak wbudowanej ewaluacji na zbiorze testowym – można dodać metryki (np. SI-SDR, SDR) i walidację.
- Brak gotowego skryptu inferencji – patrz przykład powyżej (możemy dodać `infer.py` na życzenie).

---

## Dlaczego mono i maski?

- Mono upraszcza i przyspiesza trening (mniej pamięci), a jakość na start bywa wystarczająca.
- Maski w [0,1] mnożone przez miks są stabilne numerycznie i ograniczają „halucynacje” modelu – model nie generuje sygnału od zera, ale „wybiera”, co należy do danego źródła.

---

Jeśli chcesz, mogę dodać skrypt `infer.py`, który:
- wczyta checkpoint,
- przyjmie dowolny plik audio,
- zresampluje do 44.1 kHz mono,
- zapisze 4 rozdzielone ścieżki WAV do wybranego katalogu.

