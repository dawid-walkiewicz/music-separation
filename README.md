# music-separation

Prosty pipeline do separacji źródeł audio:
- Separacja gotowym modelem Demucs
- Trening własnego lekkiego modelu (1D U-Net) na MUSDB18
- Checkpointy zapisywane co N kroków
- Wsparcie GPU (CUDA) jeśli dostępne

## Wymagania

Zainstaluj zależności (najlepiej w wirtualnym środowisku):

```
pip install -r requirements.txt
```

Następnie zainstaluj PyTorch odpowiedni dla Twojej karty/CUDA (Windows):
- Wejdź na https://pytorch.org/get-started/locally/
- Wybierz pip, Python, Windows i wersję CUDA zgodną ze sterownikami GPU, np.:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Lub CPU-only:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Struktura danych MUSDB18

Umieść dane w `./musdb18` (jak w repo):

```
musdb18/
  train/
  test/
```

Wersja `*.stem.mp4` jest obsługiwana przez `musdb` + `stempeg`.

## Uruchomienie separacji Demucs

W PyCharm uruchom skrypt:
- `Test/separate_demucs.py`

lub z terminala (Windows cmd):

```
python -m demucs -n htdemucs -o separated "musdb18/test/Al James - Schoolboy Facination.stem.mp4"
```

Wyniki pojawią się w `./separated/htdemucs/<nazwa_utworu>/`.

## Trening własnego modelu

Domyślnie użyje GPU jeśli jest dostępny (AMP włączony). Uruchom:

```
python -m traning.train --data_root ./musdb18 --workdir ./runs/unet1d --max_steps 2000 --ckpt_every 200 --log_every 20
```

Aby automatycznie wznowić z ostatniego checkpointu:
```
python -m traning.train --resume auto
```

Checkpointy zapisują się do `./runs/unet1d/checkpoints/step_XXXXXX.pt`.

## Uwaga

- Model jest uproszczony (mono, 1D). Lepsze wyniki: architektury oparte o STFT/ISTFT, Conv-TasNet, Demucs.
- Jeżeli `musdb` nie znajduje plików, upewnij się że folder `musdb18` zawiera podfoldery `train/` i `test/` z plikami `.stem.mp4`.
