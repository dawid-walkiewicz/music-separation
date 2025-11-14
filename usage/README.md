Instrukcja użycia skryptu `separate.py`.

Wymagane biblioteki (jedna z dróg):
- soundfile (pysoundfile) + resampy (resampling), lub
- torchaudio (IO + resampling w jednym), lub
- dla plików MUSDB `.stem.mp4`: `stempeg` (jest już w requirements) — wówczas można przetwarzać bez soundfile/torchaudio.

Instalacja (przykłady):
- `pip install soundfile resampy`
- albo: `pip install torchaudio`

Podstawowe użycie:
- `--ckpt` – ścieżka do checkpointu (np. `runs\unet1d\checkpoints\final_step_000300.pt`)
- `--input` – plik audio lub folder (obsługa: wav/mp3/flac/ogg/m4a i `.stem.mp4`)
- `--out` – katalog wyjściowy (domyślnie `usage/separated`)
- `--device` – `cuda` lub `cpu` (domyślnie auto)

Parametry anty-OOM (GPU):
- `--chunk_seconds` – długość fragmentu przetwarzanego jednorazowo (domyślnie 8.0 s). Zmniejsz do 2.0–4.0 s, jeśli masz mało VRAM.
- `--overlap_seconds` – nakładanie między fragmentami (domyślnie 1.0 s). Pozwala uniknąć artefaktów na łączeniach.
- `--no_amp` – wyłącza mixed precision na CUDA. Domyślnie AMP jest włączone (mniej pamięci), więc używaj `--no_amp` tylko gdy masz problemy z NaN.
- Skrypt ma fallback: przy błędzie "CUDA out of memory" automatycznie przetworzy dany plik na CPU.

Przykłady:
- Separacja pojedynczego WAV z niskim zużyciem pamięci: ustaw `--chunk_seconds 2.0`.
- Separacja całego `musdb18\test` (pliki `.stem.mp4`) wymaga `stempeg` i zadziała bez soundfile/torchaudio.

Wynik:
- Pliki WAV zapisywane w `usage/separated/<nazwa>/<vocals|drums|bass|other>.wav`.

Uwagi:
- Jeśli chcesz całkowicie uniknąć GPU, dodaj `--device cpu`.
- Jeżeli przetwarzasz `.stem.mp4`, skrypt używa miksu z pliku STEM (jeśli obecny jako pierwszy stem) lub sumy źródeł.
