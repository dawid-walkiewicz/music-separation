import os
import sys
import subprocess
from pathlib import Path
from tkinter import Tk, filedialog


def choose_file():
    """Otwiera okno wyboru pliku i zwraca jego ścieżkę."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Wybierz plik audio lub wideo",
        filetypes=[
            ("Pliki audio/wideo", "*.mp3 *.wav *.flac *.ogg *.m4a *.mp4 *.stem.mp4"),
            ("Wszystkie pliki", "*.*"),
        ],
    )
    return file_path


def separate_with_demucs(file_path: str):
    """Wykonuje separację pliku przy użyciu Demucs i zapisuje wynik w ./separated/<nazwa_utworu>/"""
    audio_path = Path(file_path)
    if not audio_path.exists():
        print(f"❌ Plik nie istnieje: {audio_path}")
        return

    # Gdzie ma się zapisać wynik (w folderze gdzie uruchamiany jest skrypt)
    current_dir = Path.cwd()
    output_root = current_dir / "separated"
    output_song_dir = output_root / audio_path.stem
    output_song_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎧 Uruchamiam separację Demucs dla pliku: {audio_path.name}\n")

    command = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs",
        str(audio_path)
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Błąd podczas działania Demucs:", e)
        return

    # Szukamy wyników w ./separated/htdemucs/<nazwa_pliku> (domyślna lokalizacja Demucsa)
    demucs_base = Path("separated") / "htdemucs" / audio_path.stem
    if not demucs_base.exists():
        print("❌ Nie znaleziono wyników Demucs:", demucs_base)
        return

    # Przenosimy wszystkie pliki .wav do naszego folderu ./separated/<nazwa_utworu>/
    for wav in demucs_base.glob("*.wav"):
        dest = output_song_dir / wav.name
        wav.replace(dest)
        print(f"✅ Zapisano: {dest}")

    # Usuwamy oryginalny folder demucsa (opcjonalnie)
    try:
        for f in demucs_base.iterdir():
            f.unlink()
        demucs_base.rmdir()
    except Exception:
        pass

    print(f"\n🎵 Zakończono! Wszystkie ścieżki zapisane w: {output_song_dir}\n")


if __name__ == "__main__":
    file_path = choose_file()
    if file_path:
        separate_with_demucs(file_path)
    else:
        print("❌ Nie wybrano pliku.")
