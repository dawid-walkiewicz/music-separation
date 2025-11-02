import stempeg
audio, stems = stempeg.read_stems("../musdb18/test/Al James - Schoolboy Facination.stem.mp4")
print(audio.shape)
