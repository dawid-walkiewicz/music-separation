import csv
from pathlib import Path


class Logger:
    def __init__(self, path: Path, fieldnames: list):
        self.path = path
        self.fieldnames = fieldnames

        if not self.path.exists():
            with open(self.path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, data: dict):
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: v for k, v in data.items() if k in self.fieldnames})