#!/usr/bin/python3
import os
from datetime import datetime
import subprocess

MIN_SZ = 2 ** 3
MAX_SZ = 2 ** 32 # 4 GiB
N_RUNS = 100
INPUT_FILE_PATH = os.path.join('.', 'input_data', 'test.txt')

jsonDirectory = os.path.join('.', 'measurements')

os.makedirs(jsonDirectory, exist_ok=True)

now = datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S') + f"{now.microsecond // 1000:03d}"
outDirectory = os.path.join(jsonDirectory, timestamp)

os.mkdir(outDirectory)

idx = 0
cmd = os.path.join('.', 'bin', 'histogram')

for arg in ['oas', 'oasl']:
    result = subprocess.run(
        [cmd, INPUT_FILE_PATH, arg, f'{N_RUNS:0.0f}'],
        stdout=subprocess.PIPE, universal_newlines=True, check=True
    )
    with open(os.path.join(outDirectory, f'{timestamp}-{idx:03.0f}.json'), 'w') as file:
        file.write(result.stdout)
    idx += 1

def generateSzArgsLetterCombinations():
    szArgsCombinations = []
    sz = MIN_SZ
    while sz <= MAX_SZ:
        args = 's'
        if sz <= 2 ** 31:
            args += "oa"
        szArgsCombinations.append([f'{sz}', args])
        szArgsCombinations.append([f'{sz}', args + 'u'])
        szArgsCombinations.append([f'{sz}', args + 'l'])
        szArgsCombinations.append([f'{sz}', args + 'ul'])
        sz *= 2
    return szArgsCombinations

for sz, args in generateSzArgsLetterCombinations():
    result = subprocess.run(
        [cmd, '--', f'{sz}', args, f'{N_RUNS:0.0f}'],
        stdout=subprocess.PIPE, universal_newlines=True, check=True
    )
    with open(os.path.join(outDirectory, f'{timestamp}-{idx:03.0f}.json'), 'w') as file:
        file.write(result.stdout)
    idx += 1
