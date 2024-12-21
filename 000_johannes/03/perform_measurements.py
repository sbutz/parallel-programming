#!/usr/bin/python3
import os
from datetime import datetime
import subprocess

jsonDirectory = os.path.join('.', 'measurements')

os.makedirs(jsonDirectory, exist_ok=True)

now = datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S') + f"{now.microsecond // 1000:03d}"
outDirectory = os.path.join(jsonDirectory, timestamp)

os.mkdir(outDirectory)

idx = 0
sz = 8
maxSz = 2 ** 16 # 2 GiB
while sz <= maxSz:
    args = "s"
    if sz <= 2 ** 31:
        args += "oa"

    cmd = os.path.join('.', 'bin', 'histogram')
    result = subprocess.run(
        [cmd, '--', f'{sz}', args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    with open(os.path.join(outDirectory, f'{timestamp}-{idx:03.0f}.json'), 'w') as file:
        file.write(result.stdout)
    idx += 1
    sz *= 2
