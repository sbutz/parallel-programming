#!/usr/bin/env python3

import os
from datetime import datetime
import glob
import re
import subprocess

N_RUNS = 11

jsonDirectory = os.path.join('.', 'measurements')

os.makedirs(jsonDirectory, exist_ok=True)

now = datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S') + f"{now.microsecond // 1000:03d}"
outDirectory = os.path.join(jsonDirectory, timestamp)

os.mkdir(outDirectory)

deviceQueryCmd = os.path.join('.', 'bin', 'deviceQuery')
result = subprocess.run(
    [deviceQueryCmd], stdout=subprocess.PIPE, universal_newlines=True, check=True
)
with open(os.path.join(outDirectory, f'{timestamp}-devicequery.txt'), 'w') as file:
    file.write(result.stdout)

binaries = [
    'a_andrade',
    'a_andrade_01',
    'a_andrade_01',
    'a_andrade_02',
    'a_andrade_04',
    'a_andrade_05',
    'a_andrade_texture',
    'c_boehm'
]
dataFileDirectory = os.path.join('.', 'sample_data')

def getListOfDataFiles():
    filePattern = os.path.join(dataFileDirectory, 'data_*_*.dat')
    fileNames = [ os.path.basename(s) for s in glob.glob(filePattern) ]

    # pairs (number of clusters, number of data points)
    lst = []
    pattern = r'data_(\d+)_(\d+)\.dat'
    for fileName in fileNames:
        match = re.match(pattern, fileName)
        if match:
            nClusters, nPoints = match.groups()
            lst.append({
                'fileName': fileName,
                'nClusters': int(nClusters),
                'nPoints': int(nPoints)
            })
    return lst

dataFileEntries = getListOfDataFiles()

def getDbscanArguments(nPoints):
    r = 1.4 # always
    coreThreshold = min(10000, int(nPoints / (nClusters * 5)))
    return (coreThreshold, r)

idx = 0
for binary in binaries:
    for dataFileEntry in [
        e for e in dataFileEntries if e['nPoints'] <= 150000
    ]:
        cmd = os.path.join('.', 'bin', binary)
        dataFileWithPath = os.path.join('.', dataFileDirectory, dataFileEntry['fileName'])
        nClusters = dataFileEntry['nClusters']
        nPoints = dataFileEntry['nPoints']
        coreThreshold,r = getDbscanArguments(dataFileEntry['nPoints'])

        result = subprocess.run(
            [cmd, dataFileWithPath, f'{coreThreshold:.0f}', f'{r}', 'w', f'{N_RUNS}'],
            stdout=subprocess.PIPE, universal_newlines=True, check=True
        )

        outFileName = f'{timestamp}-{binary}-{nClusters:.0f}-{nPoints:.0f}-{idx:03.0f}.json'
        with open(os.path.join(outDirectory, outFileName), 'w') as file:
            file.write(result.stdout)
        
        idx += 1
