#!/usr/bin/env python3

import logging
import os
import pandas as pd
import subprocess
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

measurementsDirectory = os.path.join('.', 'measurements')
os.makedirs(measurementsDirectory, exist_ok=True)

now = datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S') + f"{now.microsecond // 1000:03d}"
currentMeasurementDirectory = os.path.join(measurementsDirectory, timestamp)
os.mkdir(currentMeasurementDirectory)

currentMeasurementTempDirectory = os.path.join(currentMeasurementDirectory, 'temp')
os.mkdir(currentMeasurementTempDirectory)

imagesOutDirectory = os.path.join('.', 'images_out')
os.makedirs(imagesOutDirectory, exist_ok=True)
currentImagesOutDirectory = os.path.join(imagesOutDirectory, timestamp)
os.mkdir(currentImagesOutDirectory)

deviceQueryCmd = os.path.join('.', 'bin', 'deviceQuery')
result = subprocess.run(
    [deviceQueryCmd], stdout=subprocess.PIPE, universal_newlines=True, check=True
)
with open(os.path.join(currentMeasurementDirectory, f'{timestamp}-devicequery.txt'), 'w') as file:
    file.write(result.stdout)


def profile(binary, args):
    logger.info(f'Profiling: {binary} {" ".join(args)}')
    cmd = [
        "nsys",
        "profile",
        "--stats=true",
        f"--output={currentMeasurementTempDirectory}/report%n",
        binary,
        *args,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, check=True
    )
    return result.stdout.decode("utf-8").split("\n")

# [0] = Time (%)
# [1] = Total Time (ns)
# [2] = Instances
# [3] = Avg (ns)
# [4] = Med (ns)
# [5] = Min (ns)
# [6] = Max (ns)
# [7] = StdDev (ns)
# [8] = Name
NSYS_ROW = [ 'Time (percent)', 'Time (total)', 'Instances', 'Time (avg.)', 'Time (med.)', 'Time (min.)', 'Time (max.)', 'Time (std. dev.)', 'Name']
NSYS_IDX = { s: i for i, s in enumerate(NSYS_ROW) }

def get_kernel_execution_time(report, kernel_name):
    lines = [l for l in report if f" {kernel_name}(" in l]
    parts = lines[0].split()
    # convert times to us
    return {
        f'Exec {s}': (float(parts[i]) if s in ['Time (percent)', 'Instances']
            else ' '.join(parts[NSYS_IDX['Name']:]) if s == 'Name'
            else float(parts[i].replace("'", '').replace(",", '')) / 1000
        ) for s, i in NSYS_IDX.items() 
    }

def get_kernel_launch_time(report):
    lines = [l for l in report if "cudaLaunchKernel" in l]
    parts = lines[0].split()
    # convert times to us
    return {
        f'Launch {s}': (float(parts[i]) if s in ['Time (percent)', 'Instances']
            else ' '.join(parts[NSYS_IDX['Name']:]) if s == 'Name'
            else float(parts[i].replace("'", '').replace(",", '')) / 1000
        ) for s, i in NSYS_IDX.items() 
    }

def get_memcpy_to_device_time(report):
    lines = [l for l in report if "[CUDA memcpy Host-to-Device]" in l]
    parts = lines[0].split()
    # convert times to us
    return {
        f'Memcpy To Device {s}': (float(parts[i]) if s in ['Time (percent)', 'Instances']
            else ' '.join(parts[NSYS_IDX['Name']:]) if s == 'Name'
            else float(parts[i].replace("'", '').replace(",", '')) / 1000
        ) for s, i in NSYS_IDX.items() 
    }

def get_memcpy_to_host_time(report):
    lines = [l for l in report if "[CUDA memcpy Device-to-Host]" in l]
    parts = lines[0].split()
    # convert times to us
    return {
        f'Memcpy To Host {s}': (float(parts[i]) if s in ['Time (percent)', 'Instances']
            else ' '.join(parts[NSYS_IDX['Name']:]) if s == 'Name'
            else float(parts[i].replace("'", '').replace(",", '')) / 1000
        ) for s, i in NSYS_IDX.items() 
    }

scenarios = (
    [ {
        'binary': os.path.join(".", "bin", "01_grayscale"),
        'fixed_args': [],
        'kernel_name': 'RgbToGrayscale',
        'scenario_id': f'grayscale'
    } ] +
    [ {
        'binary': os.path.join(".", "bin", "02_blur"),
        'fixed_args': [ str(margin) ],
        'kernel_name': 'Blur',
        'scenario_id': f'blur-{margin:02d}'
    } for margin in [ 1, 2, 3 ] ]
)

def image_filter():
    imageRecords = []
    for image_dir in [ os.path.join(".", img) for img in [ 'images', 'images_from_web' ] ]:
        try:
            with open(os.path.join(image_dir, 'info.json'), 'r') as file:
                for entry in json.load(file):
                    imageRecords.append({
                        'path': os.path.join(image_dir, entry['filename']),
                        'directory': image_dir,
                        'filename': entry['filename'],
                        'width': entry['width'],
                        'height': entry['height']
                    })
        except:
            pass

    # Profile kernels
    d = []
    for record in imageRecords:
        basename, ext = os.path.splitext(record['filename'])
        for scenario in scenarios:
            report = profile(
                scenario['binary'],
                scenario['fixed_args'] + [
                    record['path'],
                    os.path.join(currentImagesOutDirectory, f'{basename}-{scenario["scenario_id"]}{ext}')
                ]
            )

            with open(os.path.join(currentMeasurementDirectory, f'report-{basename}-{scenario["scenario_id"]}.txt'), 'w') as file:
                file.write('\n'.join(report))
            
            dict = {
                "directory": record['directory'],
                "image": record['filename'],
                "width": record['width'],
                "height": record['height'],
                "scenario_id": scenario['scenario_id']
            }
            dict.update(get_kernel_execution_time(report, scenario['kernel_name']))
            dict.update(get_kernel_launch_time(report))
            dict.update(get_memcpy_to_device_time(report))
            dict.update(get_memcpy_to_host_time(report))

            d.append(dict)
    with open(os.path.join(currentMeasurementDirectory, 'results.json'), 'w') as file:
        json.dump(d, file, indent=4)

def main():
    image_filter()

if __name__ == "__main__":
    main()
