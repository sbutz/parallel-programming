from collections import defaultdict
import tempfile
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NvProfReport:
    def __init__(self, binary, args):
        self.binary = binary
        logger.info(f'Profiling: {self.binary} {" ".join(args)}')
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [self.binary] + args
            if not self.binary.endswith(".py"):
                cmd = [
                    "nsys",
                    "profile",
                    "--stats=true",
                    f"--output={tmpdir}/report%n",
                ] + cmd
            print(" ".join(cmd))
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            self.report = result.stdout.decode("utf-8").split("\n")
            self.trace = result.stderr.decode("utf-8").split("\n")

    def get_traced_value(self, key):
        lines = [l for l in self.trace if f"TRACE: {key} " in l]
        assert len(lines) == 1, f"Expected 1 line with key {key}, got {len(lines)}"
        return float(lines[0].split()[2])

    def get_kernel_exec_stats(self, kernel_name):
        if self.binary.endswith(".py"):
            return defaultdict(lambda: 0)

        lines = [l for l in self.report if f" {kernel_name}(" in l]
        parts = lines[0].split()
        return {
            "time": float(parts[1].replace("'", '')) / 1000,  # in us
            "instances": int(parts[2]),
            "avg": float(parts[3].replace("'", '')) / 1000,  # in us
            "med": float(parts[4].replace("'", '')) / 1000,  # in us
            "min": float(parts[5].replace("'", '')) / 1000,  # in us
            "max": float(parts[6].replace("'", '')) / 1000,  # in us
            "stddev": float(parts[7].replace("'", '')) / 1000,  # in us
            "name": parts[8],
        }

    def get_kernel_launch_stats(self):
        if self.binary.endswith(".py"):
            return defaultdict(lambda: 0)

        lines = [l for l in self.report if "cudaLaunchKernel" in l]
        parts = lines[0].split()
        return {
            "time": float(parts[1].replace("'", '')) / 1000,  # in us
            "instances": int(parts[2]),
            "avg": float(parts[3].replace("'", '')) / 1000,  # in us
            "med": float(parts[4].replace("'", '')) / 1000,  # in us
            "min": float(parts[5].replace("'", '')) / 1000,  # in us
            "max": float(parts[6].replace("'", '')) / 1000,  # in us
            "stddev": float(parts[7].replace("'", '')) / 1000,  # in us
            "name": parts[8],
        }

    def get_memcpy_to_device_stats(self):
        if self.binary.endswith(".py"):
            return defaultdict(lambda: 0)

        lines = [l for l in self.report if "[CUDA memcpy Host-to-Device]" in l]
        parts = lines[0].split()
        return {
            "time": float(parts[1].replace("'", '')) / 1000,  # in us
            "instances": int(parts[2]),
            "avg": float(parts[3].replace("'", '')) / 1000,  # in us
            "med": float(parts[4].replace("'", '')) / 1000,  # in us
            "min": float(parts[5].replace("'", '')) / 1000,  # in us
            "max": float(parts[6].replace("'", '')) / 1000,  # in us
            "stddev": float(parts[7].replace("'", '')) / 1000,  # in us
            "name": parts[8],
        }

    def get_memcpy_to_host_stats(self):
        if self.binary.endswith(".py"):
            return defaultdict(lambda: 0)

        lines = [l for l in self.report if "[CUDA memcpy Device-to-Host]" in l]
        parts = lines[0].split()
        return {
            "time": float(parts[1].replace("'", '')) / 1000,  # in us
            "instances": int(parts[2]),
            "avg": float(parts[3].replace("'", '')) / 1000,  # in us
            "med": float(parts[4].replace("'", '')) / 1000,  # in us
            "min": float(parts[5].replace("'", '')) / 1000,  # in us
            "max": float(parts[6].replace("'", '')) / 1000,  # in us
            "stddev": float(parts[7].replace("'", '')) / 1000,  # in us
            "name": parts[8],
        }
