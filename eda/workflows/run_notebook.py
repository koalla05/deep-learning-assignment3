"""Execute a local notebook through a Unix IPC socket and save its outputs."""

import argparse
import uuid
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def run(path):
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3', transport='ipc',
                            ip='/private/tmp/eda-notebook-' + uuid.uuid4().hex)
    try:
        NotebookClient(notebook, km=manager, timeout=900,
                       resources={'metadata': {'path': str(path.parent)}}).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook')
    args = parser.parse_args()
    print(run(args.notebook))
