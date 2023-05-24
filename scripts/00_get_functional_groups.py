from pathlib import Path
import pickle
from subprocess import Popen, PIPE
import time

import json


def run_command(cmd):
    """Execute the external command and get its exitcode, stdout and
    stderr."""

    t0 = time.time()
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    exitcode = proc.returncode
    dt = time.time() - t0

    return {
        "exitcode": exitcode,
        "output": out.decode("utf-8").strip(),
        "error": err.decode("utf-8").strip(),
        "elapsed": dt,
    }


d = Path("data") / "22-12-05-data"


if __name__ == "__main__":
    data = pickle.load(open(d / "xanes.pkl", "rb"))

    # Generate all_smiles.smi
    all_smiles = list(data["data"].keys())
    with open(d / "all_smiles.smi", "w") as f:
        f.write("\n".join(all_smiles))

    in_pth = d / "all_smiles.smi"
    out_path = d / "functional_groups.txt"
    run_command(f"obabel -ismi {in_pth} -ofpt -xfFP4 -xs > {out_path}")

    with open(d / "functional_groups.txt", "r") as f:
        functional_groups = f.readlines()

    # Every other line is just a ">"
    functional_groups = functional_groups[1::2]

    # Construct a dictionary from this
    functional_groups = {
        smile: group.strip().split()
        for smile, group in zip(all_smiles, functional_groups)
    }

    # Save this back as a dictionary
    with open(d / "functional_groups.json", "w") as outfile:
        json.dump(functional_groups, outfile, indent=4, sort_keys=True)
