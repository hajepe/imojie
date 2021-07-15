import json
import shutil
import sys

from allennlp.commands import main

config_file = "imojie/configs/imojie_de.json"
my_package = "imojie"

# # Use overrides to train on CPU. // not relevant at the moment
# overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "/tmp/dry_run_out"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "dry-run",
    "-s", serialization_dir,
    "--include-package", my_package,
    config_file
]

main()