import os

base_dirs = [
    "/work/nvme/bdsg/kiakh93/data_same_batch_final_1024_Olympus/",
    "/work/nvme/bdsg/kiakh93/data_same_batch_final_1024_Olympus_trip3/"
]

# Substrings from screenshot to exclude (without .tif)
exclude_keys = [
    "889A1_B", "889A1_A", "844A1_B", "844A1_A", "754A2", "637C1",
    "508B2_B", "508B2_A", "468A1_B", "468A1_A", "445K1_D", "445K1_C",
    "445K1_B", "445K1_A", "403A1_D", "403A1_C", "403A1_B", "403A1_A",
    "375C1", "272B1", "168B1", "054BS_B", "054BS_A", "045C2_B", "045C2_A"
]

out_file = "ir_file_list.txt"

with open(out_file, "w") as f:
    for base in base_dirs:
        ir_path = os.path.join(base, "ir")
        files = os.listdir(ir_path)
        for file in files:
            name = file.replace(".tif", "")
            # Exclude if startswith sun OR contains any exclude key
            if name.startswith("sun"):
                continue
            if any(key in name for key in exclude_keys):
                continue
            f.write(os.path.join(ir_path, name) + "\n")
