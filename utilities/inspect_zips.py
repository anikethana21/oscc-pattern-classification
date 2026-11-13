import zipfile
import glob
import os
from collections import Counter

zips = sorted(glob.glob(r"C:\Users\anike\Downloads\*20261111*.zip"))
class_counts = Counter()
file_map = {}

for z in zips:
    try:
        with zipfile.ZipFile(z, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            print(f"=== {os.path.basename(z)} ({len(names)} files) ===")
            for name in names:
                # determine class from folder or name
                parts = name.replace("\\", "/").split("/")
                # find category
                cat = "unknown"
                for p in parts:
                    plow = p.lower()
                    if "dysplasia" in plow:
                        cat = "dysplasia"
                        break
                    elif "normal" in plow:
                        cat = "normal"
                        break
                    elif "cohesive" in plow and "non" not in plow:
                        cat = "cohesive"
                        break
                    elif "non_cohesive" in plow or "noncohesive" in plow:
                        cat = "noncohesive"
                        break
                    elif "dispersive" in plow:
                        cat = "dispersive"
                        break
                    elif "no_feature" in plow or "nofeature" in plow:
                        cat = "no_feature"
                        break
                    elif "blank" in plow:
                        cat = "blank"
                        break
                class_counts[cat] += 1
                if cat not in file_map:
                    file_map[cat] = (os.path.basename(z), name)
    except Exception as e:
        print(f"Error reading {z}: {e}")

print("\n--- Summary of detected raw counts ---")
for k, v in class_counts.items():
    print(f"{k}: {v} (sample: {file_map[k]})")

