from pathlib import Path

files = [f.stem for f in Path("src").iterdir() if f.is_file()]

split = "utils"

save_path = Path("DLCV-CHESS") / split / "cool"


print(save_path)
