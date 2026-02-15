import subprocess

PATH_TO_EXE = "dist/InferenceModel.exe"
PATH_TO_TEST_IMAGE = (
    r"G:/Meine Ablage/DLCV/Fine-Tuning Dataset/000a19ac-df09-4f9e-9b95-8f89dccadeb9.png"
)

with open(PATH_TO_TEST_IMAGE, "rb") as f:
    image_bytes = f.read()

result = subprocess.run([PATH_TO_EXE], input=image_bytes, stdout=subprocess.PIPE)

print("STDOUT:")
print(result.stdout.decode())
