# WiFi Profile Lister in Python

import subprocess

profiles = subprocess.check_output(
    "netsh wlan show profiles",
    shell = True
).decode()

print(profiles)