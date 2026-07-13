import subprocess

adapter = subprocess.check_output(
    " netsh wlan show drivers",
    shell = True
    ).decode()

print(adapter)