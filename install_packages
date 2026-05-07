import sys

packages = [
    'numpy', 'pandas', 'matplotlib', 'rasterio',
    'geopandas', 'folium',
    'sklearn', 'torch', 'dowhy', 'netwrokx'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f'OK  {pkg}')
    except ImportError:
        print(f'FAIL  {pkg}  <-- run: python -m pip install {pkg}')
    print(f'\nPython version: {sys.version}')