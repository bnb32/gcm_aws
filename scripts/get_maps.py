"""Get maps for simluations"""
import os
import warnings
import argparse
warnings.filterwarnings("ignore")

parser=argparse.ArgumentParser(description="Download paleo-continent maps")
args=parser.parse_args()

zip_file = 'Scotese_Wright_2018_Maps_1-88_6minX6min_PaleoDEMS_nc.zip'
data_source = f'https://zenodo.org/record/5460860/files/{zip_file}'
#zip_file = 'Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc.zip'
#data_source = f'http://www.earthbyte.org/webdav/ftp/Data_Collections/Scotese_Wright_2018_PaleoDEM/{zip_file}'
cmd = f'rm -rf {os.environ["RAW_TOPO_DIR"]}'
cmd += f'; wget {data_source}'
cmd += f'; unzip {zip_file}'
cmd += f'; mv {zip_file.strip(".zip")} {os.environ["RAW_TOPO_DIR"]}'
cmd += f'; rm {zip_file}'

os.system(cmd)
