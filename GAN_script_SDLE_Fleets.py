import sys
from datetime import datetime
import pandas as pd
from time import sleep
import os 
from GAN_functional import run_script
# Do not edit the print statements!!!!

try:
    #line contains one line of your input CSV file
    parquet_path = sys.argv[1]
    batch = int(sys.argv[2])
    array_start = int(sys.argv[3])
    start = array_start + int(os.environ['SLURM_ARRAY_TASK_ID'])
    end = start + batch


    print(f'[SLURMSTART] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job started with args:-{parquet_path} {start} {end}') 
    input_df = pd.read_parquet(parquet_path, engine='pyarrow').iloc[start:end].reset_index()
    #Your code here! 
    # try catch inside of a for loop, will skip that section
    print(input_df.to_string(line_width=100))
    for index, row in input_df.iterrows():
        with open(f'/home/aml334/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman/packages/Unet_train/{index}printed.txt','w') as file:
            run_script(input_df=row)
            file.write('File Submitted'+f'[SLURMSTART] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job started with args:-{parquet_path} {start} {end}')


    # Choose your out_dir 
    out_dir  = os.path.abspath(os.path.dirname(parquet_path))

    out_file = f'{out_dir}/out-{start:08}-{end:08}.parquet'

    # Write the output file
    input_df.to_parquet(out_file)
    #Your code ends here
    print(f'[SLURMEND] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job finished successfully')
except Exception as e:
    print(f'[SLURMFAIL] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job failed, see error below')
    print(repr(e))
    sys.exit(os.EX_SOFTWARE)