import os
import subprocess
import io

input_file = './cv-template-zh.pdf'
print(os.path.abspath(input_file))

# # 方法1
output_file = './cv1.html'
subprocess.run(['pdf2htmlEX', input_file, output_file])