
import argparse
import os
import sys

new_path = os.getcwd()
sys.path.append(new_path)


parser = argparse.ArgumentParser(description='parameters')


parser.add_argument('-mp', '--model_path', type=str, default='/home/share/models/Meta-Llama-3-8B-Instruct')
parser.add_argument('-ip', '--input_path', type=str)
parser.add_argument('-op', '--output_path', type=str)
parser.add_argument('-t', '--temperature', default=0, type=float)


args = parser.parse_args()