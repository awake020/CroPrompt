
import argparse
import os
import sys

openai_api_key=""

new_path = os.getcwd()
sys.path.append(new_path)

parser = argparse.ArgumentParser(description='parameters')

parser.add_argument('-mn', '--model_name', type=str, default='gpt-3.5-turbo') 
parser.add_argument('-ip', '--input_path', type=str)
parser.add_argument('-op', '--output_path', type=str)
parser.add_argument('-t', '--temperature', default=0, type=float)


args = parser.parse_args()