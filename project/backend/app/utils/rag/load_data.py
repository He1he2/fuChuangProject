import json
import time

from openai import OpenAI
import openai

qianfan_api_key = "bce-v3/ALTAK-8Jpl4STJhWXhOOuUoMZ7m/af766013775177e66da4b51a692b36c00df576b0"

def load_skeleton_keypoint(file_path="./app/utils/video/output_pose/user_3/results_20250403142257431.json"):
    with open(file_path, 'r') as file:
        data = json.load(file)
    instance_info = data.get('instance_info', [])
    meta_info = data.get('meta_info', {})
    return instance_info, meta_info


