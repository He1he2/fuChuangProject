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

def generate_report():
    skeleton_instances, meta_info = load_skeleton_keypoint()
    client = OpenAI(
        api_key=qianfan_api_key,
        base_url="https://qianfan.baidubce.com/v2",
    )
    per_analysis = []
    
    batch_size = 5
    for i in range(0, 20, batch_size):
        batch_instances = skeleton_instances[i:i+batch_size]
        
        # Build system prompt for the current batch of frames
        system_prompt = f"""
你是一个专业的乒乓球运动动作分析专家，擅长通过人体骨骼关键点坐标进行动作识别和姿态评估。请根据以下每帧图像中进行乒乓球对打时的人体骨骼点信息，结合字段说明，对每个人的动作进行详细分析。分析内容应包括但不限于：

1. 动作是否规范（如站立、跑步、跳跃等是否符合常规标准）； 
2. 姿态是否稳定，是否存在不协调或危险的动作；
3. 动作幅度是否合理；
4. 如存在问题，请指出具体关键点及改进建议。

为每个人进行分析，并且考虑到当前图像帧的编号，便于最后根据整体的运动轨迹进行分析。
输出标签使用：人体1、人体2、人体3
以下是骨骼点字段说明（metadata）： 
{meta_info}

以下是第 {i+1} 到 {min(i + batch_size, len(skeleton_instances))} 帧图像中的每个人体的骨骼点数据：
"""
        for j, skeleton_instance in enumerate(batch_instances):
            system_prompt += f"""
第 {i + j + 1} 帧图像中的每个人体的骨骼点数据： 
{skeleton_instance}
"""
        # Send the batch of frames to the OpenAI API for processing
        try:
            completion = client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[
                    {"role": "system", "content": "你是乒乓球运动动作分析专家"},
                    {"role": "user", "content": system_prompt},
                ],
                stream=False,
            )
            per_analysis.append(completion.choices[0].message.content)
            print(completion.choices[0].message.content)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting for 30 seconds...")
            time.sleep(10)  
            completion = client.chat.completions.create(
                model="ernie-x1-32k-preview",
                messages=[
                    {"role": "system", "content": "你是乒乓球运动动作分析专家"},
                    {"role": "user", "content": system_prompt},
                ],
                stream=False,
            )
            per_analysis.append(completion.choices[0].message.content)
            print(completion.choices[0].message.content)

    # After processing all batches, generate the final report
    final_prompt = f"""
你是一个专业的乒乓球运动动作分析专家，请根据以下每一帧的分析结果，对整个运动过程进行综合性分析，并撰写一份结构清晰、内容全面的乒乓球运动动作分析报告。

请在报告中包括以下内容：
1. 动作规范性评价；
2. 是否存在危险/异常动作；
3. 动作连贯性和节奏分析；
4. 改进建议；
5. 综合评价；
等等

以下是每帧的动作分析内容：
{per_analysis}
"""
    response = client.chat.completions.create(
        model="ernie-x1-32k-preview",
        messages=[
            {"role": "system", "content": "你是乒乓球运动动作分析专家"},
            {"role": "user", "content": final_prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(generate_report())
