import os
import random

with open('i_s.txt', 'r') as sf, open("target_labels.txt", 'r') as tf, open('scene_pair_small.yaml', 'w') as res:
    subset = 10
    if subset >= 0:
        lines = random.choices(list(zip(sf.readlines(), tf.readlines())), k=subset)
    else:
        lines = zip(sf.readlines(), tf.readlines())
    for sline, tline in lines:
        filename, slabel = sline.split()
        tlabel = tline.split()[1]
        res.write(f"-\n"
                  f"    input_img: scenepair/{filename}\n"
                  f"    source_prompt: >-\n"
                  f"        Text that reads \"{slabel}\"\n"
                  f"    target_prompts:\n"
                  f"    - >-\n"
                  f"        Text that reads \"{tlabel}\"\n"
                  f"    target_codes:\n"
                  f"    - >-\n"
                  f"        {filename}-{slabel}-to-{tlabel}\n")