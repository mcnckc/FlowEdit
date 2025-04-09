import os

with open('i_s.txt', 'r') as sf, open("target_labels.txt", 'r') as tf, open('scene_pair.yaml', 'w') as res:
    for sline, tline in zip(sf, tf):
        filename, slabel = sline.split()
        tlabel = tline.split()[1]
        res.write(f"-\n"
                  f"    input_img: scenepair/{filename}\n"
                  f"    source_prompt: {slabel}\n"
                  f"    target_prompts:\n"
                  f"    - {tlabel}\n"
                  f"    target_codes:\n"
                  f"    - {filename}-{slabel}-to-{tlabel}\n")