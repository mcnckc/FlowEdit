import os

with open('i_s.txt', 'r') as sf, open("target_labels.txt", 'r') as tf, open('scene_pair.yaml', 'w') as res:
    for sline, tline in zip(sf, tf):
        filename, slabel = sline.split()
        tlabel = tline.split()[1]
        res.write(f"-\tinput_img: scenepair/{filename}\n"
                  f"\tsource_prompt: {slabel}\n"
                  f"\ttarget_prompts:\n"
                  f"\t- {tlabel}\n"
                  f"\ttarget_codes:\n"
                  f"\t- {filename}-{slabel}-to-{tlabel}\n")