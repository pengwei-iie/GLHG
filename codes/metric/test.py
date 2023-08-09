import json

# file_name = './lsy-pre.json'
# save_name = './lsy-pre.txt'
# with open(file_name, 'r', encoding='utf-8') as infile:
#     obj = json.load(infile)
#
# with open(save_name, mode='w', encoding='utf-8') as outfile:
#     for text in obj:
#         outfile.write(text)
#         outfile.write('\n')

from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='./squad.pre',
                               references=['./squad.target'])
print(metrics_dict)
