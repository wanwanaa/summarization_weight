import re

filenames = ['DATA/raw_data/src-train.txt',
             'DATA/raw_data/src-valid.txt',
             'DATA/raw_data/src-test.txt']
filenames_segment = ['DATA/seg_data/src-train.txt',
                     'DATA/seg_data/src-valid.txt',
                     'DATA/seg_data/src-test.txt']

# filenames = ['DATA/raw_data/src-test.txt']
# filenames_segment = ['DATA/seg_data/src-test.txt']

for i in range(len(filenames)):
    result = []
    with open(filenames[i], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # sentences segmentation
            # sentences = re.split('(。”|。"|，|。|；|？”|？|！”|！|\\n)', line)
            sentences = re.split('(。”|。"|。|；|？”|？|！”|！)', line)
            new_sents = []
            for j in range(int(len(sentences)/2)):
                sent = sentences[2*j] + sentences[2*j+1]
                new_sents.append(sent)
            if len(sentences) == 1:
                new_sents.append(sentences[0])
            new_sents[-1] += '\n'
            # new_sents.append('\n')
            # print(new_sents)
            # if new_sents[-1] == '\n':
            #     new_sents.pop()
                # print('temp2:', new_sents[-1])
            # result.append('[CLS] ' + ' [SEP] '.join(new_sents))
            # result.append(' [SEP] '.join(new_sents))
            result.append('\n'.join(new_sents))
    with open(filenames_segment[i], 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))