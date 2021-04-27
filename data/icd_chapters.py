# divide ICD-10 data into the 21 chapters
import json
from collections import defaultdict
import csv
from pattern_tokenizer import tokenize

code_ranges = {
    '1': ('A00', 'B99'),
    '2': ('C00', 'D49'),
    '3': ('D50', 'D89'),
    '4': ('E00', 'E89'),
    '5': ('F01', 'F99'),
    '6': ('G00', 'G99'),
    '7': ('H00', 'H59'),
    '8': ('H60', 'H95'),
    '9': ('I00', 'I99'),
    '10': ('J00', 'J99'),
    '11': ('K00', 'K95'),
    '12': ('L00', 'L99'),
    '13': ('M00', 'M99'),
    '14': ('N00', 'N99'),
    '15': ('O00', 'O99'),
    '16': ('P00', 'P96'),
    '17': ('Q00', 'Q99'),
    '18': ('R00', 'R99'),
    '19': ('S00', 'T99'),
    '20': ('V00', 'Y99', 'W99', 'X99'),
    '21': ('Z00', 'Z99'),
    # '22': ('U00', 'U99'),
}

lookup = {}
for k, v in sorted(code_ranges.items()):
    for x in v:
        lookup[x[0]] = k

new_d = 5
new_h = 6

infile = 'codes.csv'

if __name__ == "__main__":
    with open(infile, 'r') as f:

        chapters = defaultdict(set)
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            code = line[0]
            name = line[3]

            # determine label
            if code[0] == 'D':
                if int(code[1]) >= new_d:
                    label = '3'
                else:
                    label = '2'
            elif code[0] == 'H':
                if int(code[1]) > new_h:
                    label = '8'
                else:
                    label = '7'
            else:
                label = lookup[code[0]]

            # preprocess name
            name = ' '.join(tokenize(name)).lower()
            # add name
            chapters[label].add(name)

    chapters = {k: sorted(v) for k, v in chapters.items()}

    outfile = 'icd10.json'
    with open(outfile, 'w') as f:
        json.dump(chapters, f)
