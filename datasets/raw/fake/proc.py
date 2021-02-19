import csv
try:
    import tqdm
    import efficiency
except ImportError:
    import os
    os.system('pip install tqdm efficiency')

from tqdm import tqdm
from efficiency.nlp import NLP
import sys

csv.field_size_limit(sys.maxsize)

train_file = 'train.csv'
test_file = 'test.csv'
submit_file = 'submit.csv'
nlp = NLP()

with open(submit_file) as f:
    reader = csv.DictReader(f)
    id2pol = {line['id']: line['label'] for line in reader}

with open(test_file) as f:
    reader = csv.DictReader(f)
    test_content = list(reader)

    for line_ix, line in enumerate(test_content):
        doc_ix = line['id']
        pol = id2pol[doc_ix]
        test_content[line_ix]['label'] = pol


for file in [train_file, test_file]:
    if file == test_file:
        content = test_content
    else:
        with open(file) as f:
            reader = csv.DictReader(f)
            content = [line for line in reader]

        pols = [line['label'] for line in content]
        print('[Info] all labels:', set(pols))

    new_content = []
    for line in tqdm(content):
        title = line['title']
        body = line['text']
        pol = line['label']
        new_pol = str(int(pol) + 1)

        doc = '. '.join([title, body])
        doc_tok = nlp.word_tokenize(doc)
        new_content.append([doc_tok, new_pol])

        import pdb;pdb.set_trace()

    new_file = file.replace('.csv', '_tok.csv')
    with open(new_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_content)


