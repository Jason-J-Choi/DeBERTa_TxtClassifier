import csv
from tqdm import tqdm
try:
    import efficiency
except ImportError:
    import os
    os.system('pip install efficiency')

from efficiency.nlp import NLP

train_file = 'train.csv'
test_file = 'test.csv'
nlp = NLP()
for file in [train_file, test_file]:
    with open(file) as f:
        reader = csv.reader(f)
        content = list(reader)

    new_content = []
    pols, _ = zip(*content)

    for pol, doc in tqdm(content):
        doc_tok = nlp.word_tokenize(doc)
        new_content.append([doc_tok, pol])

    new_file = file.replace('.csv', '_tok.csv')
    with open(new_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_content)


