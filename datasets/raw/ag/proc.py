import csv
try:
    import tqdm
    import efficiency
except ImportError:
    import os
    os.system('pip install tqdm efficiency')

from tqdm import tqdm
from efficiency.nlp import NLP

train_file = 'train.csv'
test_file = 'test.csv'
nlp = NLP()

for file in [train_file, test_file]:
    with open(file) as f:
        reader = csv.reader(f)
        content = list(reader)

    new_content = []
    for pol, title, body in tqdm(content):
        body = body.replace('\\', ' ')
        doc = '. '.join([title, body])
        doc_tok = nlp.word_tokenize(doc)
        new_content.append([doc_tok, pol])

    new_file = file.replace('.csv', '_tok.csv')
    with open(new_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_content)


