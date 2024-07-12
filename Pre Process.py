import spacy
import numpy as np
import nltk
from nltk.corpus import cmudict
from PyPDF2 import PdfReader
import random

nltk.download('cmudict')

def process_doc(doc):
    input_list = np.array([])
    syllable_count_list = np.array([])

    d = cmudict.dict()

    token_count = 0
    for token in doc:
        if token.is_alpha:
            input_list = np.append(input_list, token.text)
            word_lower = token.text.lower()
            if word_lower in d:
                syllables = [len(list(y for y in x if y[-1].isdigit())) for x in d[word_lower]]
                syllable_count_list = np.append(syllable_count_list, syllables[0] if syllables else 0)
            token_count += 1
            if token_count >= 100:
                break

    syllable_sum = np.sum(syllable_count_list >= 3)

    return input_list, syllable_sum

nlp = spacy.load("en_core_web_sm")

data = []

for _ in range(3):
    input_array = []
    true_classes = []

    # Open the PDF file
    with open(r'C:\Users\blake\Desktop\Machine Learning\Main\Books\College+ [4]\Cosmos.pdf', 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)
        
        # Choose a random page to extract from
        random_page_num = random.randint(0, len(pdf_reader.pages) - 1)
        random_page = pdf_reader.pages[random_page_num]
        
        # Read text from the random page
        text = random_page.extract_text()
        
        # Process the text using spaCy
        doc = nlp(text)
        input_list, syllable_sum = process_doc(doc)
        input_array.append([len(word) for word in input_list])

        if syllable_sum <= 1:
            true_class = 0
        elif 1 < syllable_sum <= 5:
            true_class = 1
        elif 5 < syllable_sum <= 8:
            true_class = 2
        elif 8 < syllable_sum <= 11:
            true_class = 3
        else:
            true_class = 4
        true_classes.append(true_class)

    for i, (input_data, true_class) in enumerate(zip(input_array, true_classes), start=1):
        print(f"Book {i}:")
        print("Input:", input_data)
        print("True Class:", true_class)
        print()
        data.append((input_data, true_class))

print(data)
