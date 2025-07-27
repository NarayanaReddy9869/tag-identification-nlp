import urllib.request
from bs4 import BeautifulSoup
import nltk
import matplotlib.pyplot as plt

# Download necessary data
nltk.download('stopwords')
nltk.download('punkt')  # for word_tokenize
nltk.download('averaged_perceptron_tagger')  # for POS tagging

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Fetch webpage
response = urllib.request.urlopen('https://en.wikipedia.org/wiki/Tata_Group')
html = response.read()

# Parse and extract text
soup = BeautifulSoup(html, 'html5lib')
text = soup.get_text(strip=True)

# Tokenize
tokens = word_tokenize(text)

# Clean tokens (remove stopwords)
stop_words = set(stopwords.words('english'))
clean_tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]

# Frequency Distribution
freq = nltk.FreqDist(clean_tokens)

# Plot top 50 frequent words
freq.plot(50, cumulative=False)

# POS Tagging (optional - "tag identification")
tagged = nltk.pos_tag(clean_tokens[:100])  # tagging only first 100 to limit output
print("\nPOS Tags Sample (first 100 tokens):")
for word, tag in tagged:
    print(f"{word} --> {tag}")
