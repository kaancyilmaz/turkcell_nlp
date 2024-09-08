# metinlerdeki fazla bosluklari temizle
text = "Hello,      World!     2035" # "Hello, World! 2035"

cleaned_text1 = " ".join(text.split())

print(cleaned_text1)

# %% buyuk -> kucuk harf cevrimi

text = "Hello, World! 2035"
cleaned_text2 = text.lower()
print(cleaned_text2)

# %% noktalama isaretlerini kaldir

import string

text = "Hello, World! 2035"
cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))

print(cleaned_text3)

# %% ozel karakterleri kaldir

import re

text = "Hello, World! 2035"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]","", text)
print(cleaned_text4)

# %% yazim hatalarini duzelt

from textblob import TextBlob

text = "Helıo, Wirld! 2035"
cleaned_text5 = str(TextBlob(text).correct())
print(cleaned_text5)

# %% html yada url etiketlerinin kaldir

from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2035</div>"
cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(cleaned_text6)





























