# should be same a the data format in which model is trained
import re


def clean_text(text: str) -> str:
<<<<<<< HEAD
    # lowercase
    text = text.lower()
    # remove url
    text = re.sub(r"http\S+|www\S+", " ", text)
    #remove gmail
    text = re.sub(r"\S+@\S+", " ", text)
    #remove everything other than lowercase numbers and spaced
    text = re.sub(r"[^a-z\s]", " ", text)
    #replace many spaces with a single space
=======
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
>>>>>>> deb16531cab961bf2aa4c3aa8a76b689f5b6e638
    text = re.sub(r"\s+", " ", text).strip()
    return text
