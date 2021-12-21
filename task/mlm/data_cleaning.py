import re
import string


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


# remove punctuations علامات الترقيم
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


# start processing the tweet
arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def preprocess(text: str, is_tweet: bool = False):
    # Replace @username with empty string
    text = re.sub('@[^\s]+', ' ', str(text))

    # Replace username with empty string
    text = re.sub('USERNAME', ' ', str(text))
    # Replace RT with empty string
    text = re.sub('RT', ' ', str(text))

    # Convert www.* or https?://* to " "
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', str(text))
    text = re.sub('URL', ' ', str(text))

    if is_tweet:
        # remove numbers
        text = re.sub('[0-9]+', '', str(text))

    # remove Dicarities
    text = re.sub(arabic_diacritics, '', str(text))
    # remove emojies

    # Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', str(text))

    if is_tweet:
        # remove English words
        text = re.sub('[a-zA-Z0-9]+', '', str(text))

        # remove punctuations
        text = remove_punctuations(text)

    # normalize the text
    text = normalize_arabic(text)

    # remove repeated letters
    text = remove_repeating_char(text)
    text = deEmojify(text)

    # remove multiple spaces
    text = re.sub("\s+", " ", text)

    return text