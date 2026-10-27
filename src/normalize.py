
import re
import unicodedata


_CONTRACTIONS = {
    "can't": "cannot", "cannot": "cannot", "won't": "will not",
    "shan't": "shall not", "ain't": "is not", "i'm": "i am",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "i'd": "i would",
    "you'd": "you would", "we'd": "we would", "they'd": "they would",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "mustn't": "must not",
}

_ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor",
    "prof": "professor", "st": "saint", "jr": "junior", "sr": "senior",
    "etc": "et cetera", "vs": "versus",
}

_SPELLING_VARIANTS = {
    "colour": "color", "colours": "colors", "favour": "favor",
    "favours": "favors", "favourite": "favorite", "favourites": "favorites",
    "honour": "honor", "honours": "honors", "labour": "labor",
    "centre": "center", "centres": "centers", "theatre": "theater",
    "theatres": "theaters", "metre": "meter", "metres": "meters",
    "travelling": "traveling", "travelled": "traveled", "defence": "defense",
    "licence": "license", "organisation": "organization",
    "organisations": "organizations", "recognise": "recognize",
    "recognised": "recognized", "realise": "realize", "realised": "realized",
}

_FILLERS = {"uh", "um", "erm", "hmm"}
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = [
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
]
_ORDINAL_TENS = {
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}


def _integer_to_words(value):
    """Convert a non-negative integer below one million to plain English words."""
    if value < 20:
        return _ONES[value]
    if value < 100:
        return _TENS[value // 10] + (" " + _ONES[value % 10] if value % 10 else "")
    if value < 1_000:
        return _ONES[value // 100] + " hundred" + (
            " " + _integer_to_words(value % 100) if value % 100 else ""
        )
    if value < 1_000_000:
        return _integer_to_words(value // 1_000) + " thousand" + (
            " " + _integer_to_words(value % 1_000) if value % 1_000 else ""
        )
    return str(value)


def _expand_numeric_token(match):
    token = match.group(0).replace(",", "")
    if "." in token:
        whole, fractional = token.split(".", 1)
        return _integer_to_words(int(whole)) + " point " + " ".join(_ONES[int(d)] for d in fractional)
    return _integer_to_words(int(token))


def _integer_to_ordinal_words(value):
    """Convert a non-negative integer below one million to an English ordinal."""
    if value < 20:
        return _ORDINALS[value]
    if value < 100:
        if value % 10 == 0:
            return _ORDINAL_TENS[value]
        return _TENS[value // 10] + " " + _ORDINALS[value % 10]
    if value < 1_000:
        if value % 100 == 0:
            return _ONES[value // 100] + " hundredth"
        return _ONES[value // 100] + " hundred " + _integer_to_ordinal_words(value % 100)
    if value < 1_000_000:
        if value % 1_000 == 0:
            return _integer_to_words(value // 1_000) + " thousandth"
        return _integer_to_words(value // 1_000) + " thousand " + _integer_to_ordinal_words(value % 1_000)
    return str(value)


def _expand_currency(match):
    symbol, whole_text, cents_text = match.groups()
    whole = int(whole_text.replace(",", ""))
    unit, subunit = {
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent"),
    }[symbol]
    result = f"{_integer_to_words(whole)} {unit if whole == 1 else unit + 's'}"
    if cents_text is not None:
        cents = int(cents_text.ljust(2, "0"))
        if cents:
            suffix = subunit if cents == 1 or subunit == "pence" else subunit + "s"
            result += f" and {_integer_to_words(cents)} {suffix}"
    return result


def _expand_percent(match):
    token = match.group(1).replace(",", "")
    if "." in token:
        whole, fractional = token.split(".", 1)
        words = _integer_to_words(int(whole)) + " point " + " ".join(_ONES[int(d)] for d in fractional)
    else:
        words = _integer_to_words(int(token))
    return words + " percent"


def _expand_ordinal(match):
    return _integer_to_ordinal_words(int(match.group(1).replace(",", "")))


def normalize_basic(text):
    """Lowercase and standardize whitespace."""
    text = text.lower()
    return " ".join(text.split())


def normalize_standard(text):
    """Lowercase, remove punctuation, normalize whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Replace punctuation with spaces.
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def normalize_custom(text):
    """Product-oriented, auditable ASR normalization.

    Rules are symmetric for reference and prediction: Unicode/case and punctuation,
    hyphens, common contractions and abbreviations, numeric digits, selected
    British/American spellings, and a conservative filler list. These choices are
    intentionally explicit because each one changes what the WER treats as an error.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"([$£€])(\d[\d,]*)(?:\.(\d{1,2}))?", _expand_currency, text)
    text = re.sub(r"\b(\d[\d,]*(?:\.\d+)?)\s*%", _expand_percent, text)
    text = re.sub(r"\b(\d[\d,]*)(?:st|nd|rd|th)\b", _expand_ordinal, text)
    text = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", _expand_numeric_token, text)
    for contraction, expansion in _CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = []
    for token in text.split():
        if token in _FILLERS:
            continue
        replacement = _ABBREVIATIONS.get(token, token)
        tokens.extend(_SPELLING_VARIANTS.get(part, part) for part in replacement.split())
    return " ".join(tokens)
