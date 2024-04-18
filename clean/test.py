import re

from english_words import get_english_words_set

web2_word_list = get_english_words_set(["web2"], lower=True)


def fix_conjoined_words(text):
    # Define a regular expression pattern to split words based on lowercase followed by uppercase letters
    pattern = r"([a-z])([A-Z])"

    # Split conjoined words based on the pattern and insert spaces between them
    corrected_text = re.sub(pattern, r"\1 \2", text)

    # Split the corrected text into individual words
    words = corrected_text.split()

    # Validate and correct words
    corrected_words = [
        word if word in web2_word_list else suggest_correction(word) for word in words
    ]

    # Join the corrected words back into a string
    corrected_text = " ".join(corrected_words)

    return corrected_text


def suggest_correction(word):
    # This function can suggest a correction for a misspelled or unknown word
    # Here, we simply return the word as is, but you can implement a more sophisticated correction logic
    return word


input_text = "Hello! This is a Sentence.With an Example.Of loomingexpiration and otherConjoinedWords."
print(input_text)

fixed_text = fix_conjoined_words(input_text)

print(fixed_text)
