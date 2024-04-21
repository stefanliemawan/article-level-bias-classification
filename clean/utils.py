import re


def load_word_list():
    with open("../word_list/words_alpha.txt") as word_file:
        word_list = set(word_file.read().split())

    return word_list


word_list = load_word_list()
print(len(word_list))


def fix_conjoined_words(text):
    # pattern = r"([a-z])([A-Z])"
    # corrected_text = re.sub(pattern, r"\1 \2", text)

    words = text.split()
    corrected_words = [
        word if word.lower() in word_list else suggest_correction(word)
        for word in words
    ]
    corrected_text = " ".join(corrected_words)

    return corrected_text


def choose_best_word(correct_word_candidates):
    for correct_word_candidate in correct_word_candidates:
        if "the " in correct_word_candidate:
            return correct_word_candidate

    return correct_word_candidates[0]


def suggest_correction(input_word):
    word_candidates = []

    for word in word_list:
        if word in input_word:
            word_candidates.append(word)

    correct_word_candidates = []

    for word1 in word_candidates:
        for word2 in word_candidates:
            joined_word = word1 + word2

            if joined_word == input_word:
                correct_word_candidates.append(word1 + " " + word2)

    if len(correct_word_candidates) == 0:
        for word1 in word_candidates:
            for word2 in word_candidates:
                for word3 in word_candidates:
                    joined_word = word1 + word2 + word3

                    if joined_word == input_word:
                        correct_word_candidates.append(
                            word1 + " " + word2 + " " + word3
                        )

    if len(correct_word_candidates) == 1:
        return correct_word_candidates[0]
    elif len(correct_word_candidates) > 1:
        return choose_best_word(correct_word_candidates)

    return input_word


# corrected = suggest_correction("profileDeepfakevideos")
# print(corrected)
