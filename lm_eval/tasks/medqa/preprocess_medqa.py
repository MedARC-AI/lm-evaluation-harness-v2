import regex as re

import numpy as np
np.random.seed(1992)


LETTER_OPTIONS = ['A', 'B', 'C', 'D']


def doc_to_text(doc) -> str:
    option_choices = {'A': doc['ending0'], 'B': doc['ending1'], 'C': doc['ending2'], 'D': doc['ending3']}
    answers = "".join((f'{k}. {v}\n') for k, v in option_choices.items())
    return f"Question: {doc['sent1']}\n{answers}Answer:"


def doc_to_target(doc) -> int:
    return doc['label']


def doc_to_choice(doc, return_letters=True):
    return LETTER_OPTIONS if return_letters else [doc[f'ending{i}'] for i in range(len(LETTER_OPTIONS))]


def doc_to_text_medprompt(doc):
    question = doc['sent1']
    text = f'<<Question:>> {question}\n----\n'
    return text


def doc_to_fewshot_text(doc):
    question = doc['sent1']
    # Include rationale if in dataset (this means either that it was pre-computed or is part of fewshot context)
    explanation_str = doc.get('rationale', '')
    if len(explanation_str) == 0:
        print('Warning. No CoT rationales have been pre-computed.')
    choice_str = '\n'.join([f"{l}) {doc['ending' + str(i)]}" for i in range(len(LETTER_OPTIONS))])
    text = f'<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Explanation:>> {explanation_str}\n----\n<<Final Answer:>>'
    return text


def letter_target(doc):
    return LETTER_OPTIONS[doc['label']]


def shuffled_choice_list(letters, options, shuffle=True):
    n = len(letters)
    order = np.arange(n)
    if shuffle:
        np.random.shuffle(order)

    options_shuffled = [options[i] for i in order]
    unshuffle_map = {letters[new_idx]: letters[old_idx] for new_idx, old_idx in enumerate(order)}

    def unshuffle_answer_callback(output):
        pattern = r'<{0,2}Final Answer:?>{0,2}:?\s?([A-D])'
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is None:
            pattern = r'(?:answer )?is\W*([A-D])'
            match = re.search(pattern, output, flags=re.IGNORECASE)
            if match is None:
                option_str = '|'.join(list(map(re.escape, options)))
                literal_pattern = r'<{0,2}Final Answer:?>{0,2}:?\s?(' + option_str + ')'
                match = re.search(literal_pattern, output, flags=re.IGNORECASE)
                if match is None:
                    print(f'Answer Not Found! Check output below.\n{output}. Returning [invalid].')
                    return '[invalid]'
                else:
                    option_idx = options.index(match.group(1).strip())
                    return output[:match.start()] + '<<Final Answer:>> ' + letters[option_idx]
        return output[:match.start()] + '<<Final Answer:>> ' + unshuffle_map[match.group(1).strip().upper()]

    shuffled_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, options_shuffled)])
    return f'<<Choices:>>\n{shuffled_str}\n----\n<<Explanation:>>', unshuffle_answer_callback


