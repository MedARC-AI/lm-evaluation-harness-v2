from dataclasses import dataclass
import os

import argparse
from datasets import DatasetDict
from openai import AzureOpenAI
import numpy as np
np.random.seed(1992)
from textwrap import dedent
import regex as re

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.api.embeddings import MistralEmbeddings


EMBEDDING_TASK_DESCRIPTION = 'Given a medical question, retrieve related medical questions'

@dataclass
class CotConfig:
    name: str
    input_type: str = None
    contexts_col: str = None
    question_col: str = 'question'


CotConfigs = {
    'pubmedqa_medprompt': CotConfig(name='pubmedqa', input_type='abstract', contexts_col='CONTEXTS', question_col='QUESTION'),
    'medmcqa_medprompt': CotConfig(name='medmcqa', question_col='question'),
    'medqa_medprompt': CotConfig(name='medqa', question_col='sent1'),
    'mmlu_anatomy': CotConfig(name='mmlu_anatomy', question_col='question'),
    'mmlu_clinical_knowledge': CotConfig(name='mmlu_clinical_knowledge', question_col='question'),
    'mmlu_college_biology': CotConfig(name='mmlu_college_biology', question_col='question'),
    'mmlu_college_medicine': CotConfig(name='mmlu_college_medicine', question_col='question'),
    'mmlu_medical_genetics': CotConfig(name='mmlu_medical_genetics', question_col='question'),
    'mmlu_professional_medicine': CotConfig(name='mmlu_professional_medicine', question_col='question'),
}


def chatgpt(client, messages, model='gpt-4', temperature=0.1, max_tokens=2048):
    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def build_prompt(doc, task, cot_config):
    """
    Prompt is taken from MedPrompt
    https://github.com/microsoft/promptbase/blob/90fe3f1e2476638ae7e623687bfe9b8b2077b2bb/src/promptbase/drop/drop.py#L98
    """

    choice_letters = task.doc_to_choice(doc, return_letters=True)
    choice_str = ', '.join(choice_letters)
    question = doc[cot_config.question_col]

    if 'pubmed' in cot_config.name:
        input = '\n'.join(doc[cot_config.contexts_col])
        input_type = cot_config.input_type

        prompt = dedent(f"""
        Answer the following reading comprehension **Question** based on the **{input_type}** below.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the **Final Answer** from the choice set: {choice_str}.
        ----
        **{input_type}:** {input}
        ----
        **Question:** {question}
        ----
        **Explanation**: """
        )
    elif 'medmcqa' in cot_config.name:
        choice_options = [
            doc['opa'],
            doc['opb'],
            doc['opc'],
            doc['opd'],
        ]

        choice_str = []
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----s
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}\n
        ----
        **Explanation**: """
        )
    elif 'medqa' in cot_config.name:
        choice_options = [
            doc['ending0'],
            doc['ending1'],
            doc['ending2'],
            doc['ending3'],
        ]

        choice_str = []
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----s
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}\n
        ----
        **Explanation**: """
        )
    else:
        assert 'mmlu' in cot_config.name
        choice_options = [doc[l] for l in choice_letters]

        choice_str = []
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----s
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}\n
        ----
        **Explanation**: """
        )

    prompt = '\n'.join([x.lstrip() for x in prompt.split('\n')])

    return prompt

def generate_self_cot(doc, task, cot_config, lm_obj, embedding_model, add_self_cot=True, consistency_filter=True):
    q_embed = embedding_model(doc)
    new_cols = {f'{cot_config.question_col}_embed': q_embed, 'rationale': ''}
    if not add_self_cot:  # We don't pre-compute CoT for every split. Only "fewshot_split"
        return new_cols

    prompt = build_prompt(doc, task, cot_config)

    print(prompt)

    # Empty gen config for now
    args = (prompt, {})

    instance = lm_eval.api.instance.Instance(
        request_type='generate_until',
        arguments=args,
        doc=doc,
        idx=0
    )

    if type(lm_obj) == AzureOpenAI:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant for medical question answering.'},
            {'role': 'user', 'content': prompt}
        ]
        try:
            prediction = chatgpt(client=lm_obj, messages=messages)
        except Exception as e:
            print(e)
            print('Skipping this for CoT.')
            return new_cols
    else:
        prediction = lm_obj.generate_until(requests=[instance])[0]

    if type(prediction) != str:
        print(f'Model returned prediction of type {type(prediction)} -> {prediction}. Returning with no CoT')
        return new_cols

    split = re.split(re.escape('**Final Answer'), prediction)
    if len(split) != 2:
        print('Invalid output format. Check your prompt.')
        return new_cols

    rationale, answer_raw = split
    rationale = rationale.strip()

    letters = task.doc_to_choice(doc, return_letters=True)

    choice_regex = r'|'.join(letters)
    answer_match = re.search(rf'({choice_regex})', answer_raw, flags=re.IGNORECASE)
    if answer_match is None:
        print(f'Expected one of {choice_regex}. Got {answer_raw}. Check your prompt.')
        return new_cols

    answer_lower = answer_match.group().lower()
    target = task.doc_to_target(doc)


    if type(target) == int:
        target = letters[target]
    
    assert target in letters
    target_lower = target.lower()

    if consistency_filter and answer_lower != target_lower:
        print(f'Answer ({answer_lower}) didn\'t match ground truth target ({target_lower}). Not adding to CoT dataset.')
        return new_cols

    new_cols.update({'rationale': rationale})

    return new_cols


if __name__ == '__main__':
    # generate and cache self-COT examplars for few-shot https://github.com/microsoft/promptbase
    parser = argparse.ArgumentParser('Pre-Compute CoT rationales for MultiMedQA')

    parser.add_argument('--task', default='pubmedqa_medprompt')
    parser.add_argument('--fewshot_split', default='training', choices=['training', 'validation', 'test'])
    parser.add_argument('--max_cot_examples', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--cot_model', default='gpt-4')
    parser.add_argument('--cot_model_type', default='openai', choices=['openai', 'huggingface'])
    parser.add_argument('-remove_consistency_filter', default=False, action='store_true')

    args = parser.parse_args()

    cot_config = CotConfigs[args.task]

    output_splits = {}
    args.save_dir = f'~/.cache/huggingface/datasets/{args.task}_{args.cot_model}_preprocessed'
    args.hub_dir = f'medarc/{args.task}_{args.cot_model}_preprocessed'

    lm_eval.tasks.initialize_tasks()

    # Get Task object
    task = lm_eval.tasks.get_task_dict(args.task)[args.task]

    print('Initializing embeddings...')
    embedding_model = MistralEmbeddings(
        fewshot_col=cot_config.question_col, device=args.device,
        task_description=EMBEDDING_TASK_DESCRIPTION
    )

    # Download the task from HF or HF cache
    task.download()

    for split in ['training', 'validation', 'test']:
        add_self_cot = split == args.fewshot_split
        assert getattr(task, f'has_{split}_docs')()
        cot_data = getattr(task, f'{split}_docs')()
        n = len(cot_data)
        if add_self_cot and args.max_cot_examples < n:
            data_idxs = np.arange(n)
            np.random.shuffle(data_idxs)
            sample_idxs = data_idxs[:args.max_cot_examples]
            # TODO: Can shuffle first for randomization but this is just for debugging purposes
            print(f'Randomly sampling {args.max_cot_examples} examples from {n} total...')
            cot_data = cot_data.select(sample_idxs)

        if args.cot_model_type == 'openai':
            assert 'OPENAI_API_KEY' in os.environ
            lm_obj = AzureOpenAI(
                api_key=os.environ.get('OPENAI_API_KEY'),
                azure_endpoint='https://east-us-2-llm.openai.azure.com/',
                api_version='2023-05-15',
                azure_deployment='misc-gpt4-turbo'
            )
        else:
            lm_obj = HFLM(
                pretrained=args.cot_model,
                device='cuda:0',
                batch_size=args.batch_size,
            )

        cot_data_w_cot = cot_data.map(
            lambda doc: generate_self_cot(
                doc, task, cot_config, lm_obj, embedding_model, add_self_cot=add_self_cot,
                consistency_filter=not args.remove_consistency_filter
            )
        )

        cot_data_w_cot_valid = cot_data_w_cot.filter(lambda example: not add_self_cot or len(example['rationale']) > 0)
        valid_n = len(cot_data_w_cot_valid)

        print(f'Recorded {valid_n} / {n} CoT examples for {split} split')
        output_splits[split] = cot_data_w_cot_valid

    print(f'Saving to {args.save_dir} and to {args.hub_dir}...')
    dataset = DatasetDict(output_splits)
    dataset.save_to_disk(args.save_dir)
    dataset.push_to_hub(args.hub_dir, private=True)
