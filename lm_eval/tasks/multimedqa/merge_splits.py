from datasets import load_dataset, concatenate_datasets, DatasetDict


BASE = 'hails/mmlu_no_train'


MULTIMEDQA_SUBJECTS = [
    'anatomy',
    'clinical_knowledge',
    'college_biology',
    'college_medicine',
    'medical_genetics',
    'professional_medicine',
]


if __name__ == '__main__':
    for subject in MULTIMEDQA_SUBJECTS:
        subdata = load_dataset(BASE, subject)
        out_data = DatasetDict({
            'training': concatenate_datasets([subdata['validation'], subdata['dev']]),
            'test': subdata['test']
        })

        out_data.push_to_hub(f'medarc/mmlu_{subject}')
