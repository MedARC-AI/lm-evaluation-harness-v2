import argparse

import lm_eval
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.models.huggingface import HFLM


DEBUG_MODEL = 'HuggingFaceM4/tiny-random-LlamaForCausalLM'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Debug for MedPrompt')

    parser.add_argument('--model', default='01-ai/Yi-34B')
    parser.add_argument('--task', default='medmcqa_medprompt')
    parser.add_argument('--max_examples', default=10, type=int)
    parser.add_argument('--fewshot_override', default=None, type=int)
    parser.add_argument('-use_hf', default=False, action='store_true')

    args = parser.parse_args()

    if args.model == 'debug':
        args.model = DEBUG_MODEL

    print('Initializing model...')

    if args.use_hf:
        lm_obj = HFLM(pretrained=args.model, device='cuda', batch_size=1)
    else:
        lm_obj = VLLM(pretrained=args.model, device='cuda', batch_size=1)

    print('Initializing tasks...')
    lm_eval.tasks.initialize_tasks()

    print('Running eval...')
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        limit=args.max_examples,
        num_fewshot=args.fewshot_override,
        tasks=[args.task],
    )

    for task, metrics in results['results'].items():
        print(task)
        for k, v in metrics.items():
            try:
                print(f'\t{k} -> {round(v, 3)}')
            except:
                print(f'\t{k} -> {v}')
        print('\n----\n')
