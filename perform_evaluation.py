import argparse

from evaluation import eval_recall, eval_fact_checking, eval_question_answering

def main(args):
    if args.dataset in ['fm2', 'fever']:
        em, length = eval_fact_checking(args.data_result)
        print("Accuracy: ", em, " Avg Length: ", length)
    elif args.dataset in ['nq', 'tqa', 'webq']:
        em, length = eval_question_answering(args.data_result)
        print("Exact Match: ", em, " Avg Length: ", length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=None, type=str, help='fever, fm2', required=True)
    parser.add_argument('--data_result', default=None, type=str, required=True)
    args = parser.parse_args()
    main(args)
