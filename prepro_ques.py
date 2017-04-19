import json
import argparse
from nltk.tokenize import word_tokenize

def main():
    question = word_tokenize(args.question.replace('?', ' ? ').strip().lower())[:args.ques_len]
    history_facts = args.history.replace('?', ' ? ').split(args.delimiter)
    history, questions = [], []
    for i in history_facts:
        fact = word_tokenize(i.strip().lower())[:args.fact_len]
        if len(fact) != 0:
            history.append(fact)
            try:
                questions.append(fact[:fact.index('?')+1])
            except:
                pass

    num_hist = min(len(history), 10)
    num_ques = num_hist - 1 if num_hist > 0 else 0
    json.dump({'question': question, 'history': history[-num_hist:], 'questions': questions[-num_ques:]}, open('ques_feat.json', 'w'))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-question', type=str, default='')
    parser.add_argument('-ques_len', type=int, default=15)
    parser.add_argument('-history', type=str, default='')
    parser.add_argument('-fact_len', type=int, default=30)
    parser.add_argument('-delimiter', type=str, default='||||')
    args = parser.parse_args()
    main()
