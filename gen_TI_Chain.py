import json
import logging
import os
import random
import time
from openai import OpenAI
from openai import OpenAIError
import pandas as pd
import re


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CBT_TECHNIQUE = {"debate": 'db', "divergent": 'dv', "possible": 'ps', "benefit": 'bn'}
CBT_PROMPT = {
    "debate": 'Output the "Thought Evaluation": Extract and list evidence supporting the "Original Thought" as "Supporting Evidence", and extract and list evidence against the "Original Thought" as "Opposing Evidence".',
    "divergent": 'Output the "Thought Evaluation": Extract and list other potential explanations as "Other Explanations".',
    "possible": 'Output the "Thought Evaluation": Extract and list the best-case scenario and corresponding evidence as "Best Case", the worst-case scenario and corresponding evidence as "Worst Case", the most possible scenario and corresponding evidence as "Most Possible Case".',
    "benefit": 'Output the "Thought Evaluation": Extract and list benefits of holding the "Original Thought" as "Benefits", and extract and list drawbacks of holding the "Original Thought" as "Costs".'
    }

def call_with_messages(client, prompt, engine_type='gpt-4o'):
    try:
        completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=engine_type,
        )
        response = completion.choices[0].message.content
        response = response.replace('**', '').replace('#', '')
        return response.replace('\n\n', '\n').strip()
    except OpenAIError as e:
        logger.info(f'Request Failed: {e}')
    except Exception as e:
        logger.info('Error: {}'.format(e))

    return None


def format_sum(text, conv_id):
    def find_text(text, keyword):
        index = text.find(keyword)
        if index != -1:
            colon_index =  index + len(keyword) - 1
            return text[colon_index + 1:].strip("\"").strip("\'").strip()
        else:
            return text.strip("\"").strip("\'").strip()
        

    trans_dict = {'Emotion:\n': 'Emotion: ', 'Emotion\n': 'Emotion: ',
                  'Situation:\n': 'Situation: ', 'Situation\n': 'Situation: ',
                  'Thought:\n': 'Thought: ', 'Thought\n': 'Thought: ',
                  'Conclusion:\n': 'Conclusion: ', 'Conclusion\n': 'Conclusion: ',
                  'Thought Evaluation:\n': 'Thought Evaluation: ', 'Thought Evaluation\n': 'Thought Evaluation: '}
    default_keys = {'Situation', 'Emotion', 'Original Thought', 'Evaluation Process', 'Alternative Thought', 'Conclusion'}
    
    if not text:
        return None
    
    for k, v in trans_dict.items():
        text = text.replace(k, v)
    text = text.strip()
    sum_dict = {}
    li = text.split('\n')
    for idx, item in enumerate(li):
        if 'Situation:' in item:
            sum_dict['Situation'] = find_text(item, 'Situation:')
        elif 'Emotion:' in item:
            emo = find_text(item, 'Emotion:').lower().strip('.')
            emo_list = re.split(r',\s*| and\s*', emo)
            emo_list = [e for e in emo_list if e]
            sum_dict['Emotion'] = ', '.join(emo_list).replace('and', '').replace('  ', ' ')
        elif 'Original Thought:' in item:
            sum_dict['Original Thought'] = find_text(item, 'Original Thought:')
        elif 'Thought Evaluation:' in item:
            tmp = []
            for j in range(idx, len(li)):
                if 'Alternative Thought:' in li[j]:
                    break
                else:
                    tmp.append(find_text(li[j], 'Thought Evaluation:'))
            sum_dict['Evaluation Process'] = ' \n'.join(tmp).strip()
        elif 'Alternative Thought:' in item:
            sum_dict['Alternative Thought'] = find_text(item, 'Alternative Thought:')
        elif 'Conclusion:' in item:
            sum_dict['Conclusion'] = find_text(item, 'Conclusion:')

    for k in default_keys:
        if k not in sum_dict:
            logger.info('Incomplete Summerization! {}'.format(conv_id))
            return None
    return sum_dict



def generate_summary(client, data, key_type, output_file, existing=False, format_again=False):
    prompt = 'Please understand and summarize the important content in a given counseling conversation.\n' \
            'In a given conversation, each sentence contains its speaker (i.e., Counselor or Client), and its content, written in this format: #[speaker]: [content].  You should summarize the client\'s situation, original emotion, origin thought, and the transformed thought after the conversation. In addition, you should provide advice for the client.\n' \
            'Specifically, you should assume the role of the client and use the client\'s tone to perform a 6-step reasoning process:\n' \
            '1. Output the "Situation": Briefly summarize your current situation in one Sentence, which is also the core event discusssed in the conversation.\n' \
            '2. Output the "Emotion": Summarize your original emotions with no more than three words.\n' \
            '3. Output the "Original Thought": Extract your original thoughts at the beginning of the conversation in one sentence.\n' \
            '4. {}\n' \
            '5. Output the "Alternative Thought": Extract your transformed thought at the end of the conversation in one sentecne.\n' \
            '6. Output the "Conclusion": Based on the conversation content and "Alternative Thought", summarize brief suggestions that were given to you in one sentence.\n'.format(CBT_PROMPT[key_type])

    new_data = []
    if existing:
        with open(output_file, 'r') as f:
            new_data = json.load(f)
        id_list = [x['id'] for x in new_data]

    if format_again:
        tmp = []
        id_list = []
        for idx, conv in enumerate(new_data):
            if 'summary' in conv.keys() and conv['summary']:
                tmp.append(conv)
                id_list.append(conv['id'])
            elif 'origin_summary' in conv.keys():
                sum_dict = format_sum(conv['origin_summary'], conv['id'])
                new_conv = conv.copy()
                if sum_dict:
                    new_conv['summary'] = sum_dict
                    tmp.append(new_conv)
                    id_list.append(conv['id'])
        new_data = tmp

    print(len(new_data))
    print(len(id_list))
    print(len(data))

    for idx, conv in enumerate(data):
        if existing and conv['id'] in id_list:
            continue

        new_conv = conv.copy()
        history = []
        for utt in conv['conversations']:
            if utt['from'] == 'assistant':
                item = '#Counselor: {}' + utt['value']
            elif utt['from'] == 'user':
                item = '#Client: {}' + utt['value']
            history.append(item)
        cur_prompt = prompt + '\n' + '\n'.join(history)
        random_sleep = random.uniform(20, 40)
        time.sleep(random_sleep)
        summary = call_with_messages(client, cur_prompt)
        sum_dict = format_sum(summary, conv['id'])
        if summary:
            new_conv['origin_summary'] = summary
            new_conv['summary'] = sum_dict
        new_data.append(new_conv)

        if idx % 20 == 0:
            logger.info('Processed Data: {}'.format(idx))
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def gen_summary_text():
    for k in CBT_TECHNIQUE.keys():
        intput_file_path = 'C-TIND/{}.json'.format(k)
        output_file_path = 'C-TIND/{}-v2.json'.format(k)
        with open(intput_file_path, 'r') as f:
            data = json.load(f)
        new_data = []
        for item in data:
            new_item = item.copy()
            if 'summary' not in item.keys() or not item['summary']:
                logger.info('Invalid Data Type: {} in {}'.format(item['id'], intput_file_path))
            else:
                summ = item['summary']
                gen_text = "Situation: {}\n" \
                            "Emotion: {}\n" \
                            "Original Thought: {}\n" \
                            "Evaluation Process: {}\n" \
                            "Alternative Thought: {}\n" \
                            "Conclusion: {}".format(summ['Situation'], summ['Emotion'], summ['Original Thought'],
                                                    summ['Evaluation Process'], summ['Alternative Thought'], summ['Conclusion'])
                new_item['summary_text'] = gen_text
            new_data.append(new_item)
        with open(output_file_path, 'w', encoding='utf8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        logger.info("Save {} data to {}.".format(len(new_data), output_file_path))


if __name__ == '__main__':
    random.seed(1)
    client = OpenAI(api_key=os.environ.get('API_KEY'))

    # for k in CBT_TECHNIQUE.keys():
    #     logger.info('Process Data Type: %s' % k)
    #     new_file_path = 'C-TIND/{}.json'.format(k)
    #     json_file_path = 'full/full_{}.json'.format(k)
    #     with open(json_file_path, 'r') as f:
    #         data = json.load(f)
    #     generate_summary(client, data, k, new_file_path, existing=True, format_again=True)
        # random_sleep = random.uniform(40, 60)
    gen_summary_text()



