import json
import random

CBT_TECHNIQUE = {"debate": 'db', "divergent": 'dv', "possible": 'ps', "benefit": 'bn'}


def get_formatted_data():
    all_data = []
    for k in CBT_TECHNIQUE.keys():
        with open('C-TIND/{}-v2.json'.format(k), 'r') as file:
            data = json.load(file)
        
        formatted_data = []
        role_dict = {'assistant': 'Counselor', 'user': 'Client'}
        for conv in data:
            input_text = " ".join([f"{role_dict[turn['from']]}: {turn['value']}\n" for turn in conv['conversations']])            
            summary_text = conv['summary_text']
            formatted_data.append({
                'instruction': "Summarize the following counseling conversation.",
                'input': input_text,
                'output': summary_text
            })

        all_data.extend(formatted_data)
        with open('C-TIND/instruct_{}.json'.format(k), 'w') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)
    
    with open('C-TIND/instruct_all.json', 'w') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)


def split_formatted_data():
    get_formatted_data()
    
    with open('C-TIND/instruct_all.json', 'r') as f:
        data = json.load(f)
    
    random.seed(0)
    random.shuffle(data)
    test_data = data[:400]
    train_data = data[400:]

    with open('C-TIND/test-v2.json', 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    # with open('C-TIND/valid.json', 'w') as f:
    #     json.dump(valid_data, f, ensure_ascii=False, indent=4)
    with open('C-TIND/train-v2.json', 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
        

def gen_prompt(item, unk=True, examples=False):

    if unk:
        prompt_v1 = 'Please understand and summarize the important content in a given counseling conversation.\n' \
        'In a given conversation, each sentence contains its speaker (i.e., Counselor or Client), and its content, written in this format: #[speaker]: [content].  You should summarize the client\'s situation, original emotion, origin thought, and the transformed thought after the conversation. In addition, you should provide advice for the client.\n' \
        'Specifically, you should assume the role of the client and use the client\'s tone to perform a 6-step reasoning process: \n' \
        '1. Output the "Situation": Briefly summarize the client\'s current situation in one Sentence, which is also the core event discussed in the conversation.\n' \
        '2. Output the "Emotion": Summarize the client\'s original emotions with no more than three words.\n' \
        '3. Output the "Original Thought": Extract the client\'s original thoughts at the beginning of the conversation in one sentence.\n' \
        '4. Output the "Thought Evaluation": extract the key elements based on the CBT technique used by the counselor in this conversation.\n' \
        '- If use "prosecution-defense" technique: extract and list evidence supporting the "Original Thought" as "Supporting Evidence", and extract and list evidence against the "Original Thought" as "Opposing Evidence".\n' \
        '- if use "region-of-possibilities" technique: extract and list the best-case scenario and corresponding evidence as "Best Case", the worst-case scenario and corresponding evidence as "Worst Case", the most possible scenario and corresponding evidence as "Most Possible Case".\n' \
        '- if use "divergent thinking" technique: extract and list other potential explanations as "Other Explanations".\n' \
        '- if use "cost-benefit" analysis: extract and list benefits of holding the "Original Thought" as "Benefits", and extract and list drawbacks of holding the "Original Thought" as "Costs".\n' \
        '5. Output the "Alternative Thought": Extract the client\'s transformed thought at the end of the conversation in one sentence.\n' \
        '6. Output the "Conclusion": Based on the conversation content and "Alternative Thought", generate brief suggestions for the client in one sentence.\n'
        
        prompt_v2 = prompt_v1 + 'A few generated examples are shown below:\n' \
        'Situation: I\'m feeling anxious because I\'m constantly worried that something bad will happen to the people I love when they are away from me.\nEmotion: anxious, nervous, stressed\nOriginal Thought: I constantly think about all the worst-case scenarios that could happen to my loved ones to prepare for the worst.\nEvaluation Process: - Benefits: Staying on alert and being aware of potential dangers; feeling like worrying is a way to protect my loved ones. \n- Costs: Feeling anxious all the time, difficulty relaxing and enjoying life, strain on relationships because I\'m too overbearing.\nAlternative Thought: I care about my loved ones and will take reasonable steps to ensure their safety, but I can’t control everything, and constant worrying doesn’t help them or me.\nConclusion: I should remind myself that I can\'t control everything and try relaxation techniques like deep breathing or meditation to reinforce this new perspective and reduce anxiety.\n' \
        'Situation: My 14-year-old daughter has been skipping school and lying to me, which worries me greatly.\nEmotion: upset, frustrated, scared\nOriginal Thought: I\'m worried that if I don\'t address her lying now, she\'ll end up becoming a habitual liar.\nEvaluation Process: - Supporting Evidence: She skips school and lies about it; she lies even about small things like chores or homework; lying is her go-to response in difficult situations. \n- Opposing Evidence: She\'s told the truth in tough situations before; she\'s usually honest about chores and homework when not in treacherous situations; she can be open about her feelings when she feels safe.\nAlternative Thought: While my daughter has been lying recently, it\'s possible with the right support and guidance, she can learn to communicate more honestly and might not become a habitual liar.\nConclusion: Approach her with understanding rather than panic, create a safe space for honesty, and work on the underlying issues together.\n' \
        'Situation: I didn\'t get into graduate school, which I had hoped would occur without needing exams.\nEmotion: lost, hopeless\nOriginal Thought: I can\'t do anything well, and my future feels hopeless after not getting accepted.\nEvaluation Process: Other Explanations: \n1. The competition was particularly tough this year. \n2. I may not have met all the specific requirements. \n3. I didn\'t focus enough on certain aspects of my application, like my personal statement or recommendation letters.\nAlternative Thought: Not getting into grad school doesn\'t mean I\'m a failure – it was a competitive process, and there are specific areas I can work on for next time.\nConclusion: By recognizing the multiple reasons why I might not have been accepted and viewing this experience as a learning opportunity, I can focus on refining specific parts of my application for future attempts.\n' \
        'Situation: I planned to be productive and study during my vacation, but ended up procrastinating and not meeting my goals.\nEmotion: hopeless, frustrated\nOriginal Thought: I lack self-discipline and cannot stick to my goals, reflecting my greater life issues.\nEvaluation Process: - Best Case: I improve my self-discipline, stick to plans, and achieve my goals, which would make me feel better about myself (evidence: past success during exam periods). \n- Worst Case: I fail to stick to any future goals, and my life never improves (evidence: a history of not adhering to plans). \n- Most Possible Case: I\'ll likely make small improvements over time, even if not turning things around completely right away (evidence: mixed history with both failures and successes).\nAlternative Thought: I\'m more hopeful now, understanding that the worst-case scenario isn\'t inevitable and recognizing there\'s a range of possibilities for improvement.\nConclusion: I should reassess my goals to make them more realistic, establish a structured routine with reminders, and consider having an accountability partner to help stay on track for positive changes.\n'
        
        return prompt_v2 if examples else prompt_v1
    

    key_dict = {'benefit': ['cost-benefit', 'cost benifit', 'cost-benefit', 'benefits:'],
                'debate': ['prosecution-defense', 'prosecution defense', 'evidence:'],
                'divergent': ['divergent-thinking', 'divergent thinking', 'explanations:'],
                'possible': ['region-of-possibilities', 'region of possibilities', 'case:']}
    k_type = None
    for k, v in key_dict.items():
        for vi in v:
            if vi in item['input'].lower() or vi in item['output'].lower():
                k_type = k
                break
    
    if k_type is None:
        return None

    CBT_PROMPT = {
    "debate": 'Output the "Thought Evaluation": Extract and list evidence supporting the "Original Thought" as "Supporting Evidence", and extract and list evidence against the "Original Thought" as "Opposing Evidence".',
    "divergent": 'Output the "Thought Evaluation": Extract and list other potential explanations as "Other Explanations".',
    "possible": 'Output the "Thought Evaluation": Extract and list the best-case scenario and corresponding evidence as "Best Case", the worst-case scenario and corresponding evidence as "Worst Case", the most possible scenario and corresponding evidence as "Most Possible Case".',
    "benefit": 'Output the "Thought Evaluation": Extract and list benefits of holding the "Original Thought" as "Benefits", and extract and list drawbacks of holding the "Original Thought" as "Costs".'
    }

    prompt_example = {
    "benefit": 'Situation: I\'m feeling anxious because I\'m constantly worried that something bad will happen to the people I love when they are away from me.\nEmotion: anxious, nervous, stressed\nOriginal Thought: I constantly think about all the worst-case scenarios that could happen to my loved ones to prepare for the worst.\nEvaluation Process: - Benefits: Staying on alert and being aware of potential dangers; feeling like worrying is a way to protect my loved ones. \n- Costs: Feeling anxious all the time, difficulty relaxing and enjoying life, strain on relationships because I\'m too overbearing.\nAlternative Thought: I care about my loved ones and will take reasonable steps to ensure their safety, but I can’t control everything, and constant worrying doesn’t help them or me.\nConclusion: I should remind myself that I can\'t control everything and try relaxation techniques like deep breathing or meditation to reinforce this new perspective and reduce anxiety.\n',
    "debate": 'Situation: My 14-year-old daughter has been skipping school and lying to me, which worries me greatly.\nEmotion: upset, frustrated, scared\nOriginal Thought: I\'m worried that if I don\'t address her lying now, she\'ll end up becoming a habitual liar.\nEvaluation Process: - Supporting Evidence: She skips school and lies about it; she lies even about small things like chores or homework; lying is her go-to response in difficult situations. \n- Opposing Evidence: She\'s told the truth in tough situations before; she\'s usually honest about chores and homework when not in treacherous situations; she can be open about her feelings when she feels safe.\nAlternative Thought: While my daughter has been lying recently, it\'s possible with the right support and guidance, she can learn to communicate more honestly and might not become a habitual liar.\nConclusion: Approach her with understanding rather than panic, create a safe space for honesty, and work on the underlying issues together.\n',
    "divergent": 'Situation: I didn\'t get into graduate school, which I had hoped would occur without needing exams.\nEmotion: lost, hopeless\nOriginal Thought: I can\'t do anything well, and my future feels hopeless after not getting accepted.\nEvaluation Process: Other Explanations: \n1. The competition was particularly tough this year. \n2. I may not have met all the specific requirements. \n3. I didn\'t focus enough on certain aspects of my application, like my personal statement or recommendation letters.\nAlternative Thought: Not getting into grad school doesn\'t mean I\'m a failure – it was a competitive process, and there are specific areas I can work on for next time.\nConclusion: By recognizing the multiple reasons why I might not have been accepted and viewing this experience as a learning opportunity, I can focus on refining specific parts of my application for future attempts.\n',
    "possible": 'Situation: I planned to be productive and study during my vacation, but ended up procrastinating and not meeting my goals.\nEmotion: hopeless, frustrated\nOriginal Thought: I lack self-discipline and cannot stick to my goals, reflecting my greater life issues.\nEvaluation Process: - Best Case: I improve my self-discipline, stick to plans, and achieve my goals, which would make me feel better about myself (evidence: past success during exam periods). \n- Worst Case: I fail to stick to any future goals, and my life never improves (evidence: a history of not adhering to plans). \n- Most Possible Case: I\'ll likely make small improvements over time, even if not turning things around completely right away (evidence: mixed history with both failures and successes).\nAlternative Thought: I\'m more hopeful now, understanding that the worst-case scenario isn\'t inevitable and recognizing there\'s a range of possibilities for improvement.\nConclusion: I should reassess my goals to make them more realistic, establish a structured routine with reminders, and consider having an accountability partner to help stay on track for positive changes.\n'

}
    

    if not examples:
        prompt = 'Please understand and summarize the important content in a given counseling conversation.\n' \
            'In a given conversation, each sentence contains its speaker (i.e., Counselor or Client), and its content, written in this format: #[speaker]: [content].  You should summarize the client\'s situation, original emotion, origin thought, and the transformed thought after the conversation. In addition, you should provide advice for the client.\n' \
            'Specifically, you should assume the role of the client and use the client\'s tone to perform a 6-step reasoning process:\n' \
            '1. Output the "Situation": Briefly summarize your current situation in one Sentence, which is also the core event discusssed in the conversation.\n' \
            '2. Output the "Emotion": Summarize your original emotions with no more than three words.\n' \
            '3. Output the "Original Thought": Extract your original thoughts at the beginning of the conversation in one sentence.\n' \
            '4. {}\n' \
            '5. Output the "Alternative Thought": Extract your transformed thought at the end of the conversation in one sentecne.\n' \
            '6. Output the "Conclusion": Based on the conversation content and "Alternative Thought", summarize brief suggestions that were given to you in one sentence.\n'.format(CBT_PROMPT[k_type])
    else:
        prompt = 'Please understand and summarize the important content in a given counseling conversation.\n' \
            'In a given conversation, each sentence contains its speaker (i.e., Counselor or Client), and its content, written in this format: #[speaker]: [content].  You should summarize the client\'s situation, original emotion, origin thought, and the transformed thought after the conversation. In addition, you should provide advice for the client.\n' \
            'Specifically, you should assume the role of the client and use the client\'s tone to perform a 6-step reasoning process:\n' \
            '1. Output the "Situation": Briefly summarize your current situation in one Sentence, which is also the core event discusssed in the conversation.\n' \
            '2. Output the "Emotion": Summarize your original emotions with no more than three words.\n' \
            '3. Output the "Original Thought": Extract your original thoughts at the beginning of the conversation in one sentence.\n' \
            '4. {}\n' \
            '5. Output the "Alternative Thought": Extract your transformed thought at the end of the conversation in one sentecne.\n' \
            '6. Output the "Conclusion": Based on the conversation content and "Alternative Thought", summarize brief suggestions that were given to you in one sentence.\n' \
            'A generated example is shown below:\n' \
            '{}\n'.format(CBT_PROMPT[k_type], prompt_example[k_type])


    return prompt

def statistic():
    all_data = 0
    all_num = 0
    for k in CBT_TECHNIQUE.keys():
        avg_len = 0
        with open('C-TIND/{}-v2.json'.format(k), 'r') as file:
            data = json.load(file)
        for item in data:
            avg_len += len(item['summary_text'].split())
        all_data += avg_len
        all_num += len(data)
        avg_len /= len(data)
        print('Average length of {} is: {:.3f}'.format(k, avg_len))
    print('Average length of all is: {:.3f}'.format(all_data / all_num))

if __name__ == '__main__':
    # with open('C-TIND/test-v2.json', 'r') as f:
    #     test_data = json.load(f)


    # new_test_data = []
    # for item in test_data:
    #     prompt = gen_prompt(item, unk=False, examples=True)
    #     if prompt:
    #         item['instruction'] = prompt
    #         new_test_data.append(item)
    

    # with open('C-TIND/test-prompt-k-v2.json', 'w') as f:
    #     json.dump(new_test_data, f, ensure_ascii=False, indent=4)
    statistic()



