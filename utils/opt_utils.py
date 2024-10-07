from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def llama_call(model, tokenizer, meta_prompt):

    input_ids = tokenizer.apply_chat_template(
        meta_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        num_return_sequences=10,
        return_dict_in_generate = True, 
        output_logits = True
    )

    print(outputs.keys())
    print(outputs['logits'][1].shape)
    out = outputs['sequences'][:, input_ids.shape[-1]:]
    print(tokenizer.batch_decode(out, skip_special_tokens=True))

    # response = outputs[:, input_ids.shape[-1]:]
    # print(tokenizer.batch_decode(response, skip_special_tokens=True))
    quit()

def gpt_call(meta_prompt):


    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=meta_prompt)

    response_json = response.to_dict()
    
    return response_json


def generate_examplers(example_tuples_best, example_tuples_worst):
    examplers_best = ''
    examplers_worst = ''
    for i, (response, accuracy) in enumerate(example_tuples_best):
        examplers_best += f'{i+1}: {response} --- Accuracy: {accuracy}\n'

    for i, (response, accuracy) in enumerate(example_tuples_worst):
        examplers_worst += f'{i+1}: {response} --- Accuracy: {accuracy}\n'

    return examplers_best, examplers_worst

def generate_meta_prompt(
        dataset_name, 
        prompt, 
        step, 
        accuracy, 
        dataset_info, 
        global_responses = None, 
        global_accuracies = None,
        top_bottom_k = 5,
        df_logs = None
        ):
    # assert dataset_name == 'EuroSAT' 
    assert len(global_responses) == len(global_accuracies) if global_responses is not None else True

    if global_responses is not None:
        # global responses come in as accending order wrt accuracies 
        # global accuracies are also in accending order associated with each response

        # get the top k responses

        top_bottom_k = min(top_bottom_k, int(len(global_responses)*0.5)) # just a check to make sure we don't go out of bounds


        top_responses = global_responses[-top_bottom_k:]
        top_accuracies = global_accuracies[-top_bottom_k:]

        # get the bottom k responses :top_bottom_k -top_bottom_k:
        bottom_responses = global_responses[:top_bottom_k]
        bottom_accuracies = global_accuracies[:top_bottom_k]

        best_exm, worst_exm = generate_examplers(list(zip(top_responses, top_accuracies)), list(zip(bottom_responses, bottom_accuracies)))


            



    dataset_info = dataset_info[dataset_name]

    system_info = ('You are helpful AI assistant who is going to help me with finding the best prompt templates to embed the class names' 
                   ' for my dataset for zero-shot classification with CLIP. Let\'s Go!')
    
    if step == 0:
        task_info = (f'You are provided with a dataset name, description, a prompt template and the resulting classification accuracy of the model from this prompt.' 
                                    f' Your task is to provide me with 10 new prompt templates in the same format as the given prompt,' 
                                    f' so that I can simply replace the <category> placeholders with the actual class names in' 
                                    f' the dataset and use it for zero-shot classification with CLIP. The goal is to get an increase' 
                                    f' in accuracy by using the newly generated prompts. You can use the dataset'
                                    f' description to provide me with more tailored prompt templates for the dataset. Be creative! Good luck!'
                                    f'\n\nDataset: {dataset_name}\nDescription: {dataset_info}\nPrompt: {prompt}\nAccuracy: {accuracy[0]}')
        
    else: 
        task_info = (f'You are provided with a dataset name, description, top {top_bottom_k} and worst {top_bottom_k}' 
                                    f' example prompt templates with their associated accuracies from the last {step+1} runs.' 
                                    f' Your task is to provide me with 10 new prompt templates in the same format as the given prompts,' 
                                    f' so that I can simply replace the <category> placeholders with the actual class names in' 
                                    f' the dataset and use it for zero-shot classification with CLIP. The goal is to get an increase' 
                                    f' in accuracy by using the newly generated prompts. You can use the dataset'
                                    f' description and the best and worst example prompts as context for improving accuracy. Be creative! Good luck!'
                                    f'\n\nDataset: {dataset_name}\nDescription: {dataset_info}\nBest Templates:\n {best_exm}\nWorst Templates:\n {worst_exm}') 
        
    meta_prompt = [
        {"role": "system", "content": f'{system_info}'},
        {"role": "user", "content": f'{task_info}'}, 
    ]
    return meta_prompt


def generate_meta_prompt_llama(
        dataset_name, 
        prompt, 
        step, 
        accuracy, 
        dataset_info, 
        global_responses = None, 
        global_accuracies = None,
        top_bottom_k = 5,
        df_logs = None,
        mistakes=None
        ):
    # assert dataset_name == 'EuroSAT' 
    assert len(global_responses) == len(global_accuracies) if global_responses is not None else True

    if global_responses is not None:
        # global responses come in as accending order wrt accuracies 
        # global accuracies are also in accending order associated with each response

        # get the top k responses

        top_bottom_k = min(top_bottom_k, int(len(global_responses)*0.5)) # just a check to make sure we don't go out of bounds


        top_responses = global_responses[-top_bottom_k:]
        top_accuracies = global_accuracies[-top_bottom_k:]

        # get the bottom k responses :top_bottom_k -top_bottom_k:
        bottom_responses = global_responses[:top_bottom_k]
        bottom_accuracies = global_accuracies[:top_bottom_k]

        top_accuracies = [int(acc) for acc in top_accuracies]
        bottom_accuracies = [int(acc) for acc in bottom_accuracies]

        # print(bottom_accuracies, top_accuracies)
        # quit()

        best_exm, worst_exm = generate_examplers(list(zip(top_responses, top_accuracies)), list(zip(bottom_responses, bottom_accuracies)))


            



    dataset_info = dataset_info[dataset_name]

    system_info = ('You are helpful AI assistant who is going to help me with finding the best prompt templates to embed the class names' 
                   ' for my dataset for zero-shot classification with CLIP. Let\'s Go!')
    if mistakes is None:
        if step == 0:
            task_info = (f'You are provided with a dataset name, description, a prompt template and the resulting classification accuracy of the model from this prompt.' 
                                        f' Your task is to provide me with a new prompt template in the same format as the given prompt,' 
                                        f' so that I can simply replace the <category> placeholders with the actual class names in' 
                                        f' the dataset and use it for zero-shot classification with CLIP. The goal is to get an increase' 
                                        f' in accuracy by using the newly generated prompts. You can use the dataset'
                                        f' description to provide me with more tailored prompt templates for the dataset. Be creative! Good luck!'
                                        f'\n\nDataset: {dataset_name}\nDescription: {dataset_info}\nPrompt: {prompt}\nAccuracy: {accuracy[0]}\n'
                                        f'# Remember to only provide me the prompt as a response, and nothing else.'
                                        )
            
        else: 
            task_info = (f'You are provided with a dataset name, description, top {top_bottom_k} and worst {top_bottom_k}' 
                                        f' example prompt templates with their associated accuracies from the last {step+1} runs.' 
                                        f' Your task is to provide me with 1 new prompt templates in the same format as the given prompts,' 
                                        f' so that I can simply replace the <category> placeholders with the actual class names in' 
                                        f' the dataset and use it for zero-shot classification with CLIP. The goal is to get an increase' 
                                        f' in accuracy by using the newly generated prompts. You can use the dataset'
                                        f' description and the best and worst example prompts as context for improving accuracy. Be creative! Good luck!'
                                        f'\n\nDataset: {dataset_name}\nDescription: {dataset_info}\nBest Templates:\n {best_exm}\nWorst Templates:\n {worst_exm}'
                                        f'### Remember to only provide me the prompt as a response, and nothing else. ####'
                                        ) 
            
    else:
        if step == 0:
            task_info = (f'You are provided with a dataset name, description, a prompt template and the resulting classification accuracy of the model from this prompt.' 
                                        f' Additionally you are also provided with the names of classes which were not classified correctly in the last run.' 
                                        f' Your task is to provide me with a new prompt template in the same format as the given prompt,' 
                                        f' so that I can simply replace the <category> placeholders with the actual class names in' 
                                        f' the dataset and use it for zero-shot classification with CLIP. The goal is to get an increase' 
                                        f' in accuracy by using the newly generated prompts. You can use the dataset'
                                        f' description to provide me with more tailored prompt templates for the dataset. Additionally you may use the names of the classes' 
                                        f' which were classified wrongly in the last run as additional context so that the next prompts can correct those mistakes. However, '
                                        f' the template also should work for other classes which were classified correctly.'
                                        f'\n\nDataset: {dataset_name}\nDescription: {dataset_info}\nPrompt: {prompt}\nAccuracy: {accuracy[0]}\nClasses which were classified wrongly: {mistakes}\n'
                                        f'### Remember to only provide me the prompt as a response, and nothing else. ####'
                                        )
            
        else: 
            task_info = (f'You are provided with a dataset name, description, top {top_bottom_k} and worst {top_bottom_k}' 
                                        f' example prompt templates with their associated accuracies from the last {step+1} runs.'
                                        f' Additionally you are also provided with the names of classes which were not classified correctly in the last run.' 
                                        f' Your task is to provide me with 1 new prompt templates in the same format as the given prompts,' 
                                        f' so that I can simply replace the <category> placeholders with the actual class names in' 
                                        f' the dataset and use it for zero-shot classification with CLIP. The goal is to get an increase' 
                                        f' in accuracy by using the newly generated prompts. You can use the dataset'
                                        f' description to provide me with more tailored prompt templates for the dataset. Additionally you may use the names of the classes' 
                                        f' which were classified wrongly in the last run as additional context so that the next prompt can correct those mistakes. However, '
                                        f' the templates also should work for other classes which were classified correctly.'
                                        f'\n\nDataset: {dataset_name}\nDescription: {dataset_info}\nBest Templates:\n {best_exm}\nWorst Templates:\n {worst_exm}'
                                        f'### Remember to only provide me the prompt as a response, and nothing else. ####'
                                        )
        
    meta_prompt = [
        {"role": "system", "content": f'{system_info}'},
        {"role": "user", "content": f'{task_info}'}, 
    ]



    return meta_prompt, system_info, task_info


def main(dataset_name, prompt, step, accuracy, dataset_info, 
         response_dict = None, df_logs = None, top_bottom_k = 5, 
         global_responses = None, global_accuracies = None, args = None,
         llm_model=None, writer=None, mistakes=None):

    if 'gpt' in args.llm:

        print('Loading GPT model...')
        meta_prompt = generate_meta_prompt(dataset_name=dataset_name, prompt=prompt, step=step, accuracy=accuracy, dataset_info=dataset_info, 
                                    global_responses=global_responses, global_accuracies=global_accuracies, top_bottom_k=top_bottom_k, 
                                    df_logs=df_logs)

        response = gpt_call(meta_prompt)

    elif 'llama' in args.llm:

        meta_prompt, system_prompt, task_info = generate_meta_prompt_llama(dataset_name=dataset_name, prompt=prompt, step=step, accuracy=accuracy, dataset_info=dataset_info, 
                            global_responses=global_responses, global_accuracies=global_accuracies, top_bottom_k=top_bottom_k, 
                            df_logs=df_logs, mistakes=mistakes)
        if not args.do_proxy_tuning:

            response, logits, final_output_embdeddnings = llm_model.text_generation(meta_prompt) # normal output            
        
        elif args.alpha == 0.0: # basically resorting to random search!
            response, logits = llm_model.manipulate_embeddings(meta_prompt, writer, i=step) # proxy tuning output

        elif 'proxy_tuning' in args.exp_name:
            response, logits = llm_model.proxy_tuning(meta_prompt) # proxy tuning output
        
        else:
            response, logits = llm_model.manipulate_embeddings(meta_prompt, writer, i=step) # proxy tuning output

        return response, meta_prompt, task_info, logits



        # response = response.to_dict()

    else: 

        raise ValueError('Please provide a valid LLM model')

    return response, meta_prompt


