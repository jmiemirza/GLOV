from typing import Optional, Dict, Any
import torch
from transformers_local import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers_local.generation.utils import (
    ModelOutput,
    # top_k_top_p_filtering,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict
from transformers_local import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
# import transformer_lens
from typing import Optional, Union, Tuple, Callable, List, cast
from functools import partial


def generate_noise(num_samples):
    noise = torch.randn(num_samples).cuda()
    # scaled_noise = noise / torch.max(torch.abs(noise))
    return noise


def initialize_dict(out):
    # Initialize the nested dictionary structure properly

    text = ''.join(out)


    text = ''

    for i in out:

        i = i.replace('\n', '')
        i = i.replace('assistant', '')

        text += i + '\n'


    data = {
        'choices': [
            {
                'message': {
                    'content': text
                }
            }
        ]
    }

    return data

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class LlamaPart1(nn.Module):
    def __init__(self, original_model):
        super(LlamaPart1, self).__init__()
        self.embed_tokens = original_model.model.embed_tokens  # Embedding layer
        self.layers = original_model.model.layers  # LLaMA layers (0-31)
    
    def forward(self, input_ids):
        # Embedding
        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Pass through each LLaMA layer
        for layer in self.layers:
            x = layer(x[0], position_ids=position_ids)
            # print(x.shape)

        
        return x
    
class LlamaPart2(nn.Module):
    def __init__(self, original_model):
        super(LlamaPart2, self).__init__()
        self.norm = original_model.model.norm  # Final normalization layer
        self.lm_head = original_model.lm_head  # Language model head (output projection)

    
    def forward(self, x):
        # Apply normalization
        x = self.norm(x)
        x = self.lm_head(x)
        
        return x



import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class LlamaModel:
    def __init__(
        self,
        base_model_name_or_path='/system/user/publicdata/llm/Meta-Llama-3-8B-Instruct/',
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        device_map="auto",
        load_in_8bit=False,
        convert_to_half=False,
        use_fast_tokenizer=True,
        padding_side="left",
        args = None

    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
        # 'output_hidden_states': True,
        }
        self.alpha = alpha
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **model_kwargs)

        self.model_embeddings = LlamaPart1(self.model) # get the embeddings from the model from the 32 layers of llama
        self.model_output = LlamaPart2(self.model) # pass the embeddings through the final layer of llama (the layer norm and the lm_head)

        if convert_to_half:
            self.model = self.model.half()
        self.model.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=use_fast_tokenizer)
        self.tokenizer = self.add_pad_token(tokenizer, padding_side)

        self.worst_prompt = 'a photo of a {}.'

        self.base_logits = self.get_base_embeddings() # for the first stage!!

        # print(self.base_logits.values)
        # quit()

        self.best_prompt = 'a photo of a {}.' # initialize with the prompt 'a photo of a {}'

        self.best_prompt_logits = self.base_logits # initialize with the logits from the prompt 'a photo of a {}'

        self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        
        # initialize the diff vector with a tensor of zeros == the shape of the embeddings1
        # self.diff_vector = torch.zeros_like(self.base_logits) # this is only for EMA!!!
        self.ema_alph = self.args.ema_alpha
        


    @torch.no_grad()
    def get_base_logits(self):
        input_ids = self.tokenizer('a photo of a {}.', return_tensors="pt").to(self.model.device) # returns input ids and attention masks!!!! 
        output = self.model(**input_ids).logits # (1, 6, vocab_size)        
        return output
    
    def get_logits_for_proxy_tuning(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # returns input ids and attention masks!!!! 
        output = self.model(**input_ids).logits # (1, 6, vocab_size)        
        return output

    
    @torch.no_grad()
    def get_base_embeddings(self, prompt='a photo of a {}.'):

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # returns input ids and attention masks!!!! 
        hidden_state = self.model(**input_ids, return_dict=True, output_hidden_states=True)
        hidden_state = hidden_state['hidden_states']
        out_put_emb = dict()

        if self.args.all_layer_diff == 'True':
            final_layer_outputs = [l.mean(dim=1).squeeze() for l in hidden_state][1:]

        elif self.args.specific_layer_diff == 'True' and self.args.mean_emb=='False':
            hidden_state = hidden_state[1:]

            final_layer_outputs = hidden_state[self.args.diff_layer]

        elif self.args.mean_emb=='True' and self.args.specific_layer_diff == 'True':

            for l in self.args.diff_layer:
            
                hidden_state = hidden_state[1:]
                final_layer_outputs = hidden_state[l].mean(dim=1)
                # final_layer_outputs = hidden_state[l][:,-1,:]
                # print(final_layer_outputs.shape)
                # quit()
                # if self.args.normalize == 'True':

                #     final_layer_outputs =  F.normalize(final_layer_outputs, p=2, dim=1)
                    
                out_put_emb[l] = final_layer_outputs

            return out_put_emb

        else:

            final_layer_outputs = hidden_state[-1].squeeze()

        return final_layer_outputs # 4096


    def add_pad_token(self, tokenizer, padding_side="left"):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = padding_side
        return tokenizer
    
    @torch.no_grad
    def text_generation(self, prompt):

        tokenized_prompts = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=False,
            padding="longest", return_tensors="pt", #, add_special_tokens=True
            ).to(self.model.device) # 1 x 241
    
        output_text = self.model.generate(
            tokenized_prompts,
            max_new_tokens=50,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            num_return_sequences=self.args.num_prompt, #take care about this!!!
            return_dict_in_generate = True, 
            output_logits = True,
            output_hidden_states = True,
            )

        out = output_text['sequences'][:, tokenized_prompts.shape[-1]:] # remove the input text from the output
        out = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        data = initialize_dict(out) # to keep consistent with the format output from gpt
        return data, None, None # return the current logits (num_of_tokens, [num_return_sequences, vocab_size]) -- seems like its a tuple   
    
    @torch.no_grad
    def proxy_tuning(self, prompt):



        tokenized_prompts = self.tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=False,
                    padding="longest", return_tensors="pt", #, add_special_tokens=True
                    return_dict=True
                    ).to(self.model.device) # dtype: dict
        
        input_ids = tokenized_prompts['input_ids']
        attention_masks = tokenized_prompts['attention_mask']
        expert_input_ids = input_ids

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)
        


        generated_tokens = list()
        for step in range(50): # where 50 is the max number of tokens to generate
            print(step, self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
            # We should manipulate the logits of the context/prompts of the target model, before generating new tokens from it.
            # NOTE that things are done util time step t during decoding. The context/prompts are also changing.

            antiexpert_next_token_logits = self.model(input_ids).logits  # bs x seqlen x vocab
            # expert_next_token_logits = self.best_prompt_logits[..., -1, :]
            expert_next_token_logits = self.model(input_ids).logits # bs x seqlen x vocab
            target_model_logits = self.model(input_ids).logits

            new_logits = target_model_logits + self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)

            predictions = torch.softmax(new_logits[:, -1, :], dim=1)
            next_token_id = torch.argmax(predictions).unsqueeze(0)
            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)

        batch_outputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        print(batch_outputs)
        return input_ids # still need to decode!!!!

    def prepare_prompts(self, best_prompt: str, worst_prompt: str) -> tuple:
        def tlen(prompt): return len(self.tokenizer(prompt)['input_ids'])
        def pad_right(prompt, length): return prompt + "<|eot_id|>" * (length - tlen(prompt))        
        l = max(tlen(best_prompt), tlen(worst_prompt))
        return pad_right(best_prompt, l), pad_right(worst_prompt, l)

    @torch.no_grad
    def get_last_layer_embeddings(self, model):
        pass

    @torch.no_grad
    def manipulate_embeddings(self, meta_prompt, writer=None, i = None):
        # here use the base prompt logits and the best prompt logits to calculate the new prompt logits

        print('\n\n ----- doing proxy tuning ----- \n\n')
        print('WORST PROMPT: ', self.worst_prompt)
        print('BEST PROMPT: ', self.best_prompt)

        # if self.args.mean_emb == 'False':

        self.best_prompt, self.worst_prompt = self.prepare_prompts(self.best_prompt, self.worst_prompt)


        if isinstance(self.worst_prompt, list):
            base_embeddings = torch.stack([self.get_base_embeddings(p) for p in self.worst_prompt]).mean(dim=0)
            best_prompt_embeddings = torch.stack([self.get_base_embeddings(p) for p in self.best_prompt]).mean(dim=0)
        else:
            base_embeddings = self.get_base_embeddings(self.worst_prompt) # should be a dict
            best_prompt_embeddings = self.get_base_embeddings(self.best_prompt) # should be a dict

        

        tokenized_prompts = self.tokenizer.apply_chat_template(
            meta_prompt,
            add_generation_prompt=False,
            padding="longest", return_tensors="pt", #, add_special_tokens=True
            # return_dict=True
            ).to(self.model.device) # 1 x 241
        
        tokenized_prompts_copy = tokenized_prompts.clone()

        
        # tokenized_prompts_copy = tokenized_prompts.deepcopy()
        noise_std = 0.8

        diff_dict = dict()

        # diff = best_prompt_embeddings - base_embeddings

        norm = 0
        for k, _ in base_embeddings.items():
            diff_dict[k] = best_prompt_embeddings[k] - base_embeddings[k] 
            norm+=diff_dict[k].norm(p=2).item()

        writer.add_scalar('norm_diff', norm, i)




        with torch.no_grad():
            all_text = list()
            for _ in range(self.args.num_prompt):
                generated_tokens = list()
                tokenized_prompts = tokenized_prompts_copy
                # noise = (torch.randn([4096]) * noise_std).cuda()
                noise = generate_noise(4096)
                for time_step in range(75):
                    
                    # Proxy-tuning:
                    emb_base = base_embeddings
                    emb_best = best_prompt_embeddings                    

                    if self.args.cross_attention == 'True' and self.args.all_layer_diff == 'False' and self.args.specific_layer_diff == 'False'and self.args.mean_emb=='False' and self.args.cross_attention_middle_layer == 'False':

                        logits = self.model(tokenized_prompts, cross_attention=True, 
                                            vectors_for_cross_attention=[emb_base, emb_best], 
                                            alpha = self.alpha, writer=writer).logits # bs x seqlen x vocab
                        
                        
                    
                    elif self.args.all_layer_diff == 'False' and self.args.cross_attention == 'False' and self.args.specific_layer_diff == 'False' and self.args.mean_emb=='False' and self.args.cross_attention_middle_layer == 'False':

                        diff = self.diff_vector
                        logits = self.model(tokenized_prompts, 
                                            diff_vectors=diff * self.alpha, time_step=time_step
                                            ).logits # bs x seqlen x vocab

                    elif self.args.all_layer_diff == 'True' and self.args.cross_attention == 'False' and self.args.specific_layer_diff == 'False' and self.args.mean_emb=='False' and self.args.cross_attention_middle_layer == 'False':
                        all_layers_difference = [self.args.alpha*(emb_best[i] - emb_base[i]) for i in range(len(emb_base))]
                        logits = self.model(tokenized_prompts, all_layer_diff=True, all_layers_differences=all_layers_difference).logits # bs x seqlen x vocab


                    elif self.args.specific_layer_diff == 'True' and self.args.cross_attention == 'False' and self.args.all_layer_diff == 'False' and self.args.mean_emb=='False' and self.args.cross_attention_middle_layer == 'False':

                        layer_diff = self.args.alpha * self.diff_vector



                        logits = self.model(tokenized_prompts, specific_layer_diff=True, layer_offset=layer_diff, diff_layer=self.args.diff_layer).logits # bs x seqlen x vocab

                    elif self.args.all_layer_diff == 'False' and self.args.cross_attention == 'False' and self.args.specific_layer_diff == 'True' and self.args.mean_emb=='True' and self.args.cross_attention_middle_layer == 'False':
                        diff_dict = {key: self.args.alpha * value for key, value in diff_dict.items()}
                        logits = self.model(tokenized_prompts, mean_emb_guidance=True, layer_offset=diff_dict, diff_layer=self.args.diff_layer).logits # bs x seqlen x vocab

                    elif self.args.all_layer_diff == 'False' and self.args.cross_attention == 'False' and self.args.specific_layer_diff == 'True' and self.args.mean_emb == 'False' and self.args.cross_attention_middle_layer == 'True':

                        vectors_for_att = [emb_base, emb_best]
                        
                        logits = self.model(tokenized_prompts, cross_attention_middle=True, vectors_for_crs_att_middle=vectors_for_att, 
                                            # layer_offset=layer_diff, 
                                            diff_layer=self.args.diff_layer, alpha=self.args.alpha).logits # bs x seqlen x vocab
                    else:
                        raise ValueError('INVALID MODE FOR PROXY TUNING!')



                    if self.args.do_sample == 'True':
                        logits = logits[0, -1, :] / self.args.temperature
                        filtered_logits = top_k_top_p_filtering(logits, top_k=self.args.top_k, top_p=self.args.top_p)
                        next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
                    else:                    
                        predictions = torch.softmax(logits[:, -1, :], dim=-1)
                        next_token_id = torch.argmax(predictions).unsqueeze(0)

                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break

                    generated_tokens.append(next_token_id.item())
                    
                    tokenized_prompts = torch.cat([tokenized_prompts, next_token_id.unsqueeze(0)], dim=1)

                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_text.append(generated_text)
            data = initialize_dict(all_text) # to keep consistent with the format output from gpt
        return data, None



    @torch.no_grad
    def simple_greedy_decoding(self, meta_prompt, writer=None, i = None):
        print('\n\n simple greedy decoding...')

        tokenized_prompts =self.tokenizer.apply_chat_template(
            meta_prompt,
            add_generation_prompt=False,
            padding="longest", return_tensors="pt", #, add_special_tokens=True
            # return_dict=True
            ).to(self.model.device) # 1 x 241
        
        tokenized_prompts_copy = tokenized_prompts.clone()


        with torch.no_grad():
            all_text = list()
            for _ in range(10):
                generated_tokens = list()
                tokenized_prompts = tokenized_prompts_copy
                noise = generate_noise(4096)
                for _ in range(50):

                    logits = self.model(tokenized_prompts, cross_attention=True, 
                                        simple_greedy_decoding = True,
                                            alpha = self.alpha, writer=writer).logits # bs x seqlen x vocab

                    # if self.args.do_sample == 'True':
                    logits = logits[0, -1, :] / self.args.temperature
                    filtered_logits = top_k_top_p_filtering(logits, top_k=self.args.top_k, top_p=self.args.top_p)
                    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
                    generated_tokens.append(next_token_id.item())
                
                    # Append the new token to the input sequence for the next iteration
                    tokenized_prompts = torch.cat([tokenized_prompts, next_token_id.unsqueeze(0)], dim=1)
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break

                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_text.append(generated_text)
            data = initialize_dict(all_text) # to keep consistent with the format output from gpt
        return data, None
    

    @torch.no_grad
    def proxy_tuning(self, meta_prompt):

        print('\n\n ----- !!!!DOING PROXY TUNING FOR REAL!!! ----- \n\n')
        print('WORST PROMPT: ', self.worst_prompt)
        print('BEST PROMPT: ', self.best_prompt)

        self.best_prompt, self.worst_prompt = self.prepare_prompts(self.best_prompt, self.worst_prompt)

        tokenized_prompts = self.tokenizer.apply_chat_template(
            meta_prompt,
            add_generation_prompt=False,
            padding="longest", return_tensors="pt",
            ).to(self.model.device) # 1 x 241
        
        tokenized_prompts_copy = tokenized_prompts.clone()

        best_prompt_logits = self.get_logits_for_proxy_tuning(prompt=self.best_prompt)
        worst_prompt_logits = self.get_logits_for_proxy_tuning(prompt=self.worst_prompt)


        with torch.no_grad():
            all_text = list()
            for _ in range(10):
                generated_tokens = list()
                tokenized_prompts = tokenized_prompts_copy
                for _ in range(50): # where 50 is the max number of tokens to generate
                    target_model_logits = self.model(tokenized_prompts, 
                                        simple_greedy_decoding = True).logits # bs x seqlen x vocab
                    logits = target_model_logits + self.args.alpha * (best_prompt_logits - worst_prompt_logits)

                    if self.args.do_sample == 'True':
                        logits = logits[0, -1, :] / self.args.temperature
                        filtered_logits = top_k_top_p_filtering(logits, top_k=self.args.top_k, top_p=self.args.top_p)
                        next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
                    else:                    
                        predictions = torch.softmax(logits[:, -1, :], dim=-1)
                        next_token_id = torch.argmax(predictions).unsqueeze(0)

                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break

                    generated_tokens.append(next_token_id.item())
                    tokenized_prompts = torch.cat([tokenized_prompts, next_token_id.unsqueeze(0)], dim=1)
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_text.append(generated_text)
            data = initialize_dict(all_text) # to keep consistent with the format output from gpt
        return data, None



