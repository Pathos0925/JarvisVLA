import os
import time

from rich import print
from openai import OpenAI
import random
from typing import Literal
import copy
from pathlib import Path
from collections import Counter
import numpy as np

from jarvisvla import QWEN_SPECIAL_TOKENS
from jarvisvla.inference import action_mapping, load_model, processor_wrapper
from jarvisvla.utils.file_utils import load_json_file

#################
# prompt
#################

INSTRUCTION_TEMPLATE = [
    "help me to craft a {}.",
    "craft a {}.",
    "Craft a {}.",
    "Could you craft a {} for me?",
    "I need you to craft a {}.",
    "Please craft a {} in the game.",
    "Craft me a {} quickly.",
    "Make sure to craft a {} for the task.",
    "Craft a {} so I can use it.",
    "Let’s craft a {} for this project.",
    "I need you to craft {} right now.",
]

class VLLM_AGENT:
    # Per-action token budget: <|act_start|> + up to 9 button tokens + (0 or 1) inventory
    # + 2 camera bins + <|act_end|>. Typical 6-10, max ~14. 16 gives slack without bloating
    # the worst-case decode budget that hurts p99 latency.
    _PER_ACTION_TOKEN_BUDGET = 16

    def __init__(self,checkpoint_path, base_url, api_key="EMPTY",
                 history_num=0,action_chunk_len=1, bpe=0,
                 instruction_type:Literal['simple','recipe','normal'] = 'normal',
                 temperature=0.1):
        
        self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(checkpoint_path=checkpoint_path)

        # Load tokenizer first so we can resolve action-token IDs against it.
        self.tokenizer = None
        from transformers import AutoTokenizer
        if self.LLM_backbone in {"llama-3","llama-2","qwen2_vl","qwen3_5"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
            # Older Qwen2-VL checkpoints may have been saved before special tokens were
            # registered; re-add idempotently. add_special_tokens returns 0 if everything is
            # already present. For Qwen3.5 we expect the saved tokenizer to be complete.
            if self.LLM_backbone == "qwen2_vl":
                import json
                with open(QWEN_SPECIAL_TOKENS, "r") as file:
                    special_token = json.load(file)
                n_added = self.tokenizer.add_special_tokens({"additional_special_tokens": special_token})
                if n_added:
                    print(f"[yellow]agent_wrapper: re-added {n_added} special tokens missing from checkpoint")

        # Action tokenizer now resolves token IDs against the real tokenizer instead of
        # using hard-coded ID tables; this catches vocab drift loudly at startup.
        if self.tokenizer is None:
            raise RuntimeError(
                f"backbone {self.LLM_backbone!r} requires a tokenizer to build the action map"
            )
        # ACTION_SCHEMA env override: training data is Qwen2-VL-encoded
        # (<|reserved_special_token_178|>...<|reserved_special_token_179|>),
        # so models trained with --backbone=qwen3_5 still emit qwen2_vl action
        # tokens. Set ACTION_SCHEMA=qwen2_vl to parse with that schema instead
        # of the LLM backbone. See README "Debugging history" section.
        action_schema_override = os.environ.get("ACTION_SCHEMA")
        schema_backbone = action_schema_override or self.LLM_backbone
        if action_schema_override and action_schema_override != self.LLM_backbone:
            print(f"[agent_wrapper] action_schema override: parsing with {action_schema_override!r} "
                  f"(model backbone is {self.LLM_backbone!r})", flush=True)
        self.action_tokenizer = action_mapping.OneActionTokenizer.from_tokenizer(
            backbone=schema_backbone, tokenizer=self.tokenizer,
        )

        self.prompt_library = load_json_file(Path(__file__).parent/"assets"/"instructions.json") #存储我写好的instructions
        self.recipe_fold=Path(__file__).parent/"assets"/"recipes" # 存储所有recipes的文件夹
        self.recipes = dict()  #制作方案集合
        self.method_map = {
            True: "crafting table",
            False: "inventory",
        }

        self.processor_wrapper = processor_wrapper.ProcessorWrapper(None,model_name=self.VLM_backbone)

        self.actions = []
        self.action_chunk_len=action_chunk_len  # 一次返回一个action chunk
        # 用于带有记忆的agent
        self.history_num = history_num
        self.history = []
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        models = self.client.models.list()
        self.model = models.data[0].id

        self.temperature = temperature
        self.instruction_type = instruction_type
    
    def reset(self):
        self.history = []
          
    def rule_based_instruction(self,env_prompt:str):
        item_name = env_prompt[11:].replace("_"," ")
        instruction_template = random.choice(INSTRUCTION_TEMPLATE)
        return instruction_template.format(item_name)
          
    def create_basic_instruction(self,env_prompt:str):
        instruction = "."
        instructions = self.prompt_library.get(env_prompt,{}).get("instruct")
        if instructions:
            instruction = np.random.choice(instructions)
        else:
            instruction = self.rule_based_instruction(env_prompt)

        if instruction.strip()[-1] != '.':
            instruction += ". \n"
        instruction += "\n"
        return instruction
        
    def get_recipe_item_name(self,ingredient:dict):
        item_name = ingredient.get("item") 
        if not item_name:
            item_name = ingredient.get("tag") 
        return item_name
        
    def create_recipe_prompt_from_library(self,item_name:str):
        if item_name in self.recipes:
            return self.recipes[item_name]
        recipe_path = self.recipe_fold/f"{item_name}.json"
        print(recipe_path)
        if not recipe_path.exists():
            self.recipes[item_name]= ""
            return ""
        recipe_file = load_json_file(recipe_path)
        recipe_type = recipe_file.get("type",None)
        
        prompt = ""
        if not recipe_type:
            self.recipes[item_name]= ""
            return ""
        elif recipe_type=="minecraft:crafting_shapeless":
            prompt+=f"\nYou will need the following ingredients: \n"
            ingredients = recipe_file.get("ingredients",None)
            ingredients_list = []
            for ingredient in ingredients:
                ingredient_name = self.get_recipe_item_name(ingredient)
                if not ingredient_name:
                    break
                ingredients_list.append(ingredient_name[10:].replace("_"," "))
            ingredients_dict = Counter(ingredients_list)
            for item,number in ingredients_dict.items():
                prompt += f"{number} {item}, "
            prompt += "\n"
        elif recipe_type == "minecraft:crafting_shaped":
            prompt+="\nArrange the materials in the crafting grid according to the following pattern: \n"
            patterns = recipe_file.get("pattern",None)
            if not patterns:
                return ""
            ingredients = recipe_file.get("key",{})
            ingredients_dict = {}
            for ingredient_mark,value in ingredients.items():
                ingredient_name = self.get_recipe_item_name(value)
                ingredients_dict[ingredient_mark] = ingredient_name[10:]
            prompt+="\n"
            for pattern_line in patterns:
                for pattern_mark in pattern_line:
                    ingredients_name = ""
                    if pattern_mark==" ":
                        ingredients_name = "air"
                    else:
                        ingredients_name = ingredients_dict.get(pattern_mark,"air")
                    prompt += f" {ingredients_name} |"
                if prompt[-1]=='|':
                    prompt = prompt[:-1] 
                    prompt += "\n"
            prompt +="\n"
        else:
            self.recipes[item_name]= ""
            return ""
        result_num = recipe_file.get("result",{}).get("count",1)
        prompt += f"and get {result_num} {item_name.replace('_',' ')}. \n"
        self.recipes[item_name]= prompt
        return prompt
        
    def create_recipe_prompt(self,env_prompt:str,method:str="crafting_table"):
        """从原始的一句话转换成prompt """
        prompt = ""
        # 如果使用recipe book to craft
        if "recipe_book" in method:
            prompt += "\nUse the recipe book to craft. \n"
            return prompt
        # else
        recipe = self.prompt_library.get(env_prompt,{}).get("recipe")
        if recipe:
            prompt += "\nArrange the materials in the crafting grid according to the following pattern: \n"
            prompt += recipe[0]
        else:
            item_name = env_prompt.replace(" ","_").split(":")[-1]
            prompt += self.create_recipe_prompt_from_library(item_name,)
        return prompt
        
    def create_instruction(self,env_prompt,method):
        prompt =None
        if self.instruction_type == 'recipe':
            prompt = self.create_basic_instruction(env_prompt)
            recipe_prompt = self.create_recipe_prompt(env_prompt,method=method)

            prompt += recipe_prompt
        elif self.instruction_type == 'simple':
            prompt = self.create_thought(env_prompt) 
        elif self.instruction_type == 'normal':
            natural_text = env_prompt.replace("_"," ").replace(":"," ")
            prompt = random.choice(self.prompt_library.get(env_prompt,{"instruct":[natural_text]})["instruct"])
        else:
            raise ValueError(f"do not set the instruction class {self.instruction_type}")
        return prompt
        
    def create_thought(self,env_prompt):
        thought = copy.copy(self.prompt_library.get(env_prompt,{}).get("thought"))
        if not thought:
            thought = env_prompt.replace("item",str(1)).replace("_"," ").replace(":"," ")   #craft item xxx =》 craft 1 item
        thought += '. \n'
        return thought
        
    def forward(self,observations,instructions,verbos=False,need_crafting_table=False):
        if self.actions:
            if verbos:
                print(self.actions)
            if len(self.actions)>1:
                return self.actions.pop(0)
            else:
                action = self.actions[0]
                self.actions = []
                return action
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0]) 
        method = self.method_map[need_crafting_table]
        prompts = []
        private_instruction = self.create_instruction(instructions[0],method=method)
        #print(private_instruction)
        thought= self.create_thought(instructions[0]) if self.instruction_type =="recipe" else ""

        if self.history_num:
            if not self.history: #如果历史为空
                self.history = [(image,self.action_tokenizer.null_token(),copy.copy(thought),0)]*self.history_num
            new_history = [None]*self.history_num
            new_history[:-1] = self.history[1:]
            for hdx,(im, ac, past_thought,_) in enumerate(self.history):
                prompt_input = ""
                if self.instruction_type == 'recipe':
                    prompt_input = "\nthought: " + past_thought + "\nobservation: "  #往上一个prompt上加上这一步的thought
                else:
                    prompt_input = "\nobservation: "
                if not hdx: #hdx==0
                    prompt_input = private_instruction + prompt_input
                #print(ac,prompt_input,)
                messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[im]))
                messages.append(self.processor_wrapper.create_message_vllm(role="assistant",input_type="text",prompt=[ac],))
            
        prompt_input = ""
        if self.instruction_type == 'recipe':
            prompt_input = "\nthought: " + thought + "\nobservation: "
        else:
            prompt_input = "\nobservation: "
        if not self.history_num:
            prompt_input = private_instruction + prompt_input
        #print(prompt_input)

        messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[image]))
        
        open_logprobs = False
        if verbos:
            print(prompts)
            open_logprobs = True
        
        # Cap per-request token budget at chunk_len × per-action budget. Old code used
        # max_tokens=1024 — if the model hallucinated long output it would burn through
        # the entire budget before we got an action, ruining p99 latency.
        max_tokens = self.action_chunk_len * self._PER_ACTION_TOKEN_BUDGET
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=0.99,
            logprobs=open_logprobs,
            extra_body={"skip_special_tokens": False, "top_k": -1},
        )

        outputs = chat_completion.choices[0].message.content
        if self.history_num:
            new_history[-1] = (image,outputs,thought,self.history[-1][-1]+1)
            self.history = new_history
        # Re-tokenize the string output to IDs whenever we have a tokenizer. add_special_tokens=False
        # avoids accidental BOS insertion that would offset the action-segment indices. The fragility
        # the reviewers flagged for string-roundtrip is mostly about non-special tokens — action tokens
        # are registered as atomic specials so they round-trip reliably.
        if self.tokenizer is not None:
            outputs = self.tokenizer(outputs, add_special_tokens=False)["input_ids"]

        actions = self.action_tokenizer.decode(outputs)

        # Sanity log: count tokens inside act_start/act_end segments that didn't map to a known
        # action. Loud >5% rate is a sign of tokenizer drift or output garbage.
        if isinstance(outputs, list):
            self._log_decode_health(outputs, len(actions))

        len_action = min(self.action_chunk_len, len(actions))
        if len_action < self.action_chunk_len:
            # Fewer actions than the chunk size — early EOS or model misbehavior. Log so we
            # can spot it instead of silently under-acting.
            print(f"[yellow]agent_wrapper: emitted {len(actions)} actions, expected {self.action_chunk_len}")
        self.actions = actions[:len_action]

        return self.actions.pop(0)

    def _log_decode_health(self, token_ids: list[int], n_decoded_actions: int) -> None:
        """Best-effort: count action-segment tokens that aren't in the action map.

        Useful as an early warning when the SFT'd model starts emitting garbage tokens
        between <|act_start|> and <|act_end|>.
        """
        in_segment = False
        unknown = total = 0
        token_to_action = self.action_tokenizer.maps.token_to_action
        beg = self.action_tokenizer.act_beg_id
        end = self.action_tokenizer.act_end_id
        for tid in token_ids:
            if tid == beg:
                in_segment = True
                continue
            if tid == end:
                in_segment = False
                continue
            if in_segment:
                total += 1
                if tid not in token_to_action:
                    unknown += 1
        if total > 0 and unknown / total > 0.05:
            print(f"[yellow]agent_wrapper: {unknown}/{total} action-segment tokens unrecognized "
                  f"({100*unknown/total:.1f}%) across {n_decoded_actions} actions")
