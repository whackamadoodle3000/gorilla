# Instructions

- We are running a study to benchmark tool use with Deepseek v3 model, and an augmented version of the model with the “ACE” framework to tool calling. The augmented version is exactly the same as the original version except it includes extra information in its prompt that comes from a “playbook” database.  
- First make sure you have a thorough understanding how how this codebase works, and the whole pipelines of calls in order to test a model’s function calling. If anything is unclear, ask before proceeding  
- Tell us where we need deepseek api keys  
- You will work off this repo which was originally made to run tool call benchmarks, but you will modify it to support the following changes so that we can test the “ACE” framework on tool calling.   
- I need you to set up an alternate pipeline in order to “train” this model and then test this model.   
  - First you will need to split the dataset. It will need to be split into a test and train set, where each has some of all types of tool calls. The training set will be used to create the playbook database  
  - The training component has three parts, and each part will run sequentially on each piece of training data to build up the playbook. The training component will not use the playbook, it will just prompt the model normally with the prompt from the benchmark dataset “generator”. Next there is a reflector, which will take the ground truth from dataset, along with the prompt and output of the model, and then generate some text of insights from what the model did vs what it should have done. Finally, the curator will take all of these together and then either add, modify, or remove bullet points which are linked to a particular tool. The curator will store all of these elements in a “playbook”. The curator will simply output structured text which will be converted into an operation on the playbook (add, remove a specific bullet point id, modify). The model used for these training components should be changeable  
  - Each process (Generator, Reflector, Curator) acts on the entire “training” data. First the Generator is ran, the Reflector is ran with the Generator’s output, and the Curator is ran with the output of both Generator and Reflector  
  - Maintain the playbook such that it contains “insights” per each tool group (tool groups found in data/multi\_turn\_func\_doc/)  
  - The playbook will be stored as a json somewhere in the data folder and specify where and must be prepended to the message in entirety in the generator phase and must be updated in the reflector page  
  - The new updated pipeline with ACE should be able to be triggered through a CLI call which will update the \_\_main\_\_ function as well as downstream handlers to have a new CLI arg, if this cli arg is not passed in, the model should just work as default from this repository  
- When testing the new model, it will be the original deepseek model \+ a prompt with the bullet points of things it learned from the playbook  
- Then show and save full results for the original model and new model.

Playbook format (per tool group):  
\[  
{Name: “math\_api”, entries: {“delta1””:\<curator generated deltas\>, “delta2”:\<curator generated deltas\>...},  
{Name: GENERIC TOOL GROUP from data/multi\_turn\_func\_doc/, entries:  {“delta1””:\<curator generated deltas\>, “delta2”:\<curator generated deltas\>...},  
….  
\]

# Dataset

- Split into test train   
- Seed random splits  
- Make sure each tool call is reflected in train and test

# Generator: this is what runs the benchmark: 

Input:

- User prompt  
- Playbook (if running for inference we prompt with the ENTIRE playbook, during initial training there is no playbook)

Output:

- Text and tool call

# Reflector:

Input: ground truth, playbook, example (of expected behavior and for format of output) 

Output: reflection on the example  
Format (from ace paper):  
{{  
“reasoning”: “\[Your chain of thought / reasoning / thinking process, detailed analysis and calculations\]”,  
“error\_identification”: “\[What specifically went wrong in the reasoning?\]”,  
“root\_cause\_analysis”: “\[Why did this error occur? What concept was misunderstood?\]”,  
“correct\_approach”: “\[What should the model have done instead?\]”,  
“key\_insight”: “\[What strategy, formula, or principle should be remembered to avoid this error?\]”,  
}}

Tasks:

- Take result of generator’s attempted tool call and results which is the output of berkeley benchmark tool call  
- Pass reflector and generator into LLM and generate natural language reflection based on what’s already in playbook and what is missing using above prompt and pass into curator model

# Curator: 

Input: playbook, generator output, reflector output, user prompt, Ground truth (which include the tool that was used that we are taking notes for)  
Output: any database operations on the playbook to add / modify bullet points  
Output format examples:  
{ "reasoning": "\[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here\]", "operations": \[ {{ "type": "ADD", "section": "math\_api", "content": "\[New calculation method...\]" }} \] }

{ "reasoning": "\[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here\]", "operations": \[ {{ "type": "MODIFY", "section": "math\_api", “ID”:”delta1”, "content": "\[New calculation method...\]" }} \] }

{ "reasoning": "\[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here\]", "operations": \[ {{ "type": "REMOVE", "section": "math\_api", “ID”:”delta1”}} \] }