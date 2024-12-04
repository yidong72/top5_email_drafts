
# Mission: Develop Advanced Language Models with Robust Reasoning Capabilities

- Promising Results From Our Prototype NVIDIA O1-like reasonining model (Gray-Kangaroo).
    - The model is able to generate long chain of thought to breakdown the problems, better use its knowledge, refect and error correct previous reasoning steps and propse alternative solutions.
    - The initial SFT reasoning model improves the Math and Reasoning categories compares with the Nemotron 70B model

|                | mmlu_stem | amc23 | qwen-math | gsm8k | aqua  | sat_math | aime24 |
|----------------|-----------|-------|-----------|-------|-------|----------|--------|
| Gray-Kangaroo  | 42.6*     | 52.5* | 75.2*     | 88.2  | 48    | 56.2*    | 23.3*  |
| Nemotron 70B   | 33.8      | 50    | 70.4      | 94.4* | 55.1* | 43.8     | 16.7   |

|                | bridge  | game24 | reasoning-math | remainder |
|----------------|---------|--------|-----------------|-----------|
| Gray-Kangaroo  | 30.71*  | 37.14* | 87.14*          | 5*        |
| Nemotron 70B   | 15.71   | 27.14  | 82.86           | 2.86      |

- Community Development of O1-like Models
    - Several organizations have developed reasoning-focused models that aim to rival OpenAI's O1 capabilities:
    - DeepSeek R1
        - Achieves competitive performance with O1-preview on AIME and MATH benchmarks
        - Will be open-sourced and offer API access, though licensing terms are not yet announced
    - Marco-O1 by Alibaba
        - Incorporates Chain-of-Thought (CoT) fine-tuning and Monte Carlo Tree Search (MCTS) for complex reasoning
        - Excels at strategic decision-making tasks involving uncertainty and abstract reasoning
    - QwQ-32B-Preview by Alibaba
        - Outperforms O1-preview and O1-mini on AIME and MATH benchmarks
        - Available under Apache 2.0 license for commercial applications, though only certain components are released
- More Details on Nvidia O1-like Reasoning Model
  - Our apporach:
    - Prepare long chain of thought data that mimics the O1 internal thinking process
    - Train the initial reasoning SFT model using the prepared data
    - Run MCTS using the initial reasoning SFT model to create long chain of thought data for hard prompts
    - Combine the long chain of thought data to train the final reasoning SFT data
    - Use RL to further improve the reasoning accuracy for the final reasoning SFT model
  - Reasoning benchmark
    - Carefully selected a few tasks that requires deep thinking to solve it.
    - Currently it includes tasks like game24, arc-agi, mini-crossword, remainder problem, parentheses problem and minimum bridge crossing problem. 
    - David integrates the Qwen math eval to evaluate the math performances for reasoning models. 
  - A joint System1 and System2 model 
    - AI assisstant should operate in two modes, the typical instruction following modes(system 1) that provides quick responses and system2 model that think carefully before giving the final answer.
    - The model behavior can be controlled by using different system prompt and our initial prototype shows the model can switch between modes smoothly. The system2 modes takes more time to generate answer but has better accuracy. For example, in game24 benchmark, system2 mode has accuracy of 29% in contrast to 21% with system 1 mode. 
  - Experimented with two types of models using different reansoing chain data
     - ARB-based reasoning chain model
        - Using ARB as a control signal to guide the detailed reasoning step
        - Run MCTS with different ARBs as different actions, generated long chain of thought data on game24, MATH and onmi math prompts 
     - Natural monologue reasoing chain model
        - Using few shots learning to prompt the Llama3.1 70b model to generate O1-like monologue reasoning chain.
        - Run MCTS using top-k reasoning steps as different actions.
     - Our initial results shows natural monologue reasoing chain model outperform the ARB-based reasoing model (Gray-Kangaroo is an natural monologue reasoning model). We hypotheses that natural reasoning steps has less constraits and the model can think more flexiblly. 
  - Our initial ablation study shows openO1-SFT data explains a lot of good model performance. For example, mmlu_stem benchmark is dropped from 33 to 36.6
  - We are experimenting with curriculum leanring to boost the Gray-Kangaroo model perfomrance using RL on a set of math prompts sorted by difficulties. 
  - Next steps:
    - Include more long chain of thought SFT data to train next version of reasoing model
    - Run MCTS to create better synthetic chain of thought data for solving hard prompts
    - Train better model data using the combined SFT data.
    - Keep exploring using RL to improve the reasoning performance.

### Experime

### Development Status
- Introduced System Two models for novel problem-solving through step-by-step reasoning
- Successfully implemented MCTS for Game 24 benchmark with good reasoning chains
- Achieved 45% accuracy in Amy task reasoning chains
- Identified challenges in spatial reasoning tasks (mini crossword, ARCAGI)

### Model Performance Results

| Model | Game 24 | AME Benchmark | Math Tasks |
|-------|----------|---------------|------------|
| Deep Seek R1 | Competitive | ~30% accuracy | Strong performance |
| Marko 01 | Notable | Under evaluation | Uses MCTS for reasoning |
| System 2 Mode | Outperforms System 1 | - | Improved reasoning |

## Technical Implementations

### Special Tokens Integration
- Addressing token overlapping challenges in SFT process
- Experimenting with different initialization methods
- Implementing transition words for improved reasoning chains
- Testing various initialization methods with larger standard deviation

### MCTS Implementation Results
- 97.4% success rate on math datasets
- 48% success rate on challenging OMI dataset
- Implemented filtering system for incorrect guesses
- Enhanced confidence score calculation for search process

## Value Model Development

### Training Progress
- Trained value model using combined datasets: 2K learning anteater, 13K prompts, and 7K health stair prompts
- Achieved 2% improvement in reward metrics (3.2 to 3.3)
- Planning to exclude 100 responses data in next training round for potential bigger improvements
- Adjusted hyperparameters: low P sampling probability and beta temperature for candidate tokens


## Next Steps

- Train dedicated System Two model using MCTS-generated data
- Apply reinforcement learning to enhance problem-solving capabilities
- Generate and verify large-scale reasoning chain datasets
- Continue SFT process focusing on special tokens optimization
- Implement time-based and token-based improvements for inference




Aug 8

Here's a revised version of the email with improved English and grammar:

Mission: Develop advanced models that align with human values

### System 2 Thinking

* Presented the System 2 project at the LLM F2F meeting, summarizing recent progress. View the [Slide](https://nvidia-my.sharepoint.com/:p:/r/personal/yidong_nvidia_com/_layouts/15/doc2.aspx?sourcedoc=%7BCD9464FC-E04F-4938-B6A8-DF301F7D6023%7D&file=LLM_F2F.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1) and [Recording](https://nvidia-my.sharepoint.com/personal/bginsburg_nvidia_com/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbginsburg%5Fnvidia%5Fcom%2FDocuments%2FRecordings%2FLLM%2DF2F%20%2D%203%2D20240711%5F130212%2DMeeting%20Recording%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E379ce094%2Dde05%2D422d%2D9e75%2De726fbb7baaf).

* We ran tree-logics MCTS on GSM8k datasets, achieving good accuracy and computational performance compared to the Alpha-zero approach. However, it has limitations:
  * It underestimates value, preventing further exploration of branches.
  * Using multiple nucleus samplings for estimation might work but is costly. Verification needed.
  * For finding good solutions, it's slower than random temperature sampling, as it needs to try different actions to decide which to exploit. Given limited resources and a large search space, MCTS isn't the most efficient solution-finding method.

* The team's current focus is pretraining [LM Companion](https://docs.google.com/document/d/1dqHAcs_-csdml1RrGyj6dwDhp1YNq_xf5qAC3cN_c0A/edit#heading=h.hykhwp64jt1s), a foundation value model designed to enhance language model outputs. LM Companion works alongside language models, focusing on evaluating partial generation goodness while the language model focuses on language modeling.

  * Key advantages of LM Companion:
    * Independent from the base language model, allowing flexible pairing with various instruction-following models
    * Provides a foundation for advanced search techniques like MCTS and addresses limitations of Alpha-zero and tree-logics methods
    * Applicable across a wide range of language tasks

  * Integration with Existing Models:
    * Works alongside Supervised Fine-Tuned (SFT) language models
    * SFT model generates candidate tokens, and LM Companion evaluates and ranks these candidates, achieving high-reward responses in one shot. Performance can be further improved using MCTS.

  * Rationale for our value model training approach: Despite a high percentage of nodes with a branching factor of 1, our approach offers significant improvements over RLHF by capturing more structural information, propagating values through the tree, and providing better intermediate value estimation with lower variance. A [recent paper](https://arxiv.org/pdf/2403.03950) shows that training a value model with chess game engine-annotated data can reach Alpha-zero performance without search, validating our approach.

  * Training data preparation:
    * Use Gemma2-it-27b as an actor to generate responses. Annotate responses with a newly trained llama3.1 70b reward model by @zhilin, surpassing the 340b reward model in reward bench.
    * Two approaches to generate synthetic data for training the value model:
       * Run nucleus sampling to generate multiple samples per prompt. Construct the token-tree post-response generation.
         * Average response generation time: 0.489559s
       * Tree-based text generation
         * Grow the tree in a depth-first branch manner
         * Use a customized function to define the token unit forming a node
         * Use dynamic batching to improve inference throughput
         * Average response generation time: 0.44286856s

* Training data analysis (Gerald and Yian):
    * Token-tree constructed by nucleus sampled data:
        * 98.44% of total tokens have branching_factor == 1, as expected
        * Approximately 17k responses needed to estimate intermediate value within 5% error
        * Distribution of depth for nodes with branching_factor > 1 shows concentration in the 10-40 token range, indicating natural branching points
        * Comparison of GSM8K responses from nucleus sampled data and tree-based text generation shows low diversity in nucleus sampled data, possibly due to Gemma2-it-27b being trained on GSM8K data. This is less problematic in tree-based text generation.

* Mentoring @Jeet's internship research:
    * Plan to research optimal critical nodes for LM branching and exploration of possibilities, aiming to balance compute and achieve a good value model
    * Study benchmarks to quantify the goodness of critical nodes

* Next steps (Yian and Gerald):
    * Reuse Math data generated from previous MCTS search, which has good diversity as the model wasn't trained on GSM8k
    * Start training a value model with a small dataset to ensure pipeline functionality
    * Implement evaluation benchmarks for the value model, including in-distribution and out-of-distribution evaluations


Mission: Develop advanced models that are in sync with human values
### System 2 Thinking
* Presented the system 2 project in the LLM F2F meeting, which summarize the recent progress in the system2 project. Check out the [Slide](https://nvidia-my.sharepoint.com/:p:/r/personal/yidong_nvidia_com/_layouts/15/doc2.aspx?sourcedoc=%7BCD9464FC-E04F-4938-B6A8-DF301F7D6023%7D&file=LLM_F2F.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1) and [Recording](https://nvidia-my.sharepoint.com/personal/bginsburg_nvidia_com/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbginsburg%5Fnvidia%5Fcom%2FDocuments%2FRecordings%2FLLM%2DF2F%20%2D%203%2D20240711%5F130212%2DMeeting%20Recording%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E379ce094%2Dde05%2D422d%2D9e75%2De726fbb7baaf)
* We have tried to run tree-logics MCTS on GSM8k datasets, and we have achieved good accuracy and computation performace compared with Alpha-zero approrach. However, it has the following limitations:
  * It underestimates the value and prevents further exploration of that branch​.
  * Using multiple nucleus samplings to estimate might work but is also costly. We need to verify this. ​
  * If the goal is to find a good solution, it is slower than random temperatured sampling. It needs to try different actions to decide which one to exploit. ​ Given limited resources and large search space, MCTS is not the most efficient way to find a good solution. 
* Team's current focus is to pretrained [LM Companion](https://docs.google.com/document/d/1dqHAcs_-csdml1RrGyj6dwDhp1YNq_xf5qAC3cN_c0A/edit#heading=h.hykhwp64jt1s), a foudnation value model that is designed to enhance language model outputs. LM Companion works together with Lanauge model side by side. one is focused on evaluate the goodness of the partial generaiton. and other one is focused on modeling lanaguage.
  * Key advantages of LM Companion:
    * Independent from the base language model, flexible pairing with various instruction-following models
    * Provides foundation for advanced search techniques like MCTS and address the limits with Alpha-zero and tree-logics methods
    * Applicable across a wide range of language tasks.
  * Integration with Existing Models:
    * Works alongside Supervised Fine-Tuned (SFT) language models
    * SFT model generates candidate tokens and LM Companion evaluates and ranks these candidates. Get high reward response in one shot. The perofmrance can be further improved by using MCTS.
  * Rational of our value model training approach. Despite having a high percentage of nodes with a branching factor of 1, our value model training apporach offers significant improvements over RLHF by capturing more structural information, propagating values through the tree, and providing better intermediate value estimation with lower variance. In [paper](https://arxiv.org/pdf/2403.03950), it shows training a value model with data annotated by chess game engine can rearch alpha-zero performance without search. This validates our valude model training approach.
  * Training data preparation.
    * Use Gemma2-it-27b as actor to generate responses. Annotate the responses with a newly trained llama3.1 70b reward model by @zhilin, which surpasses the 340b reward model in reward bench. 
    * Two approaches to generate synthetic data for training value model. 
       * Run neucleus sampling to generate lots of samples per prompt. Construct the token-tree post the response generation.
         * It takes 0.489559s to generate a response on average
       * Tree-based text generation
         * Grow the tree in depth-first branch manner
         *  Use a customized function to define the unit of tokens that forms a node.
         * It is using dynamic batching to improve the inference throughput.
         * It takes 0.44286856s to generate a response on average
* Training data analysis (Gerald and Yian).
    * Token-tree constructe by neucleus sampled data.
        * 98.44% of the total tokens have branching_factor == 1, which is expected since majority the tokens should have bf=1.
        * Number of repsonses we need to estimate the intermediate value is within 5% error is around 17k. That's the minimum responses we need to have a good value estimate. 
        * In distribution of depth of nodes with branching_factor > 1 graph, we can see the mass is concentrated in range 10-40 tokens. This indicates 10-40 token range is a natural place where branch happens. 
        * In a comparison of GSM8K responses from neucleus sampled data nd tree-based text generation method. It is shown that the diversity is very low in neucleus sampled data which might be caused by Gemma2-it-27b is trained on the GSM8K data. It is a lesser problem in tree-based text generation method.
* Mentoring @Jeet's internship research
    * Plan to research on what is a good critical node that the LM should spend time to branch on it and explore different possibitiles. The goal is to find a balance in compute and achieve good value model.
    * Study benchmarks to quantify the goodness of critical node.
* Next step (Yian and Gerald):
    * Resuse the Math data generated from previous MCTS search which has good diversity because the model is not trained on the GSM8k 
    * Start to train a value model with a small dataset to make sure the pipeline works
    * Implements evaluation benchmarks for value model including in-distribution, out-of-distribution evaluations.  



TreeLogic NeMo-Inference-Microservice (NIM):
Develop NIM to enable aligned models to perform System 2 thinking during inference.
Solve complex problems requiring deep searches at inference time, such as math, coding, and logic problems, paired with a verifier to validate results.
Generate higher-quality responses coupled with a reward model.
Produce high-quality positive and negative samples to refine model policies more efficiently than traditional synthetic data generation methods like random sampling and filtering.
Adjustments to Alpha-Zero Approach:
Instead of using LLM output logits for value approximation, estimate state value by greedily sampling tokens using TRTLLM and using reward from the feedback function for estimation. This ensures accurate value estimation during the initial tree search.
Implement a cache function to speed up value function evaluation for the same reasoning path.
Implement early stopping to halt tree search immediately upon finding a satisfactory solution.
Results on GSM8K Math Problem:
Implemented and tested this approach on the GSM8K Math problem, achieving orders of magnitude speedup.
Average search time per sample reduced dramatically from 2400s to 36s.
Out of 7473 GSM8K math problems, the search correctly solved 7452 with a success rate of 99.7%.
Next Steps:
Explore using a reward model as a generic feedback function, enabling the system to search for high-reward responses defined by the user.
Test two generalization hypotheses regarding the use of MCTS to enhance LM math-solving capabilities:
Exposing LLM to more responses per prompt improves math test accuracy.
Using different prompts for different policy improvement iterations enhances math test accuracy.
Model Alignment:
Collaborate on the HelpSteer 2 paper, introducing the SteerLM 2.0 method to address deficiencies in the original SteerLM.
SteerLM 2.0 employs iterative KL minimization between optimal and current policies to better align generated responses with desired attributes.
Results include applying SteerLM 2.0 to the Llama 3 70B model with just a fraction of the training data compared to Llama 3 70B Instruct, achieving an MT-Bench score of 8.28, surpassing both Llama 3 70B Instruct (8.16) and GPT-4-0613 (8.12).


Mission: Develop advanced models that are in sync with human values

 

System 2 Thinking
System 2 research is gaining popularity in the AI community, with recent emergence of several research papers exploring different MCTS techniques for solving math problems. After reviewing our approach, we have identified a few key areas of focus:
We plan to develop TreeLogic NeMo-Inference-Microservice (NIM) to enable aligned models to perform System 2 thinking during inference. This can:
Solve complex problems requiring deep searches at inference time, such as math, coding, and logic problems, paired with a verifier to validate results.
Generate higher-quality responses coupled with a reward model.
Produce high-quality positive and negative samples to refine model policies more efficiently than traditional synthetic data generation method i.e. random sampling and filtering.
We found that the alpha-zero approach, effective for complex gaming problems needing constant policy adjustment based on opponent moves, is excessive for NLP tasks. To enable TreeLogic NIM, we made several adjustments:
Instead of using LLM output logits for value approximation, we estimated state value by greedily sampling tokens using TRTLLM and using reward from the feedback function for estimation. This ensures accurate value estimation during the initial tree search.
Implemented a cache function to speed up value function evaluation for the same reasoning path.
Implemented early stopping to halt tree search immediately upon finding a satisfactory solution.
We implemented and tested this approach on the GSM8K Math problem, achieving orders of magnitude speedup. Average search time per sample reduced dramatically from 2400s to 36s. Out of 7473 GSM8K math problems, the search correctly solved 7452 with a success rate of 99.7%.
Our next step is to explore using a reward model as a generic feedback function, enabling the system to search for high-reward responses defined by the user. (@Wei Du @Zenodia Charpy)
We are testing two generalization hypotheses regarding the use of MCTS to enhance LM math-solving capabilities:
Exposing LLM to more responses per prompt improves math test accuracy. We tested with 4 responses per prompt and used tree search results to enhance policies using SFT, SFT+DPO, and Hybrid models. Results indicate increased responses improve test accuracy (@Gerald Shen):
Method

           | Method              | Accuracy  |
           |--------------------|--------|
           | Hybrid (no replica)| 0.592  |
           | SFT only           | 0.6244 |
           | SFT +DPO           | 0.6437 |
           | Hybrid Model       | 0.65   |

Using different prompts for different policy improvement iterations enhances math test accuracy. Currently, we are testing with the Llama3 8B model (@Gerald Shen).
We are progressing with the Optimal Defense Policy Improvement project that @erick implemented, involving two feedback functions to operationalize the search.
Model Alignment
Collaborating with @Zhilin Wang on the HelpSteer 2 paper, we introduced the SteerLM 2.0 method to address deficiencies in the original SteerLM:
The original SteerLM lacked a mechanism to ensure generated responses closely adhered to desired attribute distributions, leading to suboptimal alignment.
SteerLM 2.0 employs iterative KL minimization between optimal and current policies to better align generated responses with desired attributes.
Results include applying SteerLM 2.0 to the Llama 3 70B model with just a fraction of the training data compared to Llama 3 70B Instruct, achieving an MT-Bench score of 8.28, surpassing both Llama 3 70B Instruct (8.16) and GPT-4-0613 (8.12).
Published on arXiv and integrated into NeMo-Aligner (Thanks @Gerald Shen for reviewing it) with document, facilitating easy training setup with an example job file available here.
 

Attempted SteerLM 340B model alignment, yielding slight inferior results to DPO due to:
Sole reliance on reward model annotated SFT data proved inadequate; ground truth-verified data is needed to improve the SteerLM alignment results further.
Suboptimal alignment in the original SteerLM necessitates consideration of SteerLM 2.0.
Others and Miscellaneous
Presented HelpSteer poster with @Zhilin Wang at NACCL 2024, drawing significant interest from attendees. Some are keen to explore our dataset, with one researcher suggesting our poster should be recognized as the best research.


Jun 21

Mission: Develop advanced models that are in sync with human values

# System 2 Thinking
 * The system 2 research is getting popular in the AI research community, there have been a few research papers emerging recenlty that try different MCTS techniques to solve math problems. We have reviewed our approach and currently identified a few areas to focus on.
   * We plan to build TreeLogic NeMo-Inference-Microservice (NIM) that enables any aligned models to do system 2 thinking at the inference time. We think it can be used to 
      1. solve challenging problems that requires deep searches at inference time, like math, coding, logic problem etc paired with a verifier that can check the results.
      2. generating better quality responses paired with a reward model.
      3. generate high quality positive and negative samples that can be used to refine the model policy and build a stronger model. It should be more efficent than generating synthetic data by random sampling + filtering.

      * We think the alpha-zero approach is good for solving hard gaming problems where the policy needs to be constatant adjusted via tree search to consider the opponent's move. It is an over-kill for solving NLP tasks because we only need a good answer and alpha-zero approach uses too much compute resources. To enable TreeLogic NIM, we have made a few changes:
         * Instead of using LLM output logits as value appromixation, we estimated the state value by greedy sample the tokens using TRTLLM and use the reward from the feedback function as estimation. 
         * To speed up value function estimation, we implemented a hash function that can speeds up the value evaluation for the same reasoning path.
         * Implemmented early stop that stops the tree search immediately once a good solution is found. 
      * We implemented this and tested it on the GSM8K Math problem, it shows orders of magnitude speedup. average search time per sample is reduced from 2400s to 36s. Out of 7473 GSM8K math problems, it can find correct answer for 7452 of them with success rate of 99.7%. 
      * Next step is to try to use reward model as a generic feedback function. At the inference time, it can search for a high reward response defined by the user. 
    * We are testing two generalization hypothesis regarding to the using MCTS to improvle LM math solve capablities.
      * exposing LLM with more responses to the same prompt is helpful for improve the math test accuracy. 
        * We searched for 4 reponses per prompt and use the tree search results to improve the policy using SFT, SFT+DPO and Hybrid model. The following results show more responses is helpful for improving test accuracy. (Gerald)
           | Method              | Accuracy  |
           |--------------------|--------|
           | Hybrid (no replica)| 0.592  |
           | SFT only           | 0.6244 |
           | SFT +DPO           | 0.6437 |
           | Hybrid Model       | 0.65   |
      * Use different prompts for different policy improvement iterations is helpful to improve the math test accuracy. 
        * Currently we are testing it with Llama3 8B model (Gerald). 
    * We are making progress to the optimal defense policy improvement project that @erick implemented the two feedback functions and got the search running.  
# Model Alignment 
  * Working together with Zhilin on the helpsteer paper. Invented a new SteerLM 2.0 method that 
    * Solvin a problem in the oringal SteerLM that it lacks an explicit mechanism to ensure generated responses adhered closely to desired attribute distributions, leading to suboptimal alignment.
    * Introduces iterative KL minimization between the optimal policy and current policy to better align generated responses with desired attributes.
    * Results:
        * Applied to Llama 3 70B model using only 1% of the training data compared to Llama 3 70B Instruct.
        * Achieved an MT-Bench score of 8.28, surpassing Llama 3 70B Instruct (8.16) and GPT-4-0613 (8.12).
    * Publication and Implementation:
      * Method published on arXiv.
      * Integrated into NeMo-Aligner with an example job file available here.
  * Tried to SteerLM 340B model alignment. It didn't produce better results than DPO because 
    1. Using only the reward model annotated SFT data is not enough. Need to use data that is annotated from verifier i.e. the results should be check with ground truth.
    2. original SteerLM has suboptimal aligment. Need to try SteerLM 2.0 method.
# Others and misllaneous
  * Presented HelpSteer poster with Zhilin at NACCL 2024. Got a large crowed of intertested audiance. Some are interested in trying our dataset. Interestly, one researcher commented that this poster should be awarded as the best research. 







# 
Mission: Develop advanced models that are in sync with human values.

System 2 Thinking
Presents the System 2 think project in the 4/23 research report out meeting. Recording is here.
Current status in summary
We have implemented AlphaZero like method that adds tree search planning capabilities to LLM to improve the language model reasoning. This method can solve hard problems such as math, coding, and planning by using ground truth feedback.
We have designed a generic and flexible framework to integrate the tree search method with any language model and any verifier or reward model. We have also optimized the tree search efficiency by adding dynamic branching factor to limit the action space.
We have experimented with different verifiers and reward models to guide the tree search and evaluate the model outputs. We have also enabled the model to interact with different environments and use tools to get more information and feedback.
We have applied the tree search method to various tasks and domains, such as synthetic data generation, model alignment, agent policy improvement, and jailbreak defense. We have some initial promising results and improved the model performance.
We have also explored some alternative ways of doing tree search, such as using policy only network, log probability as value, and expected future value to train value network. We have done comparison between different approaches.
Projects/Results
The synthetic generated math data by the tree search method has been used to improve the model alignment task. We have shown that it helps to improve the 340b model alignment performance on the math category.
Following is a table comparing TreeSearch+LLM with common model alignment methods that it has the benefits of solving hard problems by searching more efficiently.
Methods

Prompt

Responses

Reward/Verifier

Online/Offline

Search Efficiency

RLHF

Yes

No

Reward Model

Online

Less Efficient

DPO

Yes

Yes

Preference Data

Offline

No

LLM+TreeSearch

Yes

No

Reward Model

Offline

Efficient


We compared SFT on the original GSM8K dataset, SFT on the extracted data from tree search results, and policy improve by matching the LM policy to the tree search policy.
Model

Validation Accuracy

Train accuracy

Test accuracy

Raw Model

0.3451

0.3453

0.3172

Policy Improvement

0.662

0.6991

0.629

SFT (Search Data)

0.618

0.6482

0.6194

SFT (Raw Data)

0.575


0.566


We compare the policy only network vs hybrid network (policy + value) and we show they have similar test accuracy performance though the hybrid network has better search accuracy (search without oracle). The value head is important for the tree search to guide the search process to find the solution. By adding more MCTS steps, the accuracy of solving the math problems is improved significantly.


We have seen the validation and test accuracy plateau after the 1st iterations. We hypothesized that it is caused by using the same math problem prompts from iteration to iteration. The tree search is forced to solve the same problem again and again and it won't add any new information. @Gerald Shen is planning to use the OpenMathInstruct data that has 1.8m problem-solution pairs for the next experiments.  This allows the tree search to find optimal policy for different prompts for a different iteration.  
We can barely improve Igor's Mathtool model using the tree search method. The hypothesis is that the model is already very good at solving the math problems. The LLM policy has a low perplexity for the math prompts, and the tree search method is not going to explore and find better answers.
@Makesh Narsimhan Sreedhar is using tree search to find a good reasoning path that leads to correct human helpsteer labels. We would like to build a better reward model using human label as the ground truth signal.
Collaborating with  @Erick Galinkin, @Makesh Narsimhan Sreedhar and @Shaona to work on the jailbreak defense task. By treating the jailbreak and LLM defender as a zero-sum game, we can use the tree search method to find the optimal policy to defend the jailbreak and still provides helpful responses.
@Jiaqi Zeng is exploring an alternative model alignment method that relies on LLM-as-a-judge as the reward model. The hypothesis is that by iteratively improve the LLM, the LLM-as-a-judge capability is also enhanced so we can achieve better model alignment performance.
Tried to use distance between SteerLM condition attributes and the evaluated LLM output responses as the reward signal to improve the LLM policy. The initial result shows the improved policy by tree search has better distance compared with greedy sampled responses.

Working together with @Robert Kirby and @Jialin Song, plan to integrate the tree search with Lang-RL library so it can interact with the more environments and be applied to solve more problems. We plan to work on improving the LLM capabilities as an autonomous software engineer agent. 
SteerLM Model Alignments
Work with @Zhilin Wang to align the 100 percent 340B model. We have shown the helpsteer 1.1 dataset helps to train a reward model that tops the reward model benchmark. We plan to use the reward model to annotate the best SFT data from @Jiaqi Zeng and align the 340B model. In the long term, we plan to apply the tree search method to improve the SteerLM model performance.
Model Alignment Benchmark Evaluation
Added IFEval benchmark to the model alignment launcher which can be used to evaluate the model performance on the detailed instruction following task. Our aligned model has some gaps to follow instructions well, which is found in parallel by @Mingjie Liu.  We have built an IFEval leaderboard to track the performance. We have shown using synthetic instruction following data generated by @Shengyang Sunis helpful to improve the model performance on the IFEval benchmark. He currently is working on to generate a mutiple turn version of the synthetic instruction following data which matches the MT-benchmark distribution better.
@Zhilin Wang added RewardBench to the alignment benchmark evaluation. We can use it to evaluate the reward model performance that covers a wide variety of prompts. We have a RewardBench leaderboard to track the performance. We have shown the helpsteer 1.1 dataset is helpful to train a reward model that tops the reward model benchmark.
@Shengyang Sun added AlpacaEval to the alignment benchmark evaluation. We can use it to evaluate the model alignment performance which is the second most popular model alignment benchmark. We have a AlpacaEval leaderboard to track the performance.
@Olivier Delalleau added GSM8K math benchmark to the alignment benchmark evaluation with a leaderboard to track the performance.


## Mission: Develop advanced models that are in sync with human values.


- System 2 Thinking
    - Presents the System 2 think project in the 4/23 research out meeting 
    - Current status in summary
        - We have implemented AlphaZero like method that adds tree search planing capabilities to LLM to improve the language model reasoning. This method can solve hard problems such as math, coding, and planning by using ground truth feedback and search efficiency. 
        - We have experimented with different verifiers and reward models to guide the tree search and evaluate the model outputs. We have also enabled the model to interact with the environment and use tools to get more information and feedback. 
        - We have applied the tree search method to various tasks and domains, such as synthetic data generation, model alignment, agent policy improvement, and jailbreak defense. We have achieved promising results and improved the model performance on some of the tasks. 
        - We have also explored some alternative ways of doing tree search, such as using policy only network, log probability as value, and expected future value to train value network. We have compared these methods with the hybrid network and the value head. 
        - We have designed a generic and flexible framework to integrate the tree search method with any language model and any verifier or reward model. We have also optimized the tree search efficiency by adding dynamic branching factor to limit the action space. 
    - Projects/Results
        - The synthetic generated math data by the tree search method has been used to improve the model alignment task. We have shown that it helps to improve the model performance on the math category.
        - We compared SFT on the original GSM8K dataset, SFT on the extracted data from tree search results, and policy improve by matching the LM policy to the tree search policy.
        - We compare the policy only network vs hybird network (policy + value) and we show we can achieve similar greedy test performance though the hybrid network has better search performance. The value head is important for the tree search to guide the search process to find the solution. By adding my MCTS steps, the accuracy of solving the math problems is improved significantly.
        We have seen the validation and test accuracy plateaued after 1st iterations. We hypothesized that it is caused by using the same math problem prompts from iteration to iteration. The tree search is forced to solve the same problem again and again and it won't add new inforation. gerald is planning to test the OpenMathInstruct data that has 1.8m problem-solution pairs so the tree search find optimal policy for different prompts for a different iteration.     r  
        - We can barely improve Igor's mathtool math model using the tree search method. The hypothesis is that the model is already very good at solving the math problems and the tree search method does not help much.
        - Makesh is using search to generate synthetic data for the label helpsteer label. The hope is to improve the reward model performance using human label as the ground truth signal.
        - Erick is working on the jailbreak defense task. By treating the jailbreak and LLM defender as a zero-sum game, we can use the tree search method to find the optimal policy to defend the jailbreak.
        - Jiaqi is exploring an alternative model alignment method that relies on LLM-as-a-judge as the reward model. The hope is to by iteratively improve the LLM, the LLM-as-a-judge capability is enhanced so we can achieve better model alignment performance.
        - Tried to use distance between SteerLM condition attributes and the evaluated LLM output responses as the reward signal to improve the LLM policy. The initial result shows the improved policy by tree search has better distance compared with greedy sampled responses.
        - Working together with Robert, plan to integrate the tree search with Lang-RL library so it can interact with the more environments and be applied to solve more problems.
- SteerLM Model Alignments
    - Work with Zhilin to align the 340B model. We have shown the helpsteer 1.1 dataset helps to train a reward model that tops the reward model benchmark. We plan to use the reward model to annotate the best SFT data from jiaqi and align the 340B model. In the long term, we plan to apply the tree search method to improve the model alignment performance.
- Model Alignment Benchmark Evaluation
    - Our aligned model has some gap to follow detailed instruction well. I added IFEval benchmark to the aligner_launcher. We can use it to evaluate the model performance on the detailed instruction following task. We have a IFEval leaderboard to track the performance. We have shown using synthetic data generated by Shengyang is helpful to improve the model performance on the detailed instruction following task.
    - zhilin added RewardBench to the alignment benchmark evaluation. We can use it to evaluate the reward model performance on the model alignment task. We have a RewardBench leaderboard to track the performance. We have shown the helpsteer 1.1 dataset is helpful to train a reward model that tops the reward model benchmark.
    - Shengyang added AlpacaEval to the alignment benchmark evaluation. We can use it to evaluate the model performance on the AlpacaEval task which is the second most popular model alignment benchmark. We have a AlpacaEval leaderboard to track the performance.
    - Olivier added GSM8K math benchmark to the alignment benchmark evaluation.



## Mission: Develop advanced models that are in sync with human values.

- System 2 Thinking
    - Engineering work status
        - We change the design to separate the search and training in different jobs. It has the benefits of using different model configurations for search and training. We serialize the search results in files so it can be reused for training different models. 
        - The trainer of the hybrid model has been implemented and tested on the GSM8k dataset.
        - We have implemented the dual optimizers for training the hybrid model. We make sure the policy network and the value prediction head are optimized independently. 
        - To handle the 4 hr cluster time limit, we have implemented the MCTS search checkpoints so it can load from the previous run. 
        - We have fixed most of the bugs in the MCTSs search and training pipeline. We are at the stage that it is ready to run some experiments.
    - GSM8K preliminary results
        - We have shown, by increasing the number MCTS steps, the accuracy of solving the math problems is improved significantly.
            - The greedy sampling method has an accuracy of 31% using the SteerLM Solar model. 200 MCTS steps has an accuracy of 69.1%. 400 MCTS steps has an accuracy of 73%
        - After training the hybrid model on the search results from 50% of the training data, the accuracy of solving the math problems is improved significantly. 
            |search method|train accuracy| test accuracy|
            |---|---|---|
            |oracle + value|90.5%| 90.2%|
            |no oracle + value|75.6%| 71.3%|
            |oracle + no value|77.8%| 78.0%|
            |greedy(no search)|61%|62%|
        We see the policy network is improved by training on the search results from 31% to 61%. Turing off the oracle and using value head only, the test accuracy reach 71.3%. Turning off the value head and using the oracle only, the test accuracy reach 78.0%. Combine the oracle and value head, the test accuracy reach 90.2%. This is a very promising result that it shows the value head learns the heuristics about the current partial reasoning process. This heuristics can be used to guide the search process to find the solution and generalize well to the unseen problems. 
        - We evaluated the MT benchmark score of the improved hybrid model. The math category has improved by 0.75 and the coding category has improved by 0.35. The other categories have slightly degraded performance. 
    - Next steps
        - Use Python Celery library for the distributed search that it fits well to the be used in the Slurm cluster.
        - Run full experiments on the GSM8k dataset to evaluate the hybrid model performance.
        - Switch to Mistral model for experiments so we can compare the performance with the other research papers.
        - Try harder MATH dataset.

- SteerLM Model Alignments
    - Aligned 30% 340B model with SteerLM, achieving an MT benchmark of 7.7, which is the highest SteerLM MT-benchmark we have achieved. Currently working on aligning the 340B model trained with 50% of the tokens.
    - Aligned the 15B-CT-v2 model with SteerLM, using Shengyang's synthetic generated dataset annotated by 43B steerLM reward model. It achieves MT benchmark of 7.29, which can be used commercially.
    - Aligned the 22B-CT-v2 model with SteerLM, using Jiaqi's wooden earthworm dataset annotated by 43B steerLM reward model. It achieves MT benchmark of 7.39, which is the slightly higher than the SFT model.

- Model Alignment Benchmark Evaluation
    The model alignment benchmark evaluation engineering work is almost done. 
    -  We have scripts to generate MT-benchmark responses, data factory human evaluation responses.
    - Zhilin added scripts to run Truthful QA, MT-benchmark evaluation (GPT and Mixtral), FKGL 
    - Jiaqi added scripts to evaluate the reward model performance both in-distribution and out-of-distribution.
    - Olivier integrated mlops eval-tool and it can be used to evaluate the MMLU model performance 0-shot and 5-shots
    - Shengyang contributed scripts to monitor and run the evaluation using Mixtral for checkpoint selection.

- RNN-like Transformer Development
    - Brent helps to accelerate the numerator forward computation kernel 10x faster than the original implementation. This makes our method to be practical for the training large models efficiently.
    - Zhilin has shown the RNN-like Transformer model improves the language modeling performance with increasing the order of the Taylor expansion. The gap between the original transformer and the RNN-like transformer is small with 8th order expansion.
    - Makesh has shown the RNN-like Transformer model can solve the selective copying and induction heads problems with 100% accuracy in the test set even with order 1 approximation.
    - We plan to focus on practical applications of the RNN-like Transformer model. Hopefully we can use it to solve some real world problems.





## Mission: Develop advanced models that are in sync with human values.

- Launch the System 2 Thinking Project.
    - System 2 represents a slower, more deliberative, and logical thought process. To solve complex problems such as coding, math, and reasoning, our Language Model needs to incorporate System 2 thinking. 
    - We're making significant progress in integrating the AlphaZero algorithm into the Language Model via the NeMo Aligner. This includes:
        - The creation of a tree-based KV-cache for Language Model inference, enhancing the speed of inference for the Monte Carlo Tree Search (MCTS) algorithm.
        - The implementation of Parallel MCTS and self-play algorithms in the NeMo Aligner.
        - A cache system for self-play that has achieved a 10x speed increase, reducing search time from 6 hours to 36 minutes for a single GSM8k problem.
        - The development of a trainer in the NeMo Aligner, which will soon be used to enhance the Language Model's performance.
        - Plans to develop a more memory-efficient cache system due to the current system's high memory usage.
    - The next step involves running benchmarks to evaluate how much MCTS can enhance the Language Model's performance using GSM8k datasets. 
- RNN-like Transformer Development
    - We're exploring a new concept to transform the native Transformer model into an RNN-like Transformer model using the Taylor expansion. This approach offers several benefits:
        - It maintains the original transformer's capabilities through higher-order approximation.
        - It ensures linear time complexity during training.
        - It resembles an RNN during inference by maintaining a fixed-size compressed state with O(1) complexity.
    - We've implemented the forward/backward computation in both Numba and CUDA using techniques like flash attention. This breaks down large tensor computation by computing small tensor tiles in different GPU blocks, making the RNN-like Transformer model computation more memory and compute efficient.
    - We're planning to run benchmarks to evaluate the RNN-like Transformer model's performance on long-range arena datasets and language modeling.
    - This project could be beneficial for super long-range sequence modeling due to its linear time complexity. 
- SteerLM Model Alignments
    - We've aligned several models with impressive results:
        - The 22B-NeMo-CT model achieved an MT benchmark of 6.79
        - SteerLM works out of box to align long context models from Yang's team. We have got the best MT benchmark score 6.45 and 6.31 for 32k and 16k 22b models .
        - The 15B-8T-NeMo-CT model achieved an MT benchmark of 6.89, the highest score among all the NVIDIA NeMo models we've aligned so far.
        - We've also aligned a few open-source models with SteerLM:
            - The 33B Yi-model achieved an MT benchmark of 7.18.
            - The 10B Solar model achieved an MT benchmark of 7.1, significantly higher than our 7B models. This model is being used for the System 2 thinking project.
- Model Alignment Benchmark Evaluation  
    - We're integrating the model alignment benchmark evaluation into the SteerLM-launcher. A team of five is currently working on this project.


- Initiate the System 2 thinking project.
    - System 2 is slower, more deliberative, and more logical way of thinking. Naturally, the language Model needs to be able to do System 2 thinking to solve more complex problems like coding, math and reasoning problems. 
    - We plan to integrate the AlaphaZero algorithm into the Language Model to enable System 2 thinking. Currently we are making great progress in implementing the AlphaZero algorithm in the NeMo Aligner.
        - Implemented the tree based KV-cache for Lanuage Model inference. It can help to speed up inference that fits better for Monte Carlo Tree Search (MCTS) algorithm.
        - Implemented the Parallel MCTS and self play algorithm in the NeMo Aligner.
        - Implemented the cache system for self play and got 10x speed up. It reduces the self play search time from 6 hrs to 36 minutes for one GSM8k problem.
        - Gerald is helping to implement the trainer in the NeMo Aligner. Soon we can use it to improve the performance of the Language Model.
        - The cache system uses a lot of memory. In the future, we need to implement a more memory efficient cache system.
    - Next step is to run some benchmarks to see how much MCTS can improve the performance of the Language Model using GSM8k datasets. 
- RNN-like Transformer
    - I have come up with a new idea to convert the native Transformer model into a RNN-like Transformer model by using the Taylor expansion. It has the advantages of:
        - mathematically it has the same capability as the original transformer by using higher order approximation.
        - The training time complexity is linear in sequence time.
        - At the inference, it is similar to RNN where it maintains a fixed size compressed state which has complexity of O(1).
    - I have implemented the forward/backward computation in both Numba and CUDA using the technique like flash attention. I break down large tensor computation by computing small tiles of the tensor in different GPU blocks. So the RNN-like Transformer model can be computed in a memory and compute efficient way. @doris is going to help to make the kernel computation more efficient.
    - @zhiling, @makesh are going to help out to run some benchmarks to see how the RNN-like Transformer model performs on long range arena datasets and language modeling.
    - This project can be useful for super long range sequence modeling because of the linear time complexity. 
- SteerLM model alignments
    - Aligned 22B-NeMo-CT model, MT benchmark 6.52, which is higher than the previous score 6.2. 
    - Aligned 15B-8T-NeMo-CT model, MT benchmark 6.89, which achieves the highest score among all the NVIDIA NeMo models we aligned so far.
    - Tried to align a few open source models with SteerLM and got good results. 
        - Aligned the 33B Yi-model with SteerLM, MT benchmark 7.18. 
        - Aligned the 10B Solar model with SteerLM, MT benchmark 7.1, much higher than the our 7B models. It serves as the model that I used for the System 2 thinking project.
- Model alignment benchmark evaluation  
    - Organize the model alignment benchmark evaluation integration into the SteerLM-launcher. We have a team of 5 people working on this project.


### Activities for Model Alignment
- We have developed the best commercially viable open-source chat model based on Llama2 70B, using the SteerLM method.
    - It has an MT-benchmark score of 7.54, making it the highest-scoring open-source model for commercial use.
    - The HelpSteer dataset, which made this possible, is open-sourced under the cc-by-4.0 license at Huggingface. You can access it here: https://huggingface.co/datasets/nvidia/HelpSteer
    - The HelpSteer paper is published on archive at http://arxiv.org/abs/2311.09528. Our model alignment team conducted an extensive study on this dataset, comparing multiple alignment methods with ablation studies.
    - The key takeaway is that anyone can align a top-performing model for commercial use using our released data and the NeMo framework. SteerLM is also one of the most efficient model alignment techniques that is easy to use.
    - Insights from the HelpSteer Paper:
        - SteerLM is more effective for larger base models than smaller ones. Our model alignment strategy should prioritize aligning large models. For smaller models, we can use model distillation.
        - The value model is crucial for successful model alignment. We have developed internal benchmarks for evaluating value models.
            - Larger value models perform better out of distribution (they have better generalization capability).
            - It's important to increase the diversity of training datasets for value models. We've seen improvements from using both OASST and HS for training value models.
            - For attribute-conditioned SFT, the quality of the data is important for producing high-quality aligned models. Fortunately, for larger model alignment, we don't need many SFT dataset samples.
            - Regression value models outperform the LM-based value models because they model the correlation between different attributes more accurately.
        - We plan to collect a second round of SteerLM data, focusing on:
            - Multi-turn and more natural human prompts using sharedGPT.
            - Increasing the diversity of the responses evaluated by the SteerLM value model.
- NVIDIA Base Model SteerLM Alignment
    - We've aligned the academia benchmark-enhanced 8B model with SteerLM, achieving an MT benchmark of 5.85, which is higher than the previous score of 5.6.
    - We've also aligned the latest 22B model at intermediate steps, achieving an MT benchmark of 6.52, which is significantly higher than the previous 43B model alignment.
- More Efficient SteerLM Training
    - We're collaborating with @Adi Renduchintala to experiment with LoRa + SteerLM.






## Mission: Create the next generation of smarter models that align with human values.

### Model alignment activities
- Best Open Source, commercially friendly chat model based on Llama2 70B, aligned by SteerLM method.
    - MT-benchmark score of 7.54, highest open-source model for commercial use.
    - The enabling HelpSteer dataset is open sourced with cc-by-4.0 license at Huggingface. https://huggingface.co/datasets/nvidia/HelpSteer
    - The HelpSteer paper is out at archive http://arxiv.org/abs/2311.09528. Our model alignment team did a thorough study on this dataset comparing multiple alignment methods with ablation studies.
    - The significance is everyone can align a best performing model for commercial use using our released data and NeMo framework. SteerLM is also one of the most efficient model alignment techniques that is easy to use.
    - Lessons learned from the HelpSteer Paper:
        - SteerLM works better for large base models than smaller models. Our model alignment strategy should focus on aligning large models. We can do model distillation for smaller size models.
        - The value model is the key to successful model alignment. We have developed internal value model eval benchmarks.
            - Large value models have better out of distribution performance (better generalization capability).
            - Increasing the diversity of training datasets for value models is important. We see gains from using both OASST and HS for training value models.
            - For attribute conditioned SFT, the data quality matters for producing good quality aligned models. Luckily, for larger model alignment, we don't need many SFT dataset samples.
            - Regression value models work better than the LM-based value models due to more accurate modeling of the correlation between different attributes.
        - Plan to collect 2nd round of SteerLM data. Focusing on
            - Multiple turn and more natural human prompts using sharedGPT.
            - Increasing the diversity of the responses evaluated by the SteerLM value model
- NVIDIA base model SteerLM alignment
    - Aligned academia benchmark enhanced 8B model with SteerLM, MT benchmark 5.85, which is higher than the previous score 5.6
    - Aligned the latest 22B model at intermediate steps, MT benchmark 6.52, much higher than the previous 43B model alignment.
- More efficient SteerLM training
    - Working with @Adi Renduchintala to experiment LoRa + SteerLM for efficient model alignment.
- Internal SteerLM launcher
    - Developed the steerLM launcher Yi Dong / steerlm_launcher · GitLab (nvidia.com)
    - Fully automated solution for SFT/SteerLM training, evaluation, and scoring on Slurm cluster. Everyone can try SteerLM method with ease.
- Collaborate with @Dan Su to test whether adding SteerLM control labels at the pretraining stage is helpful for SteerLM alignment.
