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
