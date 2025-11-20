from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward
from trl import GRPOConfig

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

args = GRPOConfig("Qwen/Qwen2-0.5B-Instruct-GRPO")
args.use_liger_kernel = True
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    args=args,
)
trainer.train()