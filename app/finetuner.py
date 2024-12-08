from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from huggingface_hub import HfFolder, create_repo
from qdrant_client import QdrantClient
import json
from tqdm import tqdm
import bitsandbytes as bnb
from accelerate import Accelerator
from typing import List, Dict, Optional
from configs import HF_TOKEN, QDRANT_API_KEY, QDRANT_URL
from helpers.mongo_client import get_mongo_client
from loguru import logger


@dataclass
class FineTuningConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Base model to fine-tune
    new_model_name: str = (
        None  # Your model name on HuggingFace (e.g., "username/my-awesome-model")
    )
    hf_token: str = None  # Your HuggingFace token
    private_model: bool = False  # Whether to make the model private on HuggingFace
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    learning_rate: float = 2e-4
    max_length: int = 512
    warmup_ratio: float = 0.03


# [Previous Dataset and other class implementations remain the same until FastFineTuner]


class InstructDataset(Dataset):
    def __init__(self, tokenizer, data: List[Dict[str, str]], max_length: int):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        answer = item["answer"]

        # Create prompt in the format the model expects
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{answer}</s>"

        # Tokenize with truncation
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": encoded["input_ids"][0].clone(),
        }


class FastFineTuner:
    def __init__(
        self,
        config: FineTuningConfig,
        mongo_client,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        self.config = config
        self.qdrant_client = qdrant_client
        self.mongo_client = mongo_client
        self.collection_name = collection_name
        self.accelerator = Accelerator()

        # Set up HuggingFace token if provided
        if self.config.hf_token:
            HfFolder.save_token(self.config.hf_token)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_model(self):
        """Prepare model with 4-bit quantization and LoRA."""
        # Load model in 4-bit
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model_name,
        #     # load_in_4bit=True,
        #     torch_dtype=torch.bfloat16,
        #     quantization_config={
        #         "load_in_4bit": True,
        #         "bnb_4bit_compute_dtype": torch.bfloat16,
        #         "bnb_4bit_use_double_quant": True,
        #         "bnb_4bit_quant_type": "nf4",
        #     },
        # )

        # # Prepare model for k-bit training
        # model = prepare_model_for_kbit_training(model)

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)

        return model

    def load_data_from_qdrant(self) -> List[Dict[str, str]]:
        """Load instruction-answer pairs from Qdrant."""
        # Scroll through all points in the collection
        points = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Adjust based on your dataset size
        )[0]

        # Extract instruction-answer pairs from payloads
        data = []
        for point in points:
            if "instruction" in point.payload and "answer" in point.payload:
                data.append(
                    {
                        "instruction": point.payload["instruction"],
                        "answer": point.payload["answer"],
                    }
                )

        return data

    def load_data_from_mongo(self):
        """Load instruction-answer pairs from MongoDB."""

        # Load raw documents from MongoDB

        db = self.mongo_client["rag"]
        collection = db["instruct_set"]

        data = []
        for doc in collection.find():
            data.append(
                {
                    "instruction": doc["instruction"],
                    "answer": doc["answer"],
                }
            )

        return data

    def create_huggingface_repo(self):
        """Create a new repository on HuggingFace Hub."""
        if not self.config.new_model_name:
            raise ValueError(
                "new_model_name must be specified in config to push to HuggingFace Hub"
            )

        try:
            create_repo(
                repo_id=self.config.new_model_name,
                private=self.config.private_model,
                exist_ok=True,
            )
            print(f"Repository {self.config.new_model_name} created successfully")
        except Exception as e:
            print(f"Error creating repository: {e}")
            raise

    def push_to_hub(self, model, tokenizer):
        """Push the fine-tuned model to HuggingFace Hub."""
        if not self.config.new_model_name:
            print("Skipping push to hub as new_model_name is not specified")
            return

        print(f"Pushing model to HuggingFace Hub as {self.config.new_model_name}")

        try:
            # Create model card content
            model_card = f"""
                # {self.config.new_model_name}

                This model is a fine-tuned version of {self.config.model_name} using LoRA.

                ## Training Details
                - Base model: {self.config.model_name}
                - LoRA rank (r): {self.config.lora_r}
                - LoRA alpha: {self.config.lora_alpha}
                - Training epochs: {self.config.num_epochs}
                - Batch size: {self.config.batch_size}
                - Learning rate: {self.config.learning_rate}
                            """

            # Save and push everything
            model.push_to_hub(
                self.config.new_model_name,
                private=self.config.private_model,
                commit_message="Add fine-tuned model",
            )
            tokenizer.push_to_hub(
                self.config.new_model_name,
                private=self.config.private_model,
                commit_message="Add tokenizer",
            )

            # Push the model card
            with open("README.md", "w") as f:
                f.write(model_card)

            print(f"Successfully pushed model to {self.config.new_model_name}")

        except Exception as e:
            print(f"Error pushing to hub: {e}")
            raise

    # def train(self):
    #     """Run the fine-tuning process and push to HuggingFace Hub."""
    #     print("Loading data from Qdrant...")
    #     train_data = self.load_data_from_qdrant()

    #     print("Preparing model...")
    #     model = self.prepare_model()

    #     # Create HuggingFace repository if needed
    #     if self.config.new_model_name:
    #         self.create_huggingface_repo()

    #     # [Previous training loop code remains the same until saving]

    def train(self):
        """Run the fine-tuning process."""
        logger.info("Loading data from Mongo...")
        train_data = self.load_data_from_mongo()

        # print(train_data)

        # return

        # logger.info("Preparing model...")
        # model = self.prepare_model()

        # Create dataset and dataloader
        dataset = InstructDataset(self.tokenizer, train_data, self.config.max_length)

        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        print(DataLoader)

        print(dataset)

        return

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=self.config.warmup_ratio,
            optim="paged_adamw_8bit",
        )

        # Prepare for distributed training
        model, dataloader = self.accelerator.prepare(model, dataloader)

        # Training loop
        model.train()
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=self.config.learning_rate
        )

        total_steps = len(dataloader) * self.config.num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")

        for epoch in range(self.config.num_epochs):
            for batch in dataloader:
                with self.accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

        # Save the fine-tuned model
        model.save_pretrained("./fine_tuned_model")

        # After training, push to HuggingFace Hub
        self.push_to_hub(model, self.tokenizer)


def main():
    # Initialize configuration
    config = FineTuningConfig(
        # model_name="meta-llama/Llama-2-7b-hf",
        model_name="unsloth/Meta-Llama-3.1-8B",
        new_model_name="anindaghosh/cs-gy-6613-rag-project-test",  # Replace with your desired model name
        hf_token=HF_TOKEN,  # Replace with your token
        private_model=False,  # Set to False for public model
    )

    # Initialize Mongo
    mongo_client = get_mongo_client()

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Initialize fine-tuner
    fine_tuner = FastFineTuner(
        config=config,
        qdrant_client=qdrant_client,
        mongo_client=mongo_client,
        collection_name="rag_vectors",
    )

    # Run fine-tuning and push to hub
    fine_tuner.train()


if __name__ == "__main__":
    main()
