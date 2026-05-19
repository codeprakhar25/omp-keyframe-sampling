"""Central configuration for the pilot. All run parameters live here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Models (pinned)
    target_primary: str = "claude-sonnet-4-6"
    target_cross: str = "gpt-4.1"  # update to actual GPT-5.x model ID when available/confirmed
    reformulator: str = "claude-haiku-4-5-20251001"
    messifier: str = "claude-haiku-4-5-20251001"

    # Decoding
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 2048

    # Dataset
    dataset_name: str = "bigcode/bigcodebench-hard"
    n_tasks_primary: int = 30
    n_tasks_cross: int = 15
    min_prompt_words: int = 80

    # Sampling
    n_seeds_primary: int = 5
    n_seeds_cross: int = 3
    seed_list_primary: tuple = (1, 2, 3, 4, 5)
    seed_list_cross: tuple = (1, 2, 3)


from dotenv import load_dotenv
load_dotenv()

CFG = Config()
