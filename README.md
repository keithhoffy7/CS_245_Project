# Example Agents - Setup and Usage Guide

This project contains different recommendation agents that can be used with the Web Society Simulator. This guide will help you set up and run these agents.

## Installing Dependencies

You can install all dependencies using `pip`:

```bash
pip install websocietysimulator
```

```bash
pip install -e .
```

## API Keys Setup

The agents require API keys for the LLMs they use. Most of the agents use Gemini, but we also have a baseline agent using GPT. Set these as environment variables before running the agents.

### Gemini API Key

For agents using Gemini (most agents in this folder):

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

The above will temporarily set the environment variable for your current terminal instance. To permanently set the environment variable:

```bash
echo 'export GEMINI_API_KEY="your-gemini-api-key-here"' >> ~/.bashrc
```

Restart your terminal after the above command.

### OpenAI API Key

For agents using GPT (only the base agent):

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

The above will temporarily set the environment variable for your current terminal instance. To permanently set the environment variable:

```bash
echo 'export OPENAI_API_KEY="your-openai-api-key-here"' >> ~/.bashrc
```

Restart your terminal after the above command.

## Data Setup

### 1. Process Raw Data

Before running the agents, you need to process the raw datasets. The processed data should be placed in a directory that the simulator can access. Our agents were created and tested using the amazon dataset, so these are the instructions for processing that data:

**Required data files:**
- `item.json` - Processed item data
- `review.json` - Processed review data
- `user.json` - Processed user data

**Processing the data:**

Ensure you are in the `/CS_245_Project` directory to begin. Then follow these steps:

```bash
python data_process.py --input_dir <path_to_raw_datasets> --output_dir <path_to_processed_output>
```

**Required raw data files:**
- **Amazon**: 
   - `Industrial_and_Scientific.csv`, 
   - `Musical_Instruments.csv`, 
   - `Video_Games.csv`,
   - `Industrial_and_Scientific.jsonl`, 
   - `Musical_Instruments.jsonl`, 
   - `Video_Games.jsonl`,
   - `meta_Industrial_and_Scientific.jsonl`, 
   - `meta_Musical_Instruments.jsonl`, 
   - `meta_Video_Games.jsonl`

See `/tutorials/data_preparation.md` for more detailed data preparation instructions.

### 2. Data Directory Path

The agents expect the processed data to be at `/srv/output/data1/output` because this is where we kept the data on our virtual machine we used for testing. If your data is in a different location, you'll need to modify the `data_dir` parameter in ALLL the agent scripts:

```python
simulator = Simulator(data_dir="/path/to/your/processed/data", device="auto", cache=False)
```

### 3. Task and Groundtruth Files

The agents also need task files and groundtruth files to run the sumulation. These are located in:
- `/example/track1` - Track 1 tasks (amazon, goodreads, yelp)
- `/example/track2` - Track 2 tasks (amazon, goodreads, yelp)

Each dataset folder contains:
- `/tasks` - Task definition files
- `/groundtruth` - Groundtruth ranking files

These paths in the agent scripts are absolute paths from the virtual machine we used for testing. You will need to update them to the absolute path on your machine. Or, you can use the relative path (but then you have to run the agents from inside the `/example` folder):

```python
simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")
```

Also ensure that `task_set` is set to "amazon" (should be already set in all our agents).

## Building the BPR Model Pickle File

Some agents use a BPR (Bayesian Personalized Ranking) model for recommendations. Before running these agents, you need to train and save the BPR model as a pickle file.

### Prerequisites

1. Ensure you have the `implicit` library installed:
   ```bash
   pip install implicit
   ```

2. Make sure you have processed your review data (see Data Setup section above). The training script expects a `review.json` file.

### Training the BPR Model

1. Navigate to the example directory:
   ```bash
   cd example
   ```

2. Update the review file path in `train_bpr_recommender.py` if needed:
   - The default path is `/srv/output/data1/output/review.json`
   - If your review data is in a different location, update the `REVIEW_FILE` variable

3. Run the training script:
   ```bash
   python3 train_bpr_recommender.py
   ```

4. The script will save the model to `/srv/CS_245_Project/example/bpr_model.pkl` or at any other path specified in the code. 

## Running the Agents

You can run the agents by simply running the python script containing the agent. The names correspond to what advanced strategies were used for that agent (unless it is a base agent). For example, to run the Gemini Baseline Agent:

```bash
python3 gemini_base_agent.py
```

Note that the agents allow you to configure certain parameters when running them, such as the number of workers running the tasks and how many tasks to execute. For the OpenAI base agent we had to set the number of workers to 2 to prevent too many requests per minute.

## Project Notes

1. We cloned the original AgentSociety Challenge GitHub repository and added our own commits with our improved agents to the repository. 

2. All of our group members were working on the project in the same cloned repository in a GCP virtual machine, so most of the commits are from the same user (Kriteen Jain).

3. We used various LLM tools to suggest strategies to improve our agents.