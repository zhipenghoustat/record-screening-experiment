## Record Screening Experiment

This repo contains a Python script for simulation experiment for the Paper "Enhancing Recall in Automated Record Screening: A Resampling Algorithm".

The script is a versatile tool for running experiments and checking how well the resampling algorithm from the paper works under different situations. It's set up to explore a wide range of scenarios and parameters, providing useful insights into how the algorithm behaves. 

### Function Definitions

- `calculate_sample_size(R, c)`

    Calculates required sample sizes based on specified parameters.

- `sequence_gen_random(N, prevalence, seed)`

    Generates a random sequence of data with a given prevalence.

- `sequence_gen_simulated(N, prevalence, seed)`

    Generates a simulated sequence of data with priority scores.

- `sequence_gen_real()`

    Retrieves real data from CSV files and ranked IDs from a serialized file.

- `record_sampling(y_full, k, seed)`

    Performs record sampling to select a subset of records.

- `result_analysis(y_full, id_ranked, sample_list, c)`

    Analyzes the sampled data to calculate recall, workload, and related metrics.

- `experiment(N, prevalence, R, c, seed, round, mode)`

    Conducts experiments or simulations with specified parameters and collects relevant statistics.

### Modes

The script conducts experiments in different modes:

    - "sim" mode involves simulating data with simulated sequences.
    - "random" mode simulates random data sequences.
    - "real" mode uses real data retrieved from CSV files. It loads real data from CSV files and evaluates the system's performance using the same metrics.





