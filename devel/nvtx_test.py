import evolvepy as ep
import numpy as np
from evolvepy.integrations import nvtx

chromosome = np.random.rand(100)
existence_rate = np.random.rand()
gene_rate = np.random.rand()
mutation_range = np.random.rand(2)
mutation_range = np.sort(mutation_range)

profile_range = nvtx.start_range("profile", domain="evolvepy")

nvtx.push_range("sum_mutation compilation", domain="evolvepy", category="sum")
new_ch = ep.generator.mutation.sum_mutation(chromosome, existence_rate, gene_rate, mutation_range)
nvtx.pop_range(domain="evolvepy")

for i in range(100):
    nvtx.push_range("sum_mutation", domain="evolvepy", category="sum")
    new_ch = ep.generator.mutation.sum_mutation(chromosome, existence_rate, gene_rate, mutation_range)
    nvtx.pop_range(domain="evolvepy")

new_ch = ep.generator.mutation.mul_mutation(chromosome, existence_rate, gene_rate, mutation_range)

nvtx.push_range("sum_mutation no JIT", domain="evolvepy", category="sum")
new_ch = ep.generator.mutation.sum_mutation.py_func(chromosome, existence_rate, gene_rate, mutation_range)
nvtx.pop_range(domain="evolvepy")

nvtx.end_range(profile_range)