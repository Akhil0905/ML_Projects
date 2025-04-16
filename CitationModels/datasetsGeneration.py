#!/usr/bin/env python
# coding: utf-8

# # 1) Citation Recommendation

# In[1]:


import os
import pandas as pd
import json
import time
import numpy as np
import csv

# Parameters
dataset_file = "arxiv-metadata-oai-snapshot.json"
output_file = "recommendationDataset.csv"
batch_size = 100000
num_samples = 100000  
pair_per_batch = 1000  

print("Dataset generation...")

start_time = time.time()
line_count = 0
total_samples = 0
processed_records = []

# Open the CSV file for writing using csv.writer for proper escaping
with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar='\\')
    # Write header
    writer.writerow(["citing_sentence", "citing_paper_date", "cited_paper_title", 
                     "cited_paper_abstract", "cited_paper_date", "label"])

    # Stream-process JSON
    with open(dataset_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                paper_id = record.get("id", "").strip()
                title = record.get("title", "").replace("\n", " ").strip()
                abstract = record.get("abstract", "").replace("\n", " ").strip()
                date = record.get("update_date", "").strip()

                # Skip entries with missing or incomplete data
                if not title or not abstract or not date:
                    continue

                # Store in the records list
                processed_records.append({"id": paper_id, "title": title, "abstract": abstract, "date": date})

                line_count += 1

                # If batch size is reached, process and save to CSV
                if len(processed_records) >= batch_size:
                    df = pd.DataFrame(processed_records)

                    # Generate limited number of random citation pairs
                    sampled_pairs = []
                    for _ in range(pair_per_batch):
                        try:
                            # Randomly select two different papers from the batch
                            sampled_papers = df.sample(2)
                            paper_a = sampled_papers.iloc[0]
                            paper_b = sampled_papers.iloc[1]

                            citing_sentences = paper_a['abstract'].split('. ')
                            citing_sentence = np.random.choice([s.strip() for s in citing_sentences if s.strip()])

                            # Check for empty or problematic sentences
                            if not citing_sentence or not paper_b['title'] or not paper_b['abstract']:
                                continue

                            sample = {
                                'citing_sentence': citing_sentence,
                                'citing_paper_date': paper_a['date'],
                                'cited_paper_title': paper_b['title'],
                                'cited_paper_abstract': paper_b['abstract'],
                                'cited_paper_date': paper_b['date'],
                                'label': int(np.random.rand() > 0.5)  # Random binary label
                            }

                            # Write the cleaned sample to CSV
                            writer.writerow([
                                sample['citing_sentence'],
                                sample['citing_paper_date'],
                                sample['cited_paper_title'],
                                sample['cited_paper_abstract'],
                                sample['cited_paper_date'],
                                sample['label']
                            ])

                            total_samples += 1

                        except Exception as e:
                            print(f"Error generating sample: {e}")
                            continue

                    processed_records = []

                    # Progress update
                    elapsed_time = time.time() - start_time
                    print(f"Processed {line_count:,} lines... ({elapsed_time:.2f} seconds elapsed)")
                    print(f"Total cleaned samples generated: {total_samples}")

            except Exception as e:
                print(f"Error processing line {line_count}: {e}")
                continue

# Final stats and completion message
total_time = time.time() - start_time
print(f"Dataset preprocessing complete! Processed {line_count:,} lines in {total_time:.2f} seconds.")
print(f"Total cleaned samples generated: {total_samples}")
print(f"Cleaned dataset saved to: {output_file}")


# # 2) Citation Explaination

# In[1]:


import os
import pandas as pd
import json
import time
import numpy as np
import csv

# Parameters
dataset_file = "arxiv-metadata-oai-snapshot.json"
output_file = "explanationDataset.csv"
batch_size = 100000
max_samples = 100000  
context_size = 3

print("Starting citation explanation dataset generation...")

start_time = time.time()
line_count = 0
total_samples = 0
processed_records = []

# Open the CSV file for writing using csv.writer for proper escaping
with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar='\\')
    # Write header
    writer.writerow(["context_before", "citation_sentence", "context_after", "cited_paper_title", "cited_paper_abstract"])

    # Stream-process JSON
    with open(dataset_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                paper_id = record.get("id", "").strip()
                title = record.get("title", "").replace("\n", " ").strip()
                abstract = record.get("abstract", "").replace("\n", " ").strip()
                date = record.get("update_date", "").strip()

                # Skip entries with missing or incomplete data
                if not title or not abstract or not date:
                    continue

                # Store in the records list
                processed_records.append({"id": paper_id, "title": title, "abstract": abstract, "date": date})

                line_count += 1

                # If batch size is reached, process and save to CSV
                if len(processed_records) >= batch_size:
                    df = pd.DataFrame(processed_records)

                    # Generate citation explanation samples
                    while total_samples < max_samples:
                        try:
                            # Randomly select two different papers from the batch
                            sampled_papers = df.sample(2)
                            citing_paper = sampled_papers.iloc[0]
                            cited_paper = sampled_papers.iloc[1]

                            citing_sentences = citing_paper['abstract'].split('. ')
                            citing_sentences = [s.strip() for s in citing_sentences if s.strip()]
                            if len(citing_sentences) < context_size * 2 + 1:
                                continue

                            # Randomly choose a central sentence to be the citation sentence
                            citation_idx = np.random.randint(context_size, len(citing_sentences) - context_size)

                            context_before = citing_sentences[citation_idx - context_size : citation_idx]
                            citation_sentence = citing_sentences[citation_idx]
                            context_after = citing_sentences[citation_idx + 1 : citation_idx + context_size + 1]

                            # Format the context as single strings
                            context_before_str = " ".join(context_before)
                            context_after_str = " ".join(context_after)

                            # Prepare sample row
                            sample = [
                                context_before_str,
                                citation_sentence,
                                context_after_str,
                                cited_paper['title'],
                                cited_paper['abstract']
                            ]

                            # Write the sample to CSV using the writer object
                            writer.writerow(sample)
                            total_samples += 1

                            # Stop if we have generated enough samples
                            if total_samples >= max_samples:
                                print(f"Reached max_samples limit of {max_samples}.")
                                break

                        except Exception as e:
                            print(f"Error generating sample: {e}")
                            continue

                    # Clear processed records and free memory
                    processed_records = []
                    elapsed_time = time.time() - start_time
                    print(f"Processed {line_count:,} lines... ({elapsed_time:.2f} seconds elapsed)")
                    print(f"Total samples generated: {total_samples}")

            except Exception as e:
                print(f"Error processing line {line_count}: {e}")
                continue

# Final stats and completion message
total_time = time.time() - start_time
print(f"Dataset preprocessing complete! Processed {line_count:,} lines in {total_time:.2f} seconds.")
print(f"Total samples generated: {total_samples}")
print(f"Reduced dataset saved to: {output_file}")
