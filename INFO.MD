# Information on the Scripts and Data

## Raw Data Inputs 
- Paper Input in text-form: *papers_original.txt*
- Excel-Sheet with final human screening decision for the baseline task: *shuffled_master_sheet.xlsx*
- Excel-Sheet with final human screening decisions for the sub-tasks: *master-sheet-full.xlsx*
- Excel-Sheet with both human screening decisions for the baseline task: *human_inclusion_decisions_only.xlsx*

## Coding Prompts
- Baseline Coding Prompt: *coding_prompt_2.txt* 
- Nonbinary Coding Prompt: *coding_prompt_3.txt*
- Subtask-1 Coding Prompt: *coding_prompt_6_meta-analysis.txt*
- Subtask-2 Coding Prompt: *coding_prompt_7_meta-analysis_mathematics.txt*
- Subtask-3 Coding Prompt: *coding_prompt_8_meta-analysis_mathematics_tech.txt*
- Subtask-4 Coding Prompt: *coding_prompt_9_parents.txt*

## Matched Sheets
- After Conducting the Screening, the LLMs decisions (that are in text files we did not upload) get matched with the human screening decision(s). 
- For each run, an individual matched-sheet (which is an excel sheet) is created. 
- These matched sheets are later used for the different analyses

**Naming Convention**
- the number after the *matched_master_sheet* denotes the coding prompt that was used. E.g., the baseline coding prompt had the number 2, which means that all matched-sheets with the number 2 were conducted on the baseline data. For the sub-tasks, the sheets start with *matched_master_sheet_full*.
- for gpt-5-mini, reasoning effort is specified after the number. Medium reasoning effort was the default and was thus not specified
- temperature is only specified for temperature 1, temp. 0 is not specified as it is the default
- When we conducted multiple runs on the same condition, run 2 was specified with the letter *b* together with the number, run 3 was specified with the letter *c*
- the model is specified after the number (& reasoning effort & temperature in cases where they apply)
- batch-size is specified after the model

## Analysis Scripts

### LLM Screening
- *01_coding_api.py* -- main screening script that uses the prompt and the original papers to conduct the LLM screening with OpenAI Models. Here, all settings can be adjusted. An OpenAI API Key is needed for this script to work.
- *01b_coding_local.py* -- screening script for the local models via Ollama. All settings can be adjusted here. 

### Matching
- *02_matching.py* -- script that matches the screening outputs with the human screening decisions for the baseline data.
- *02b_matching_detailed_coding.py* -- script that matches the screening ouputs with the screening sub-tasks. 

### Overlap and Variability
- *03_overlap.py* -- script that calculates the overlap for each run using the binary screening procedure. Uses the matched sheet as a basis for the analysis.
- *03b_overlap.py* -- similar to the general overlap sheet, but uses the non-binary procedure. Here, the threshold that gets used to transform the non-binary to binary decisions can be specified.
- *03c_overlap_multiple_runs.py* -- script that can assess the overlap for our strategy to combine multiple runs for screening decisions. 
- *03d_overlap_detailed_coding.py* -- similar to the general overlap sheet, used for the screening subtasks. 
- *03e_record-level_variability.py* -- Multiple matched-sheets can be input, script calculates the different outcomes regarding record-level variability within the manuscript. 
- *03f_human-disagreement_LLM-errors.py* -- Uses the matched sheets and the human screening sheet with both screeners decisions to estimate whether LLM errors were more likely to occur in records with human disagreements in their decisions. 
- *03g_record-variability_human-disagreement.py* -- combines the ideas from the previous 2 analyses to investigate whether LLM screening decisions were more variable for records with human screening disagreements. 

### Simulation
- *07_simulating_workflow.py* -- Based on a matched-sheet / multiple matched-sheets, draws of subsamples of this data and how these drawn samples generalize to the full data are estimated. Sample proportions can be specified in the script, as well as the number of draws to estimate uncertanties for the effects. 

### Confidence Intervals
- *08_confidence-intervals.py* -- takes matched-sheets as the input and calculates Wilson score intervals for Accuracy, precision, and recall. Bootstrapped intervals are calculated for Cohen's kappa.
- *08b_confidence-comparisons.py* -- for each matched-sheet included in the script, pairwise comparisons between all metrics are bootstrapped that allow for comparisons between models and settings or different runs by the same model. 