The files with a postfix of "retry" have logic written to retry only specific records.<br>
The files (summaryAgent, testAgent, tableAgent) essentially execute specific prompts to score the closeness of the answer generated by the framework to the ground truth.<br>
scoreFileGenerator.py is used to convert the files that are generated above into a file containing a list of JSON objects, are we regularly dump json objects to a file to avoid any loss due to some system failure.<br>
The fiels with the rpefix "preprocess" are used to add some fields liek ground_truth and evidence from the dataset file to the result files generated by the individual agents, which is then passsd to the LLM for scoring.
