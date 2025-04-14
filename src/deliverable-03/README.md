# Deliverable 3 - ModelBot, an Agentic helper for doing basic Machine Learning tasks

## Summary

ModelBot is an Agentic helper designed to perform regression and classification tasks from csv data and produce a PDF report for an end-user. ModelBot is designed to require minimal data science knowledge for operation and is powered by native Python libraries such as NumPy, Pandas, scikit-learn, Matplotlib, and Seaborn, with agents utilizing the Llama 3, 8 billion parameter model accessed from [Together.ai](https://www.together.ai/). As such, the ModelBot requires a Together API key to function.

The current version of ModelBot is limited in scope. A limited amount of preprocessing is performed on the dataset before modeling. Currently, the only models used for regression are Linear Regression, Polynomial (Degree 3) Regression using Lasso regularization, and ElasticNet Regression, and the only models used for classification are Logistic Regression, Decision Tree Classifiers, and Random Forest Classifiers. A small hyperparameter grid can be searched at the user specification to perform limited model tuning; however, the grid is deliberately small to reduce fitting time. The expected performance of ModelBot is significantly lower than that of a human data scientist, as the preprocessing and modeling steps are relatively simplistic.

## Basic Workflow / User Journey Description

The ModelBot is designed to be used by users with little to no Data Science knowledge. The user can request descriptions of the possible API functions, their trigger words, and their requirements by entering "list functions".

- A chat is instantiated from a fresh [ModelBot](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/interface.py), creating a fresh [DataHolder object](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/dataholder.py) and loading the API function metadata from [metadata.json](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/metadata.json)
- User interacts with the chatbot, sending messages.
- The messages are parsed and checked for trigger words to see if any APIs are being requested by the user using detect_api_call() in [helper.py](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/helper.py)
- If no API calls are detected, the chatbot responds to the user. Otherwise, the API call is processed using process_api_call() in [helper.py](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/helper.py)
- process_api_call() first checks the metadata to see what prerequisite data must be present in the DataHolder. If there is prerequisite data that is needed and that data is not in the DataHolder, process_api_call() is recursively called until an API call is successfully initiated or otherwise fails gracefully.
- If there is no prerequisite data, process_api_call() then checks for user input data for the API function in the DataHolder. Any data not in the DataHolder is requested from the user via inputs.
- Once all of the prerequisites and input data are present in the DataHolder, the [API function](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/api_list.py) is invoked. Any data that needs to be saved is added to the DataHolder object for future API calls. Status messages are displayed back to the user in the event stream, and the function returns a response dictionary denoting success or failure of the API call. If the API call failed, user input gathered for the API function call is removed from the DataHolder so that the user can retry later.
- The success status of the API call is displayed to the user, and the chat continues.

An example successful classification workflow (at any point, if a step fails, it exits gracefully and restarts the conversation loop):
- Chat instantiated, DataHolder initialized
- User sends "classification report"
- Message is parsed and triggers the [classification_report()](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/api_list.py) function. (START API 1)
- The metadata is checked and it is found that the DataHolder needs "scores" and "predictions" dictionaries.
- The DataHolder is checked and does not contain this data. The user is notified that the model has not yet been fit.
- The metadata is referenced and the prerequisite API call is the [perform_classification()](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/api_list.py) function. A sub-API call begins. (START API 2)
- The metadata is checked and it is found that the DataHolder needs an "X" dataframe and "y" series.
- The DataHolder is checked and does not contain this data. The user is notified that the data has not yet been loaded.
- The metadata is checked and it is found that the prerequisite API call is the [load_csv_and_select_target()](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/api_list.py) function. Another sub-API call begins. (START API 3)
- The metadata is checked and there is no prerequsite data. The metadata is checked again and it is found that the "csv_path" parameter is needed from the user.
- The DataHolder is checked and does not contain this data, so the user is prompted for the "csv_path". The user enters the csv path which is added to the DataHolder.
- All requirements for API 3 are satisfied, so load_csv_and_select_target() is invoked and runs to completion. "X" dataframe and "y" series are added to the DataHolder as part of resolving API 3. (END API 3)
- The prerequisites are satisfied for API 2, so the metadata is checked again and it is found that the "cleaning_strictness", "hyperparameter_tuning", and "extended_models" parameters are needed from the user.
- The DataHolder is checked and does not contain this data, so the user is prompted for the "cleaning_strictness", "hyperparameter_tuning", and "extended_models" parameters. The user enters this data which is added to the DataHolder.
- All requirements for API 2 are satisfied, so perform_classification() is invoked and runs to completion. "scores" and "predictions" dictionaries are added to the DataHolder as part of resolving API 2 (END API 2)
- The prerequisites are satisfied for API 1, so the metadata is checked again and it is found that the "save_pdf" and "save_path" parameters are needed from the user.
- The DataHolder is checked and does not contain this data, so the user is prompted for the "save_pdf" and "save_path" parameters. The user enters this data which is added to the DataHolder.
- All requirements for API 1 are satisfied, so classification_report() is invoked and runs to completion. Nothing is added to the DataHolder as a part of this function. (END API 1)
- The user is alerted to the success of their API request, and the conversation loop restarts.

The user can invoke each API individually in order, if they wish, but the ModelBot is designed such that it can recursively invoke API calls as needed.

## Requirements

The libraries and version of Python used to create this project are listed below. The requirements are also available at [requirements.txt](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/requirements.txt).
```
Python==3.12.3

beautifulsoup4==4.13.3
ipython==8.12.3
ipython==8.20.0
Markdown==3.5.2
matplotlib==3.6.3
numpy==2.2.4
pandas==2.2.3
Pillow==11.2.1
reportlab==4.3.1
scikit_learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
statsmodels==0.14.4
together==1.5.5
```

## Repository Structure

```
├── Automatic-Machine-Learner.ipynb        # Notebook for testing and for programmatically creating metadata json
├── classification_report_example.pdf      # Example classification report pdf generated by ModelBot
├── classification_test.csv                # Data to use for testing classification tasks
├── regression_test.csv                    # Data to use for testing regression tasks
├── README.md
├── modules                                # Code used for for ModelBot
│   ├── api_list.py                        # List of API functions that can be invoked by the user
│   ├── chatbot.py                         # LlamaBot class that connects to Together API to access Llama 3-8b
│   ├── dataholder.py                      # Class that defines a DataHolder which holds and tracks saved parameters and other data
│   ├── evaluate.py                        # Class that performs model evaluation tasks
│   ├── helper.py                          # Helper functions for ModelBot, including metadata loading, intent parsing, and API invocation
│   ├── interface.py                       # ModelBot class which is the main interface between the user and the API functions
│   ├── metadata.json                      # JSON file containing invocable API function metadata
│   ├── model.py                           # Class that performs model fitting and scoring tasks
|   └── preprocess.py                      # Class that performs data preprocessing tasks
```

## To Do
- Clean up this repository, including expanding the README, renaming directories for clarity, and possible restructuring of repository
- Create a user journey flowchart and system design chart
- Add a function to save a regression report (regression counterpart to save_classification_report in [evaluate module](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/evaluate.py))
- Reformat the front-end to utilize Streamlit rather than using print statements
- Standup using FastAPI, including handling serialization of custom Python objects
- Clean up any code
