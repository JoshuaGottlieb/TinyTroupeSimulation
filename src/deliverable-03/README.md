# ModelBot, an Agentic helper for doing basic Machine Learning tasks

## Summary

ModelBot is an Agentic helper designed to perform regression and classification tasks from csv data and produce a PDF report for an end-user. ModelBot is designed to require minimal data science knowledge for operation and is powered by native Python libraries such as NumPy, Pandas, scikit-learn, Matplotlib, and Seaborn, with agents utilizing the Llama 3, 8 billion parameter model accessed from [Together.ai](https://www.together.ai/). As such, the ModelBot requires a Together API key to function.

The current version of ModelBot is limited in scope. A limited amount of preprocessing is performed on the dataset before modeling. Currently, the only models used for regression are Linear Regression, Polynomial (Degree 3) Regression using Lasso regularization, and ElasticNet Regression, and the only models used for classification are Logistic Regression, Decision Tree Classifiers, and Random Forest Classifiers. A small hyperparameter grid can be searched, if requested by the user, to perform limited model tuning; however, the grid is deliberately small to reduce fitting time. The expected performance of ModelBot is significantly lower than that of a human data scientist, as the preprocessing and modeling steps are relatively simplistic.

## Requirements

The libraries and version of Python used to create this project are listed below. The requirements are also available at [requirements.txt](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/requirements.txt).

```
Python==3.12.3

beautifulsoup4==4.13.4
fastapi==0.115.12
ipython==8.20.0
Markdown==3.5.2
matplotlib==3.6.3
numpy==2.2.5
opencv-python-headless==4.11.0.86
pandas==2.2.3
Pillow==11.2.1
pydantic==2.11.4
reportlab==4.3.1
Requests==2.32.3
scikit_learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
shap==0.47.1
together==1.5.5
uvicorn==0.34.2
```

The project is designed to be used with FastAPI using the Uvicorn package to set up a localhost server. ModelBot is powered by [Together.ai](https://api.together.ai/) and thus requires a Together API key. ModelBot is intended to be interacted with through the [ModelBot Chat Jupyter Notebook](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/ModelBot-Chat.ipynb). The current configuration expects the Together API key to be part of the environment variables, so before execution, the following command needs to be executed in the terminal window used to launch Jupyter Notebooks and again in the window used to launch the FastAPI session:

```
export TOGETHER_API_KEY="YOUR_API_KEY"
```

For the Jupyter Noteobook, an alternative is to add the following snippet to the notebook:
```
import os

os.environ["TOGETHER_API_KEY"] = "YOUR_API_KEY"
```

To set up the FastAPI server on localhost, at the project's root directory, execute the following command in a terminal. It is necessary to execute the command from the project's root directory in order for relative imports to function properly:

```

uvicorn model_bot.api:app --reload

```

Reminder: The Together API key must be exported in both terminals (one for setting up the FastAPI server and one for loading the Jupyter Notebook) or must otherwise be added to your permanent bashrc.

## Basic Workflow / User Journey Description

The ModelBot is designed to be used by users with little to no Data Science knowledge. The user can request descriptions of the possible API functions, their trigger words, and their requirements by entering "list functions".

- A chat is instantiated from a fresh [ModelBot](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/interface.py), creating a fresh [DataHolder object](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/dataholder.py) and loading the API function metadata from [metadata.json](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/metadata.json)
- The user interacts with the chatbot, sending messages.
- The messages are parsed and checked for trigger words to see if any APIs are being requested by the user using detect_api_call() in [helper.py](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/helper.py)
- If no API calls are detected, the chatbot responds to the user. Otherwise, the API call is processed using process_api_call() in [helper.py](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/helper.py)
- process_api_call() first checks the metadata to see what prerequisite data must be present in the DataHolder. If there is prerequisite data that is needed and that data is not in the DataHolder, process_api_call() is recursively called until an API call is successfully initiated or otherwise fails gracefully.
- If there is no prerequisite data, process_api_call() then checks for user input data for the API function in the DataHolder. Any data not in the DataHolder is requested from the user via inputs.
- Once all of the prerequisites and input data are present in the DataHolder, the [API function](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/modules/api_list.py) is invoked. Any data that needs to be saved is added to the DataHolder object for future API calls. Status messages are displayed back to the user in the event stream, and the function returns a response dictionary denoting success or failure of the API call. If the API call failed, user input gathered for the API function call is removed from the DataHolder so that the user can retry later.
- The success status of the API call is displayed to the user, and the chat continues.

A more detailed description of the system design and user journey is available at the [ModelBot System Design document](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-03/system_design/ModelBot-System-Design-and-User-Journey.pdf).

## Repository Structure

```
├── example_reports                                 # Example reports to showcase the capabilities of ModelBot
│   ├── classification_report_example.pdf
│   └── regression_report_example.pdf
├── model_bot                                       # Code used to power ModelBot
│   ├── api.py                                      # App script containing functions that are implemented via FastAPI to create reports
│   ├── dataholder.py                               # Class that defines the DataHolder object used to hold saved data during chats
│   ├── evaluate.py                                 # Class that performs model evaluation tasks
│   ├── functions.py                                # Script containing functions callable by the user during chat
│   ├── helper.py                                   # Script for metadata loading, intent processing, and dependency resolution for function calls
│   ├── llm.py                                      # LlamaBot class that connects to Together API to acces Llama 3-8b
│   ├── metadata                                    # Metadata used for defining user-callable functions
│   │   ├── Metadata-Creation.ipynb                 # Notebook for programmatically defining metadata
│   │   └── metadata.json                           # JSON file containing function metadata
│   ├── model.py                                    # Class that performs model fitting tasks
│   ├── preprocess.py                               # Class that performs data preprocessing tasks
│   ├── serialize.py                                # Script containing functions for serialization and deserialization of data for use in HTTP requests
│   └── user_interface.py                           # ModelBot class which is the main interface with he user
├── ModelBot-Chat.ipynb                             # Jupyter Notebook to use for running ModelBot
├── README.md
├── requirements.txt
├── sample_data                                     # Sample data to use for testing ModelBot
│   ├── classification_test.csv
│   └── regression_test.csv
└── system_design                                   # System design and user journey documentation
    └── ModelBot-System-Design-and-User-Journey.pdf

```

## Future Enhancements
- Reformat the front-end to utilize Streamlit or some other fancier front-end rather than utilzing print statements in Jupyter Notebooks
