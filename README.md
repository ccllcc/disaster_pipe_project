# Disaster Response Pipeline Project

### Description
The disaster response pipeline project is a part of the Udacity Data Science nano degree.The original dataset contains pre-labelled messages from real life disaster. The objective of this project is to create a ML pipeline to classify the disaster messages and to deploy such a pipeline into a web application to display a few visualizations and allow the input of new messages and to classify them into 36 categories.

The project consists 3 main parts:
  - ETL pipeline
  - ML pipeline
  - Web app deployment

# Getting Started
### Dependancies

  > Python 3.5+ <br>
  > Statistical ML libraries: NumPy, Pandas, Sciki-Learn <br>
  > Natural language process library: NLTK <br>
  > Database library: SQLalchemy <br>
  > Web deployment and visualization: Flask, Plotly <br>

### File Structure
`data` folder contains the following:
- `disasters_messages.csv` original dataset provided by Udacity
- `disasters_categories.csv` orginal dataset provided by Udacity
- `process_data.py` python file to run in order to complete the ETL process, to generate a database file
- `DisasterResponse.db` the database file generated by `process_data.py`


`models` folder contains the following:
- `train_classifier.py` run this file to execute the ML pipeline process, generating a ML model and generating model
- `evaluate_model.jpg` is a snapshot of the model evaluation when running the `train_classifier.py`

`app` folder contains the following:
- `run.py` run the web app on localhost:3001
- `templates` folder contains two html files
  - `master.html` the index page of the web app
  - `go.html` handles the classification task to be displayed on the app
- `Viz1&2` `Viz3` `Viz4` four visualizations embedded in the web app.

# Executing Program
1. Clone this project to your local
> Git clone https://github.com/ccllcc/disaster_pipe_project
2. Run the ETL pipeline
> Please delete the `DisasterResponse.db` first as you are going to generate a new one during the ETL process. <br>
At the project root folder run: <br>
`python data/process_data.py data/disasters_messages.csv data/disasters_categories.csv data/DisasterResponse.db` 
3. Run the ML pipeline
> At the project root folder run: <br>
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
4. Run the app
> At the `app` folder run: <br>
`python run.py` <br>
Open the browser, in the address bar, go to `localhost:3001` the web should display the visualization.
Input a text message, the classification should render.

# Notes
### Running ML pipeline time
It took me more than 30 mins when training and fine-tuning the models using GridSearchCV with 6 combinations of parameters on my local machine based on RandomForestClassifier. It was more than 2 hours with K Nearest Neighbour classifier. 

### Acknowledgement
[Udacity](https://www.udacity.com/) for building up such an comprehensive learning project, covering ETL, ML, text processing, web deployment.<br>
[Figure Eight](https://appen.com/) for providing the original data.
