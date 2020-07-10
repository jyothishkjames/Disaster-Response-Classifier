# Disaster-Response-Classifier

Getting Started
---------------

The Disaster Response Classifier project hosts a Flask web application which 
aids an emergency worker to classify disaster messages into several categories. 

Prerequisites
-------------
The following libraries are used for the project:

        pandas
        SQLAlchemy
        nltk
        scikit-learn
        plotly
        flask

Extract Transform Load Pipeline
-------------------------------

To execute the ETL pipeline, go to the folder **data** and 
follow the below command.

    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

   
Machine learning Pipeline
-------------------------

To execute the Machine learning pipeline, go to the folder **models**
and follow the below command. 

    python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Running the Server
------------------

Go to the folder **app** and run the file run.py

	python run.py


Classification Report
---------------------

The classification report for the multiclass classifier can be obtained 
by running the machine learning pipeline.


Github link
-----------

https://github.com/jyothishkjames/Disaster-Response-Classifier
    

    