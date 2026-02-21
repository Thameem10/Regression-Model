my-ml-project/
│
├── data/
│   ├── raw/ - Where Raw Dataset File is stored 
│   ├── processed/ - Where Processed Dataset File is stored 
│
├── notebooks/
│   └── exploration.ipynb - You can run a code cell by cell and you can view the dataset for demo also
│
├── src/
│   ├── data_preprocessing.py - Processing the dataset and stored in processed folder
│   ├── train.py - train the model 
│   ├── predict.py - predict for new data 
│   └── utils.py - helper function 
│
├── models/   - created model stored in this folder
│   └── model.pkl 
│
├── requirements.txt
├── README.md
└── venv/


// Create the venv

python3 -m venv env 

// Activate the venv

source venv/bin/activate

// install the packages

pip install -r requirements.txt

// Processing the data 

python3 data_preprocessing.py

// Train the model 

python3 train.py

// Predict using trained model 

python3 predict.py
