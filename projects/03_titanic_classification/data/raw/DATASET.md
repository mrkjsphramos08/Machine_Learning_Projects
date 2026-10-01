# Dataset Provenance: Titanic Passenger Survival

- **Source URL**: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv` (Kaggle Titanic Competition)
- **Retrieved**: 2026-10-01
- **Licence**: CC0 Public Domain / Open Competition
- **Rows / Columns**: 891 rows / 12 columns
- **SHA256**: `4A437FDE05FE5264E1701A7387AC6FB75393772BA38BB2C9C566405AF5AF4BD7`
- **Target**: `Survived` (Binary: `1` = Survived, `0` = Did not survive)
- **Problem Type**: Supervised Binary Classification

## Features
| Feature | Type | Description |
| :--- | :--- | :--- |
| `PassengerId` | Integer | Unique identifier for each passenger (Leakage trap / drop) |
| `Survived` | Integer | Target variable (`0` = Died, `1` = Survived) |
| `Pclass` | Integer | Ticket class (`1` = 1st, `2` = 2nd, `3` = 3rd) |
| `Name` | String | Passenger name (contains titles like Mr., Mrs., Miss, Master) |
| `Sex` | String | Gender (`male`, `female`) |
| `Age` | Float | Age in years (has missing values) |
| `SibSp` | Integer | Number of siblings / spouses aboard the Titanic |
| `Parch` | Integer | Number of parents / children aboard the Titanic |
| `Ticket` | String | Ticket number |
| `Fare` | Float | Passenger fare paid |
| `Cabin` | String | Cabin number (has heavy missing values) |
| `Embarked` | String | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |
