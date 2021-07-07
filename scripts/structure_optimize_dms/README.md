```shell
# Get code
cd ${path_to_this_code}/structure_optimize_dms

# (Install virtualenv if necessary)
pip3 install virtualenv

# Create a virtual environment coined `env` for the project
virtualenv env

# Activate the environment
source env/bin/activate

# Install required packages
pip3 install -r requirements.txt

# Start notebook
jupyter notebook

# Deactivate when done working on this project to return to global settings
deactivate
```
