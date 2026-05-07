# Transformer


## Folder Structure

The project is organized as follows:

```
.
├── README.md
├── __init__.py
├── client
│   ├── __init__.py
│   ├── data_handler.py
│   └── transformation.py
├── custom_job
│   ├── __init__.py
│   └── custom_code.py
├── requirements.txt
└── setup.py
```

- **client**: Contains the core scripts for data handling and transformation.
    - data_handler.py: Utility script with helper functions to read and write datasets.
      - Class: `DataHandler`
      - Methods: `read(dataset_macro)`, `write(resultant_dataframe)`, `save_output(local_file_path)`
    - transformation.py: Script to trigger custom transformation logic.
      - Class: `Transformation`
      - Method: `transform()`
- **custom_job**: Contains the scripts for your transformations.
    - custom_code.py: Script where you implement your custom transformation logic.
- **requirements.txt**: Add your required libraries or dependencies
- **setup.py**: Contains code to generate the wheel file.

# Writing Your Code

Integrate your custom transformation logic in [custom_code.py](custom_job/custom_code.py) and ensure it is callable from [transformation.py](client/transformation.py).

or

Write your code in the [transformation.py](client/transformation.py) itself.

### Accessing DataFrames

Utilize the `DataHandler` object, initialized within the `Transformation` class, to read datasets:

```python
df = self.data_handler.read("<dataset_macro>")
```

### Read Runtime Parameters (OPTIONAL)

####  Use the run_params function of DataHandler class
```python
run_params = self.data_handler.run_params()
```
#### Get value and datatype of a parameter using below functions of DataHandler class
```python
param_value = self.data_handler.get_run_parameter_value("<param_name>")
param_datatype = self.data_handler.get_run_parameter_datatype("<param_name>")
```
#### Get column name from macro
```python
column_name = self.data_handler.get_column_for_macro("<table_macro>", "<column_macro>")
```

### Storing the Extra Outputs (OPTIONAL)

Ensure the additional output files are stored for further use by writing it through the DataHandler object:

```python
output_file_path = f"{os.environ['PWD']}/extra_output_data.txt"
extra_output_object = {'myoutput': 'extraoutput'} # this can be any object.
with open(output_file_path, 'w') as f:
  f.write(json.dumps(extra_output_object))
self.data_handler.save_output(output_file_path)
```

### Storing the Result (MANDATORY)

Ensure the resultant DataFrame is stored for further use by writing it through the DataHandler object:

```python
self.data_handler.write(resultant_dataframe)
```

This should be the final step of your application

# Package your application

### Prerequisites
Install the necessary dependencies to generate a wheel file:

```bash
pip install wheel setuptools
```

### Adding Dependencies

Specify any required libraries or dependencies in [requirements.txt](requirements.txt)

### Building the Wheel
Generate the wheel file by running:

```bash
python3 setup.py bdist_wheel
```
The wheel file will be created and placed in the `dist/` directory.



## Testing

Test your transformations locally with synthetic data before deployment:

```bash
# Navigate to test directory
cd test

# Test
python3 synthetic_validator.py
```

The testing framework automatically:
- Sets up virtual environment with required dependencies
- Generates realistic test data based on your schema
- Runs transformations locally with Spark
- Exports test data as CSV files for review

**For detailed testing documentation, configuration options, and troubleshooting, see [`tests/README.md`](tests/README.md).**
