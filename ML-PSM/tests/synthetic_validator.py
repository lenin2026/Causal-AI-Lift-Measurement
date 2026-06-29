#!/usr/bin/env python3
"""
Simple Spark test runner - no fancy classes, just basic functions.
"""

import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path


def setup_venv():
    """Setup virtual environment if needed."""
    venv_path = Path(__file__).parent / ".venv"
    
    if not venv_path.exists():
        print(f"Creating virtual environment...")
        import venv
        venv.create(venv_path, with_pip=True)
        
        pip_path = venv_path / "bin" / "pip"
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        # Install test requirements
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            print(f"Installing test requirements from {requirements_file}")
            subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        
        # Install template requirements
        template_requirements = Path(__file__).parent.parent / "requirements.txt"
        if template_requirements.exists():
            print(f"Installing template requirements from {template_requirements}")
            subprocess.run([str(pip_path), "install", "-r", str(template_requirements)], check=True)
    
    # Restart in venv if not already
    if not os.environ.get("VENV_RERUN"):
        print("Restarting in virtual environment...")
        venv_python = venv_path / "bin" / "python"
        env = os.environ.copy()
        env["VENV_RERUN"] = "1"
        env["PYTHONPATH"] = f"{Path(__file__).parent.parent}:{env.get('PYTHONPATH', '')}"
        
        result = subprocess.run([str(venv_python)] + sys.argv, env=env)
        sys.exit(result.returncode)


def load_config():
    """Load YAML configuration."""
    import yaml
    config_file = Path(__file__).parent / "config" / "test_data_config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix None values
    if config.get('datasets') is None:
        config['datasets'] = {}
    if config.get('dataset_types') is None:
        config['dataset_types'] = {}
    
    return config


def create_spark_session():
    """Create Spark session."""
    from pyspark.sql import SparkSession
    
    return SparkSession.builder \
        .appName("SyntheticTest") \
        .master("local[2]") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()


def generate_test_data(config, use_existing=True):
    """Generate or load test data from CSV files if they exist."""
    from faker import Faker
    import random
    import csv

    fake = Faker()
    test_data = {}
    data_dir = Path(__file__).parent / "data"

    # Merge datasets and dataset_types from config
    datasets = {}
    if config.get('datasets'):
        datasets.update(config.get('datasets'))
    if config.get('dataset_types'):
        datasets.update(config.get('dataset_types'))

    for dataset_name, dataset_config in datasets.items():
        schema = dataset_config.get('schema', [])
        types = dataset_config.get('types', [])
        num_rows = dataset_config.get('num_rows', 5)
        
        rows = []
        csv_file = data_dir / f"{dataset_name}.csv"
        print("Generating Data for:", csv_file, f"({num_rows} rows)")
        if use_existing and csv_file.exists():
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = [row for row in reader]
                print("Using Existing Data:", dataset_name, f"({len(rows)} rows)")
        else:
            for i in range(num_rows):
                row = []
                for j, (col_name, col_type) in enumerate(zip(schema, types)):
                    # Enhanced data generation with full type support
                    col_type_lower = col_type.lower()

                    if col_type_lower in ["integer", "int"]:
                        if j == 0:  # First column is ID
                            value = str(i + 1)
                        else:
                            value = str(random.randint(1, 100))

                    elif col_type_lower == "long":
                        if j == 0:  # First column is ID
                            value = str(i + 1)
                        else:
                            value = str(random.randint(1000000, 9999999999))

                    elif col_type_lower in ["double", "float"]:
                        value = str(round(random.uniform(10.0, 1000.0), 2))

                    elif col_type_lower == "decimal":
                        # Generate decimal with 2 decimal places
                        value = str(round(random.uniform(10.0, 1000.0), 2))

                    elif col_type_lower == "boolean":
                        value = str(random.choice([True, False])).lower()

                    elif col_type_lower == "string":
                        if "name" in col_name.lower():
                            value = fake.name()
                        elif "email" in col_name.lower():
                            value = fake.email()
                        elif "phone" in col_name.lower():
                            value = fake.phone_number()
                        elif "address" in col_name.lower():
                            value = fake.address().replace('\n', ', ')
                        elif "product" in col_name.lower():
                            value = fake.word().title() + " " + fake.word().title()
                        elif "status" in col_name.lower():
                            value = random.choice(["active", "inactive", "pending", "completed", "cancelled"])
                        else:
                            value = fake.word()

                    elif col_type_lower == "date":
                        if "signup" in col_name.lower() or "created" in col_name.lower():
                            # Earlier dates for signup/creation
                            value = fake.date_between(start_date='-2y', end_date='-1m').strftime('%Y-%m-%d')
                        elif "order" in col_name.lower() or "purchase" in col_name.lower():
                            # Recent dates for orders/purchases
                            value = fake.date_between(start_date='-6m', end_date='today').strftime('%Y-%m-%d')
                        else:
                            value = fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d')

                    elif col_type_lower == "timestamp":
                        if "signup" in col_name.lower() or "created" in col_name.lower():
                            # Earlier timestamps for signup/creation
                            value = fake.date_time_between(start_date='-2y', end_date='-1m').strftime('%Y-%m-%d %H:%M:%S')
                        elif "order" in col_name.lower() or "purchase" in col_name.lower():
                            # Recent timestamps for orders/purchases
                            value = fake.date_time_between(start_date='-6m', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            value = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')

                    else:
                        # Fallback for unknown types
                        value = fake.word()
                
                row.append(value)
            rows.append(row)
        
        test_data[dataset_name] = {
            'schema': schema,
            'data': rows
        }

    return test_data


def save_test_data_to_csv(test_data):
    """Save test data to CSV files in data folder for review."""
    import csv
    import shutil
    
    data_dir = Path(__file__).parent / "data"
    
    if data_dir.exists():
        shutil.rmtree(data_dir)
    
    data_dir.mkdir(exist_ok=True)
    
    for table_name, table_info in test_data.items():
        schema = table_info['schema']
        data = table_info['data']
        
        csv_file = data_dir / f"{table_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(schema)
            writer.writerows(data)
        
        print(f"Saved {table_name} data to {csv_file} ({len(data)} rows)")


def create_test_tables(spark, test_data):
    """Create Spark tables from test data."""
    for table_name, table_info in test_data.items():
        schema = table_info['schema']
        data = table_info['data']
        
        df = spark.createDataFrame(data, schema)
        df.createOrReplaceTempView(table_name)
        
        print(f"Created table {table_name} with {len(data)} rows")


def run_transformation(config):
    """Run the transformation."""
    # Add template to path
    template_path = Path(__file__).parent.parent
    sys.path.insert(0, str(template_path))
    
    # Import transformation
    from client.transformation import Transformation
    
    # Setup environment
    os.environ['PWD'] = tempfile.mkdtemp()
    os.environ['OUTPUT_EXTRA_PATH'] = tempfile.mkdtemp()
    
    # Get runtime parameters
    run_params = {}
    for param_name, param_config in config.get('run_parameters', {}).items():
        if isinstance(param_config, dict):
            value = param_config.get('value')
            param_type = param_config.get('type', 'string')
        else:
            value = param_config
            param_type = 'string'
        run_params[param_name] = [value, param_type]
    
    # Collect column macro mappings from dataset_types
    macro_to_columns = {}
    dataset_types = config.get('dataset_types', {})
    for dataset_name, dataset_config in dataset_types.items():
        column_macro_mapping = dataset_config.get('column_macro_mapping', {})
        if column_macro_mapping:
            macro_to_columns[dataset_name] = column_macro_mapping
            print(f"Added column macro mapping for {dataset_name}: {column_macro_mapping}")
    
    # Get has_extra_outputs from config
    has_extra_outputs = config.get('has_extra_outputs', True)
    
    # Create transformation parameters
    params = {
        'spark': None,
        'has_extra_outputs': has_extra_outputs,
        'custom_packages_path': '',
        'run_parameters': run_params,
        'macro_to_columns': macro_to_columns
    }
    
    return Transformation, params


def main():
    """Main function."""
    # Setup virtual environment
    if not os.environ.get("VENV_RERUN"):
        setup_venv()
    
    try:
        # Load configuration
        config = load_config()
        print("Running synthetic validator...")
        
        # Create Spark session
        spark = create_spark_session()
        
        # Generate and create test data
        test_data = generate_test_data(config)
        save_test_data_to_csv(test_data)
        create_test_tables(spark, test_data)
        
        # Run transformation
        TransformationClass, params = run_transformation(config)
        params['spark'] = spark
        
        transformation = TransformationClass(**params)
        transformation.transform()
        
        # Check results
        result_df = spark.sql("SELECT * FROM result")
        count = result_df.count()
        print(f"Test passed: {count} result rows")
        
        # Cleanup
        spark.stop()
        time.sleep(1)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
