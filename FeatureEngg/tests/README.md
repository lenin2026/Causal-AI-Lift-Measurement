# Synthetic Validator

## Quick Start

```bash
# Run the simple test
python3 synthetic_validator.py
```
The test will:
1. Set up a virtual environment automatically
2. Install required packages
3. Load test configuration
4. Generate test data and save to `data/` folder for review
5. Run your transformation
6. Show results

## Files

- `synthetic_validator.py` - Main test runner
- `config/test_data_config.yaml` - Test configuration
- `requirements.txt` - Test-specific packages (Don't change)

## Configuration

Edit `config/test_data_config.yaml` to:
- Define datasets and schemas/data type
- Define dataset_types schemas/data type/column macro mappings
- Set runtime parameters
- Configure `has_extra_outputs` (true/false) to enable file output

### Supported Data Types

The test runner supports the following data types with intelligent data generation:

| Type | Description | Example Generated Data |
|------|-------------|----------------------|
| `boolean` | True/false values | `true`, `false` |
| `date` | Date values (YYYY-MM-DD) | `2023-05-15` |
| `decimal` | Decimal numbers with 2 decimal places | `123.45` |
| `double` | Double precision floating point | `456.78` |
| `float` | Floating point numbers | `789.12` |
| `integer` / `int` | Whole numbers | `42` |
| `long` | Large integers | `1234567890` |
| `string` | Text values | Context-aware (names, emails, etc.) |
| `timestamp` | Date and time (YYYY-MM-DD HH:MM:SS) | `2023-05-15 14:30:25` |

### Column Macro Mapping 

You can define `column_macro_mapping` in the `dataset_types` section. This mapping will be automatically passed to the transformation as `macro_to_columns` parameter.

Example configuration:
```yaml

dataset_types:
  owner:
    schema: ["ID", "CUSTOMER_NAME", "EMAIL"]
    types: ["integer", "string", "string"]
    column_macro_mapping:
      ID: "ID"
      CUSTOMER_NAME: "CUSTOMER_NAME"
      EMAIL: "EMAIL"
    num_rows: 10
```

The `column_macro_mapping` from all dataset_types will be collected and passed to the transformation constructor as the `macro_to_columns` parameter, allowing your transformation to use the DataHandler's `get_column_for_macro()` function.

### Configuration Structure

Basic configuration structure:
```yaml
has_extra_outputs: true

datasets:
  # Your datasets here

dataset_types:
  # Your dataset types with column_macro_mapping here

run_parameters:
  # Your runtime parameters here
```

## How It Works

1. **Virtual Environment**: Automatically creates `.venv` if needed
2. **Test Data**: Generates fake data based on column names and types
3. **Data Export**: Saves test data as CSV files in `data/` folder (cleaned up each run)
4. **Spark Tables**: Creates temporary tables from test data
5. **Transformation**: Runs your transformation code
6. **Results**: Shows count of result rows


## Troubleshooting

If something breaks:
1. Delete the `.venv` folder
2. Run the test again
3. Check the error message
