# Source Schema Reference

Use this with `eda_m_grocery_campaign.md`. Shared placeholder replacements are documented in `README.md`.

## Tables

### `@campaign`: Campaign ID to Brand Name Mapping - Flight Dates

Owned by Albertson - HQ. Join on `product_id` for conversion, not `<rrrr_id>`.

Columns:

- `campaign_id` STRING
- `product_brand` STRING
- `product_manufacturer` STRING
- `product_id` STRING
- `flight_start` DATE
- `flight_end` DATE
- `campaign_id_long` LONG
- `publisher_name` STRING

### `@conversion`: Conversion Events - Incrementality - Generic

Owned by Albertson - HQ.

Columns:

- `lr_id` STRING
- `transaction_category` STRING
- `order_id` STRING
- `product_brand` STRING
- `banner` STRING
- `division` STRING
- `transaction_timestamp_unix` INTEGER
- `transaction_amount` DECIMAL
- `quantity` INTEGER
- `product_id` INTEGER

### `@exposure`: meta Production - GCS - Generic

Owned by meta Prod.

Columns:

- `ts` LONG
- `tp_id` STRING
- `experiment_id` STRING
- `ad_id` LONG
- `adset_id` LONG
- `campaign_id` LONG
- `account_id` LONG
- `event_type` STRING
- `placement_type` STRING
- `device_platform` STRING
- `impression_device` STRING

### `@demographics`: Sample - Insights Graph - IDENTITY_GRAPH

Owned by /L Measurement Services - meta and Albertsons.

Columns:

- `rampid` STRING: online identity key from the source identity graph, not a one-person individual ID. One person can have multiple `ramp` IDs.
- `Grouping_Indicator` STRING: person-level record grouping key in `@demographics`; resolves/dedupes multiple `ramp` online identity IDs that belong to the same source person-level record
- `hhpel` STRING: household-level ID in `@demographics`
- `addressLink` STRING: physical/address-level linkage field; not the household ID
- `gender` STRING
- `age` INTEGER
- `state` INTEGER
- `county` INTEGER
- `hh_income` INTEGER: coded household-income value. Treat this as a source code, not a dollar amount. Model-ready one-hot/count/share income features must be created in feature engineering, not preprocessing.
- `poc` BOOLEAN: true when the household has a child, false otherwise
- `install_date` DATE

### `@mapping`: Transcoding Mapping - M1 App V1 - IDENTITY_GRAPH

Owned by /L Measurement Services - meta and Albertsons.

Columns:

- `install_date` DATE
- `rampid_dpm` STRING: maps to `@demographics.rampid`; online identity domain for demographic joins
- `rampid_meta` STRING: maps to `@exposure.tp_id`; online identity domain for exposure joins
- `rampid_rmn` STRING: maps to `@conversion.lr_id`; online identity domain for conversion joins
