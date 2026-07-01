WITH campaign_products AS (
  SELECT
    COALESCE(MAX(CAST(campaign_id AS STRING)), CAST(120234240297670307 AS STRING)) AS campaign_id,
    CAST(product_id AS STRING) AS product_id
  FROM @campaign
  WHERE campaign_id_long = 120234240297670307
     OR campaign_id = '120234240297670307'
  GROUP BY CAST(product_id AS STRING)
),
latest_mapping AS (
  SELECT
    rampid_dpm,
    rampid_meta,
    rampid_rmn
  FROM (
    SELECT
      rampid_dpm,
      rampid_meta,
      rampid_rmn,
      ROW_NUMBER() OVER (
        PARTITION BY rampid_dpm, rampid_meta, rampid_rmn
        ORDER BY install_date DESC NULLS LAST
      ) AS rn
    FROM @mapping
    WHERE rampid_dpm IS NOT NULL
      AND rampid_rmn IS NOT NULL
  ) m
  WHERE rn = 1
),
latest_demographics AS (
  SELECT
    rampid,
    addressLink,
    Grouping_Indicator,
    hhpel
  FROM (
    SELECT
      rampid,
      addressLink,
      Grouping_Indicator,
      hhpel,
      ROW_NUMBER() OVER (
        PARTITION BY rampid
        ORDER BY install_date DESC NULLS LAST
      ) AS rn
    FROM @demographics
    WHERE rampid IS NOT NULL
  ) d
  WHERE rn = 1
),
eligible_conversion_identity_to_addresslink AS (
  SELECT DISTINCT
    m.rampid_rmn AS lr_id,
    m.rampid_meta AS online_identity_id,
    d.addressLink,
    d.hhpel,
    d.Grouping_Indicator
  FROM latest_mapping m
  INNER JOIN latest_demographics d
    ON m.rampid_dpm = d.rampid
  WHERE d.addressLink IS NOT NULL
    AND d.addressLink <> ''
    AND m.rampid_rmn IS NOT NULL
),
eligible_conversion_ids AS (
  SELECT
    lr_id
  FROM eligible_conversion_identity_to_addresslink
  GROUP BY lr_id
),
conversion_scoped AS (
  SELECT
    c.lr_id,
    c.transaction_category,
    c.order_id,
    c.product_brand,
    c.banner,
    c.division,
    CAST(c.transaction_timestamp_unix AS BIGINT) AS transaction_timestamp_unix,
    c.transaction_amount,
    c.quantity,
    CAST(c.product_id AS STRING) AS product_id,
    to_date(from_unixtime(CAST(c.transaction_timestamp_unix AS BIGINT))) AS transaction_date
  FROM @conversion c
  LEFT SEMI JOIN eligible_conversion_ids ei
    ON c.lr_id = ei.lr_id
  -- One timestamp window only: 2024-11-12 through 2026-01-06 inclusive.
  WHERE CAST(c.transaction_timestamp_unix AS BIGINT) >= unix_timestamp('2024-11-12', 'yyyy-MM-dd')
    AND CAST(c.transaction_timestamp_unix AS BIGINT) < unix_timestamp('2026-01-07', 'yyyy-MM-dd')
),
conversion_deduped AS (
  SELECT
    lr_id,
    transaction_category,
    order_id,
    product_brand,
    banner,
    division,
    transaction_timestamp_unix,
    transaction_date,
    transaction_amount,
    quantity,
    product_id
  FROM (
    SELECT
      lr_id,
      transaction_category,
      order_id,
      product_brand,
      banner,
      division,
      transaction_timestamp_unix,
      transaction_date,
      transaction_amount,
      quantity,
      product_id,
      ROW_NUMBER() OVER (
        PARTITION BY lr_id, order_id, transaction_timestamp_unix, product_id
        ORDER BY transaction_amount DESC NULLS LAST,
                 quantity DESC NULLS LAST,
                 transaction_category,
                 product_brand,
                 banner,
                 division
      ) AS rn
    FROM conversion_scoped
  ) c
  WHERE rn = 1
),
conversion_rows_mapped_to_addresslink AS (
  SELECT /*+ BROADCAST(cp) */
    eia.addressLink,
    eia.hhpel,
    eia.Grouping_Indicator,
    eia.online_identity_id,
    cp.campaign_id,
    c.order_id,
    c.product_id,
    c.product_brand,
    c.banner,
    c.division,
    c.transaction_category,
    c.transaction_timestamp_unix,
    c.transaction_date,
    CAST(c.transaction_amount AS DOUBLE) AS transaction_amount,
    CAST(c.quantity AS DOUBLE) AS quantity,
    CASE WHEN cp.product_id IS NOT NULL THEN 1 ELSE 0 END AS is_campaign_product
  FROM conversion_deduped c
  INNER JOIN eligible_conversion_identity_to_addresslink eia
    ON c.lr_id = eia.lr_id
  LEFT JOIN campaign_products cp
    ON c.product_id = cp.product_id
),
-- A person can have multiple lr_ids (multiple emails). If two lr_ids for the same
-- person both resolve to the same addressLink and share an order_id, the join above
-- produces duplicate transaction rows. Deduplicate at the addressLink-order-product
-- grain so FeatureEngg SUM aggregations are not inflated.
conversion_rows_deduped AS (
  SELECT
    addressLink,
    hhpel,
    Grouping_Indicator,
    online_identity_id,
    campaign_id,
    order_id,
    CAST(product_id AS BIGINT) AS product_id,
    product_brand,
    banner,
    division,
    transaction_category,
    transaction_timestamp_unix,
    transaction_date,
    transaction_amount,
    quantity,
    is_campaign_product,
    ROW_NUMBER() OVER (
      PARTITION BY addressLink, order_id, transaction_timestamp_unix, CAST(product_id AS BIGINT)
      ORDER BY transaction_amount DESC NULLS LAST, quantity DESC NULLS LAST
    ) AS rn
  FROM conversion_rows_mapped_to_addresslink
)
SELECT
  addressLink,
  hhpel,
  Grouping_Indicator,
  online_identity_id,
  campaign_id,
  order_id,
  product_id,
  product_brand,
  banner,
  division,
  transaction_category,
  transaction_timestamp_unix,
  transaction_date,
  transaction_amount,
  quantity,
  is_campaign_product
FROM conversion_rows_deduped
WHERE rn = 1;