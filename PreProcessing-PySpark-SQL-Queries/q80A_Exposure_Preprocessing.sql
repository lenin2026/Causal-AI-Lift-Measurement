WITH latest_mapping AS (
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
  ) d
  WHERE rn = 1
),
eligible_mapped_online_identity_to_addresslink AS (
  SELECT DISTINCT
    m.rampid_meta AS online_identity_id,
    d.addressLink,
    d.Grouping_Indicator,
    d.hhpel
  FROM latest_mapping m
  INNER JOIN latest_demographics d
    ON m.rampid_dpm = d.rampid
  WHERE d.addressLink IS NOT NULL
    AND d.addressLink <> ''
    AND m.rampid_meta IS NOT NULL
),
exposure_deduped AS (
  SELECT
    tp_id,
    ts,
    experiment_id,
    campaign_id,
    ad_id,
    adset_id,
    account_id,
    event_type,
    placement_type,
    device_platform,
    impression_device
  FROM (
    SELECT
      tp_id,
      ts,
      experiment_id,
      campaign_id,
      ad_id,
      adset_id,
      account_id,
      event_type,
      placement_type,
      device_platform,
      impression_device,
      ROW_NUMBER() OVER (
        PARTITION BY tp_id, ts, experiment_id, campaign_id, ad_id, adset_id, account_id, event_type, placement_type, device_platform, impression_device
        ORDER BY ts DESC
      ) AS rn
    FROM @exposure
    WHERE campaign_id = 120234240297670307
      AND to_date(from_unixtime(CAST(ts AS BIGINT)))
        BETWEEN to_date('2025-11-12') AND to_date('2026-01-06')
  ) e
  WHERE rn = 1
),
campaign_exposed_online_identities AS (
  SELECT
    tp_id,
    MIN(ts) AS min_exposure_ts,
    COUNT(*) AS exposure_frequency_deduped
  FROM exposure_deduped
  GROUP BY tp_id
),
eligible_online_identity_with_exposure_flag AS (
  SELECT
    em.online_identity_id,
    em.addressLink,
    em.Grouping_Indicator,
    em.hhpel,
    ce.min_exposure_ts,
    ce.exposure_frequency_deduped,
    CASE WHEN ce.tp_id IS NOT NULL THEN 1 ELSE 0 END AS online_identity_exposed
  FROM eligible_mapped_online_identity_to_addresslink em
  LEFT JOIN campaign_exposed_online_identities ce
    ON em.online_identity_id = ce.tp_id
),
addresslink_treatment_assignment_after_any_online_identity_exposure_rollup AS (
  SELECT
    addressLink,
    MAX(online_identity_exposed) AS treatment,
    MIN(CASE WHEN online_identity_exposed = 1 THEN min_exposure_ts END) AS min_exposure_ts,
    SUM(CASE WHEN online_identity_exposed = 1 THEN exposure_frequency_deduped ELSE 0 END) AS exposure_frequency_deduped,
    COUNT(DISTINCT online_identity_id) AS mapped_online_identity_count,
    COUNT(DISTINCT CASE WHEN online_identity_exposed = 1 THEN online_identity_id END) AS exposed_online_identity_count,
    COUNT(DISTINCT Grouping_Indicator) AS person_record_count,
    COUNT(DISTINCT hhpel) AS hhpel_count,
    CASE
      WHEN COUNT(DISTINCT CASE WHEN online_identity_exposed = 1 THEN online_identity_id END) > 0
       AND COUNT(DISTINCT CASE WHEN online_identity_exposed = 0 THEN online_identity_id END) > 0
      THEN 1 ELSE 0
    END AS has_partial_exposure_within_addresslink
  FROM eligible_online_identity_with_exposure_flag
  GROUP BY addressLink
),
addresslink_assignment_with_control_flag AS (
  SELECT
    addressLink,
    treatment,
    CASE WHEN treatment = 0 THEN 1 ELSE 0 END AS is_eligible_control,
    min_exposure_ts,
    exposure_frequency_deduped,
    mapped_online_identity_count,
    exposed_online_identity_count,
    person_record_count,
    hhpel_count,
    has_partial_exposure_within_addresslink
  FROM addresslink_treatment_assignment_after_any_online_identity_exposure_rollup
)
SELECT
  addressLink,
  treatment,
  is_eligible_control,
  min_exposure_ts,
  exposure_frequency_deduped,
  mapped_online_identity_count,
  exposed_online_identity_count,
  person_record_count,
  hhpel_count,
  has_partial_exposure_within_addresslink
FROM addresslink_assignment_with_control_flag
ORDER BY treatment DESC, addressLink;