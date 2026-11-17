# Data dictionary

All data is synthetic, produced by `timefence generate` (default: 10,000 customers, seed 7). Timestamps
are South African Standard Time (UTC+2) and stored without a time zone. Every warehouse load lands at
02:00, and credit decisions are made between 08:00 and 19:59.

## Sources used by the feature store

Each source has a valid-time column (when the fact was true) and `ingested_at` (when this version first
landed). `record_version` breaks ties between versions that land in the same load.

### `customers`
| Column | Type | Notes |
|---|---|---|
| `customer_id` | text | Record key. `C000000` to `C009999`. |
| `onboarded_at` | timestamp | **Valid time.** 2016-01 to 2023-06. |
| `income_band` | int | 1 (lowest) to 5. Correlated with latent risk. |
| `age_band` | text | Not used as a feature. |
| `province` | text | Analytics only. Blocked for credit decisioning as a potential proxy. |
| `ingested_at` | timestamp | 02:00 on the day after onboarding. |

### `account_snapshots`: one row per account per month-end, plus restatements
| Column | Type | Notes |
|---|---|---|
| `account_id`, `snapshot_at` | text, timestamp | Record key. `snapshot_at` is month-end 23:59:59 (**valid time**). |
| `customer_id`, `product` | text | `card` or `personal_loan`. |
| `balance`, `credit_limit` | float | Rand. For loans, `credit_limit` is the original principal. |
| `dpd` | int | Days past due, in 30-day buckets up to 180. |
| `ingested_at` | timestamp | v1 lands at 02:00 on the 1st. About 15% of months are delayed 1-5 days by a late batch. |
| `record_version` | int | v2 is a restatement. It is issued when a debit-order reversal lands after v1 and carries the true DPD. |

### `payments`
| Column | Type | Notes |
|---|---|---|
| `payment_id` | text | Record key. |
| `paid_at` | timestamp | **Valid time.** Days 20-28 of the month, 06:00-21:59. |
| `amount_due`, `amount_paid` | float | Card minimum due is 5% of balance (minimum R200). Loans: fixed instalment. |
| `status` | text | `cleared` (v1, lands the next night) or `reversed` (v2, lands 3-10 days after payment). |

### `bureau_enquiries`: credit enquiries at other lenders
| Column | Type | Notes |
|---|---|---|
| `enquiry_id` | text | Record key. |
| `enquired_at` | timestamp | **Valid time.** |
| `lender_type` | text | `bank`, `retailer` or `micro_lender`. Micro-lender share rises with financial stress. |
| `ingested_at` | timestamp | Monthly bureau file: 02:00 on the 12th of the following month. 8% slip to the next file. |

### `collections`
| Column | Type | Notes |
|---|---|---|
| `referral_id` | text | Record key. Raised when an account moves from 30 to 60 DPD. |
| `effective_at` | timestamp | **Valid time, back-dated** to the month-end when the arrears began. |
| `ingested_at` | timestamp | 4-15 days after the month-end at which 60 DPD was reached. |

## Modelling tables

### `applications` (the spine)
`application_id`, `customer_id`, `decision_time`, `product`, `requested_amount`. Applications run from
2022-01 to 2024-12. Customers become eligible 90 days after onboarding, and stressed customers apply
more often.

### `labels`
| Column | Notes |
|---|---|
| `bad` | 1 if any of the customer's accounts is at 90+ DPD at any of the 12 month-ends starting with the decision month. |
| `label_matured_at` | The last of those month-ends. `attach_labels` refuses labels that mature after the training cut-off. |

### `accounts`
Reference table (`account_id`, `customer_id`, `product`, `opened_at`, `credit_limit`). The feature
store does not read it.

## The latent process (for reproducibility, not for modelling)

Each customer has a persistent risk score `z ~ N(0, 1)` and a transient stress term `s` that follows an
AR(1) process (`s = 0.8 s + N(0, 0.55)`). A macro term ramps from mid-2022, peaks through 2023 and eases
through 2025. Monthly risk is `0.9 z + s + macro`, and it drives missed payments, reversals, utilisation,
enquiries, micro-lender share and application rates. The modeller sees none of these latent variables.
