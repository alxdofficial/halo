# Legacy V1 application evaluation

These tables are the frozen V1 pilot and do not describe the current Task-1 V2 protocol or its
rebuilt synthetic corpus. Do not combine these numbers with V2 results. Current V2 results belong
in a separately versioned report after its controls and common-unit evaluation are complete.

Status: complete deterministic evaluation, but provisional for publication. Subject-level
confidence intervals and raw-signal or physical-feature controls are not yet included.

All thresholds were selected once on development data. Each table reports the
complete sealed test split for one dataset and one readout; results are not pooled
across datasets.

For Task 3, B-cubed scores are conditional on matched predicted events. They must be
interpreted together with occurrence precision, occurrence recall, and false occurrences
per hour; they are not an end-to-end recurrence-discovery score on their own.

## TASK1 - c_mhad - direct_dtw

| encoder | event_precision | event_recall | event_f1 | false_alarms_per_hour | mean_onset_error_sec | mean_offset_error_sec | mean_absolute_count_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / direct_dtw | 0.0665 | 0.2661 | 0.1063 | 158.1125 | 0.4499 | 0.4648 | 5.4666 |
| harnet / direct_dtw | 0.0988 | 0.0451 | 0.0619 | 17.4194 | 0.4462 | 0.4476 | 1.4702 |
| normwear / direct_dtw | 0.0171 | 0.3864 | 0.0327 | 940.7992 | 0.5268 | 0.4907 | 30.3270 |
| unimts / direct_dtw | 0.1369 | 0.0892 | 0.1080 | 23.7816 | 0.4648 | 0.4748 | 1.4913 |

## TASK1 - c_mhad - learned_metric_dtw

| encoder | event_precision | event_recall | event_f1 | false_alarms_per_hour | mean_onset_error_sec | mean_offset_error_sec | mean_absolute_count_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / learned_metric_dtw | 0.0706 | 0.3465 | 0.1173 | 192.8196 | 0.4207 | 0.4422 | 6.2682 |
| harnet / learned_metric_dtw | 0.0828 | 0.1307 | 0.1014 | 61.2530 | 0.5076 | 0.4145 | 2.3227 |
| normwear / learned_metric_dtw | 0.0228 | 0.2469 | 0.0417 | 448.5596 | 0.5478 | 0.4697 | 13.9448 |
| unimts / learned_metric_dtw | 0.0451 | 0.2547 | 0.0766 | 228.1410 | 0.4949 | 0.5241 | 7.1744 |

## TASK1 - oca - direct_dtw

| encoder | event_precision | event_recall | event_f1 | false_alarms_per_hour | mean_onset_error_sec | mean_offset_error_sec | mean_absolute_count_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / direct_dtw | 0.0557 | 0.2918 | 0.0935 | 322.5411 | 1.3598 | 1.3169 | 154.3627 |
| harnet / direct_dtw | 0.1058 | 0.0044 | 0.0084 | 2.4132 | 0.6623 | 1.0009 | 29.1438 |
| normwear / direct_dtw | 0.0383 | 0.5568 | 0.0718 | 910.2718 | 1.4574 | 1.4929 | 404.6275 |
| unimts / direct_dtw | 0.2804 | 0.0058 | 0.0114 | 0.9710 | 0.5030 | 0.5397 | 29.2516 |

## TASK1 - oca - learned_metric_dtw

| encoder | event_precision | event_recall | event_f1 | false_alarms_per_hour | mean_onset_error_sec | mean_offset_error_sec | mean_absolute_count_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / learned_metric_dtw | 0.0588 | 0.2504 | 0.0953 | 261.0629 | 1.4247 | 1.3924 | 126.5294 |
| harnet / learned_metric_dtw | 0.0602 | 0.0324 | 0.0422 | 32.9631 | 0.6071 | 0.8204 | 32.0359 |
| normwear / learned_metric_dtw | 0.0462 | 0.3673 | 0.0821 | 493.9105 | 1.0934 | 1.0829 | 215.7549 |
| unimts / learned_metric_dtw | 0.1171 | 0.2642 | 0.1623 | 129.7961 | 1.6333 | 1.1534 | 51.2026 |

## TASK1 - wear - direct_dtw

| encoder | event_precision | event_recall | event_f1 | false_alarms_per_hour | mean_onset_error_sec | mean_offset_error_sec | mean_absolute_count_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / direct_dtw | 0.0435 | 0.1098 | 0.0623 | 5.0330 | 7.5136 | 6.3772 | 4.7617 |
| harnet / direct_dtw | 0.0189 | 0.0014 | 0.0026 | 0.1526 | 28.9400 | 8.6800 | 1.7265 |
| normwear / direct_dtw | 0.0097 | 0.1738 | 0.0184 | 36.8874 | 10.2164 | 10.8281 | 29.2829 |
| unimts / direct_dtw | 0.1716 | 0.0781 | 0.1074 | 0.7865 | 7.2386 | 6.5884 | 1.7840 |

## TASK1 - wear - learned_metric_dtw

| encoder | event_precision | event_recall | event_f1 | false_alarms_per_hour | mean_onset_error_sec | mean_offset_error_sec | mean_absolute_count_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / learned_metric_dtw | 0.0558 | 0.0943 | 0.0701 | 3.3294 | 6.3839 | 6.6999 | 3.5728 |
| harnet / learned_metric_dtw | 0.0806 | 0.1323 | 0.1002 | 3.1460 | 8.5348 | 8.8750 | 3.2735 |
| normwear / learned_metric_dtw | 0.0249 | 0.1084 | 0.0405 | 8.8495 | 7.2136 | 7.0005 | 7.4742 |
| unimts / learned_metric_dtw | 0.1022 | 0.2730 | 0.1487 | 5.0036 | 6.7483 | 6.6142 | 4.0646 |

## TASK3 - c_mhad - direct_cosine_recurrence

| encoder | occurrence_precision | occurrence_recall | bcubed_f1 | mean_fragments_per_true_motif | false_occurrences_per_hour | mean_absolute_count_error | matched_mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / direct_cosine_recurrence | 0.0368 | 0.8730 | 0.5880 | 1.1371 | 5854.9646 | 192.8750 | 0.7223 |
| harnet / direct_cosine_recurrence | 0.0359 | 0.7283 | 0.6941 | 1.2793 | 5017.8942 | 163.9167 | 0.7043 |
| normwear / direct_cosine_recurrence | 0.0349 | 0.8843 | 0.5671 | 1.2457 | 6279.6001 | 207.0375 | 0.7265 |
| unimts / direct_cosine_recurrence | 0.0372 | 0.8117 | 0.7520 | 1.3284 | 5383.9159 | 176.7500 | 0.7210 |

## TASK3 - c_mhad - learned_metric_recurrence

| encoder | occurrence_precision | occurrence_recall | bcubed_f1 | mean_fragments_per_true_motif | false_occurrences_per_hour | mean_absolute_count_error | matched_mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / learned_metric_recurrence | 0.0362 | 0.8784 | 0.6162 | 1.1437 | 6000.7444 | 197.7500 | 0.7233 |
| harnet / learned_metric_recurrence | 0.0346 | 0.7185 | 0.7521 | 1.3028 | 5138.5178 | 167.8292 | 0.7034 |
| normwear / learned_metric_recurrence | 0.0343 | 0.9063 | 0.5391 | 1.2228 | 6539.7145 | 215.8417 | 0.7319 |
| unimts / learned_metric_recurrence | 0.0365 | 0.8146 | 0.6889 | 1.2879 | 5507.6841 | 180.8750 | 0.7193 |

## TASK3 - oca - direct_cosine_recurrence

| encoder | occurrence_precision | occurrence_recall | bcubed_f1 | mean_fragments_per_true_motif | false_occurrences_per_hour | mean_absolute_count_error | matched_mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / direct_cosine_recurrence | 0.0531 | 0.9009 | 0.4206 | 12.2946 | 6327.6845 | 2881.2115 | 0.7332 |
| harnet / direct_cosine_recurrence | 0.0613 | 0.7935 | 0.2286 | 17.4951 | 4785.1771 | 2155.1346 | 0.7211 |
| normwear / direct_cosine_recurrence | 0.0540 | 0.9150 | 0.4606 | 9.0796 | 6306.6555 | 2874.1154 | 0.7374 |
| unimts / direct_cosine_recurrence | 0.0613 | 0.8301 | 0.3827 | 13.1230 | 5005.6252 | 2262.7308 | 0.7232 |

## TASK3 - oca - learned_metric_recurrence

| encoder | occurrence_precision | occurrence_recall | bcubed_f1 | mean_fragments_per_true_motif | false_occurrences_per_hour | mean_absolute_count_error | matched_mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / learned_metric_recurrence | 0.0522 | 0.9082 | 0.4589 | 10.8135 | 6484.1641 | 2954.2115 | 0.7321 |
| harnet / learned_metric_recurrence | 0.0575 | 0.8523 | 0.3950 | 13.2862 | 5493.1128 | 2490.0769 | 0.7239 |
| normwear / learned_metric_recurrence | 0.0551 | 0.9635 | 0.3990 | 3.1946 | 6505.8227 | 2974.1154 | 0.7376 |
| unimts / learned_metric_recurrence | 0.0589 | 0.9022 | 0.4711 | 8.0091 | 5669.4461 | 2579.8654 | 0.7326 |

## TASK3 - wear - direct_cosine_recurrence

| encoder | occurrence_precision | occurrence_recall | bcubed_f1 | mean_fragments_per_true_motif | false_occurrences_per_hour | mean_absolute_count_error | matched_mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / direct_cosine_recurrence | 0.0032 | 0.5529 | 0.6729 | 2.1750 | 6431.0120 | 5146.8958 | 0.7538 |
| harnet / direct_cosine_recurrence | 0.0039 | 0.5487 | 0.6730 | 1.9948 | 5240.4021 | 4191.4167 | 0.7541 |
| normwear / direct_cosine_recurrence | 0.0031 | 0.5598 | 0.4375 | 1.4179 | 6635.9941 | 5311.5833 | 0.7463 |
| unimts / direct_cosine_recurrence | 0.0035 | 0.5501 | 0.6845 | 2.1261 | 5797.8912 | 4638.7917 | 0.7538 |

## TASK3 - wear - learned_metric_recurrence

| encoder | occurrence_precision | occurrence_recall | bcubed_f1 | mean_fragments_per_true_motif | false_occurrences_per_hour | mean_absolute_count_error | matched_mean_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| halo_pb04 / learned_metric_recurrence | 0.0031 | 0.5522 | 0.6699 | 2.1671 | 6549.8471 | 5242.2292 | 0.7599 |
| harnet / learned_metric_recurrence | 0.0036 | 0.5535 | 0.6698 | 2.0031 | 5775.2250 | 4620.7083 | 0.7515 |
| normwear / learned_metric_recurrence | 0.0030 | 0.5605 | 0.3093 | 1.1207 | 6990.2925 | 5595.8958 | 0.7452 |
| unimts / learned_metric_recurrence | 0.0033 | 0.5556 | 0.6699 | 1.9940 | 6174.3105 | 4941.0000 | 0.7589 |
