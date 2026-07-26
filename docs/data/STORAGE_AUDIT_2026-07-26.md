# Workstation storage audit (2026-07-26)

Read-only audit of all storage visible without administrator authentication. No files were deleted.

## Filesystems

| Device | Filesystem | State | Capacity | Used | Free |
|---|---|---:|---:|---:|---:|
| `/dev/mapper/ubuntu--vg-ubuntu--lv` mounted at `/` | ext4 | mounted | 1.9 TB | 618 GB (35%) | 1.2 TB |
| `/dev/nvme0n1p1` | NTFS | **not mounted** | 931.5 GB | unknown | unknown |
| `/boot` | ext4 | mounted | 2.0 GB | 201 MB | 1.7 GB |

Root has about 120 million inodes; only 2.0 million (2%) are used. Capacity, not inode exhaustion, is
the relevant constraint. The unmounted NTFS volume could not be audited: both `udisksctl` and a
read-only mount require Polkit/sudo authentication unavailable to this process. Therefore this is a
complete audit of the mounted Linux filesystem, not a claim about the NTFS volume's contents.

`df` is authoritative for free space. Some protected system directories under `/var` and other
root-owned paths were unreadable without sudo.

The first snapshot in this audit showed 749 GB used / 1.1 TB free. During the work, another process or
user removed about 131 GB: the final snapshot is 618 GB used / 1.2 TB free. This task did not delete
caches or project files. The tables below describe the final state.

## Largest visible trees

| Path | Approximate size | Main contents |
|---|---:|---|
| `/home/alex/code` | 298 GiB | research repos, datasets, environments, outputs |
| `/home/alex/.cache` | 270 GiB | predominantly Hugging Face model/dataset caches |
| `/home/alex/code/HALO` | 181 GiB | 109 GiB current repo + 72 GiB legacy tree |
| `/home/alex/code/neuromorphic` | 70 GiB | 44 GiB code/outputs + 26 GiB baselines |
| `/home/alex/code/memoriesai` | 33 GiB | 25 GiB OSWorld qcow2 tree + 8 GiB environment |
| `/home/alex/code/vea2` | 11 GiB | project and playground trees after concurrent cleanup |
| `/home/alex/code/vea-playground` | 3.1 GiB | project tree after concurrent cleanup |

The current HALO tree gained roughly 39 GB net from the uncapped Capture-24 native/harmonised grids
and bounded ExtraSensory/NHANES materialisations during this work. About 1.2 TB remains free.

## Cache concentration

Hugging Face accounts for about 267 GiB of `/home/alex/.cache`: roughly 160.7 GiB in `hub`, 101.5 GiB
in `datasets`, and 4.6 GiB in `xet`. The largest individual cached assets include:

| Cache | Approximate size |
|---|---:|
| FineWeb-Edu dataset + hub artifacts | 81 GiB |
| `YuWangX/mplus-8b` | 36 GiB |
| Llama 3.1 8B | 30 GiB |
| Qwen 2.5 7B | 14 GiB |
| `BASH-Lab/LLaSA-7B` | 13 GiB |
| `infmem-4B` | 8 GiB |

Other current consumers include VS Code server (5.2 GiB), Cursor server (8.4 GiB), Codex (3.9 GiB),
and Claude (3.3 GiB). Large pip/compiler/npm caches seen in the first snapshot disappeared during the
concurrent cleanup.

## Duplicate and large project assets

- Capture-24's source zip exists in both current and legacy HALO trees at about 6.9 GB each.
- ExtraSensory's large archives already existed in the legacy tree; the current fetcher symlinks them
  rather than making another 7.5 GB copy.
- Other single large HALO assets include the ImageBind checkpoint (4.8 GB), TNDA archive (4.8 GB),
  RealWorld archive (3.7 GB), and large HHA phone signal files (about 2.7 GB combined).

## Cleanup order

1. Evict unused Hugging Face model/dataset snapshots with the Hugging Face cache tooling, after
   checking which experiments still need offline access. This is the largest low-risk opportunity.
2. Clear any remaining reproducible package/compiler caches. They will be rebuilt or downloaded again.
3. Remove stale editor/agent server versions through their own maintenance tools.
4. Deduplicate source archives and repeated inputs only after checking that no scripts rely on
   the existing path. Prefer a single canonical file plus symlinks.
5. Treat research outputs, qcow2 images, checkpoints, and converted datasets as owner-reviewed
   deletions, not cache cleanup.

No cleanup is needed to run HALO training now. A substantially larger bounded NHANES pilot fits
comfortably in the current 1.2 TB free space.
