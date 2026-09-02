# Google Drive on this box — one-time setup

Goal: `~/GoogleDrive` on this machine == the `HALO` folder in Google Drive
(`My Drive > … > HALO`), the same folder Drive syncs on the laptop. Slides and
figures written there appear on the laptop automatically.

Already done: rclone v1.75 installed at `~/.local/bin/rclone`; FUSE is available;
mount script at `~/.local/bin/mount-gdrive` (mounts only the HALO folder, id
`1EiX5txMxwjzG1vV8yVmfzq8xj9yMFNxn`, with write-back caching).

Remaining: one OAuth authorization, which needs a browser. Two ways:

## Option A — authorize from the laptop (recommended)
1. On the laptop (rclone installed there: `brew install rclone` / winget / etc.):
   `rclone authorize "drive"`
   A browser opens; sign in as maleficent219@gmail.com; rclone prints a token JSON.
2. In this Claude session, paste it into:
   `! ~/.local/bin/rclone config create gdrive drive scope drive token '<PASTE-THE-JSON>'`

## Option B — SSH port forward, no rclone on laptop
1. Reconnect to this box with: `ssh -L 53682:localhost:53682 <this-box>`
2. In this Claude session run: `! ~/.local/bin/rclone config create gdrive drive scope drive`
   It prints a `http://127.0.0.1:53682/auth?...` URL — open it in the laptop
   browser (the forward makes it reach this box), approve, done.

## Then
`! mount-gdrive`  → `~/GoogleDrive` is live. New decks/figures get written (or
copied) there under `comms/`.

To make the mount survive reboots, add to crontab: `@reboot ~/.local/bin/mount-gdrive`.
