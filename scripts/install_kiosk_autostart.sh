#!/usr/bin/env sh
set -eu

AUTOSTART_DIR="/home/raspi/.config/autostart"
CHROME_PROFILE="/home/raspi/.cache/rollcall-chromium"
DESKTOP_FILE="${AUTOSTART_DIR}/rollcall-kiosk.desktop"

mkdir -p "${AUTOSTART_DIR}" "${CHROME_PROFILE}"
cp /home/raspi/Documents/raspi_rollcall/scripts/rollcall-kiosk.desktop "${DESKTOP_FILE}"
chown -R raspi:raspi /home/raspi/.config /home/raspi/.cache/rollcall-chromium
chmod 700 "${CHROME_PROFILE}"
chmod 644 "${DESKTOP_FILE}"

echo "Installed: ${DESKTOP_FILE}"
