import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw
from datetime import datetime
import threading
import time
import tkinter as tk
from tkinter import messagebox

# ======================
# CONFIGURE COUNTDOWNS
# ======================
COUNTDOWNS = {
    "GATE 2026": (2026, 2, 7, 9, 0, 0),
    "Project Deadline": (2026, 5, 1, 0, 0, 0),
}

# ----------------------
# ICON (emoji only here)
# ----------------------
def create_icon():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((18, 18), "⏳", fill=(255, 255, 255))
    return img

# ----------------------
# TIME FORMAT (macOS SAFE)
# ----------------------
def format_hms(target: datetime) -> str:
    diff = target - datetime.now()

    if diff.total_seconds() <= 0:
        return "000:00:00"

    total = int(diff.total_seconds())
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60

    # Constant width → no menu bar jumping
    return f"{hours:03}:{minutes:02}:{seconds:02}"

# ----------------------
# DETAILS POPUP
# ----------------------
def show_details():
    root = tk.Tk()
    root.withdraw()

    msg = ""
    for name, dt in targets.items():
        msg += f"{name}\n{format_hms(dt)}\n\n"

    messagebox.showinfo("Countdowns", msg.strip())
    root.destroy()

# ----------------------
# MENU BAR UPDATER
# ----------------------
def update_title(icon, first_target):
    while True:
        icon.title = format_hms(first_target)
        time.sleep(1)

# ----------------------
# EXIT HANDLER
# ----------------------
def exit_app(icon, _):
    icon.stop()

# ----------------------
# MAIN
# ----------------------
def main():
    global targets
    targets = {name: datetime(*dt) for name, dt in COUNTDOWNS.items()}
    first_target = next(iter(targets.values()))

    icon = pystray.Icon(
        "Countdown",
        create_icon(),
        title="000:00:00",
        menu=pystray.Menu(
            item("View all countdowns", lambda _: show_details()),
            item("Exit", exit_app)
        )
    )

    threading.Thread(
        target=update_title,
        args=(icon, first_target),
        daemon=True
    ).start()

    icon.run()

# ----------------------
# ENTRY POINT
# ----------------------
if __name__ == "__main__":
    main()
