from collections import deque
import csv
import json
import os
import time
import cv2, numpy as np, tkinter as tk
from typing import Optional
from mss import mss

framebuffer = deque(maxlen=10)

def capture_screen(screenshot_size: int = 320, allow_user_to_select_region: bool = False, x: Optional[int] = None, y: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Captures a screenshot of the monitor and returns it as a numpy array.

    Args:
        screenshot_size (int): The size of the square screenshot to capture. Defaults to 320.
        allow_user_to_select_region (bool): Whether or not to allow the user to select the region to capture. Defaults to False.
        x (Optional[int]): The x-coordinate of the top-left corner of the region to capture. If not provided, defaults to the center of the screen.
        y (Optional[int]): The y-coordinate of the top-left corner of the region to capture. If not provided, defaults to the center of the screen.

    Returns:
        Optional[numpy.ndarray]: The screenshot as a numpy array with shape (screenshot_size, screenshot_size, 3), or None if cancelled.
    """
    global selected_x, selected_y
    
    if allow_user_to_select_region:
        select_region(screenshot_size)
        x, y = selected_x, selected_y
    else:
        with mss() as screen_capture:
            if len(screen_capture.monitors) < 2:
                raise ValueError("At least two monitors are required.")
            screen_region = screen_capture.monitors[1]
            if x is None or y is None:
                x, y = map(lambda coord: int(coord / 2 - screenshot_size / 2), (screen_region['width'], screen_region['height']))
    
    with mss() as screen_capture:
        screen_region = {'left': x, 'top': y, 'width': screenshot_size, 'height': screenshot_size}
        return cv2.cvtColor(np.array(screen_capture.grab(screen_region)), cv2.COLOR_BGRA2RGB)

def select_region(screenshot_size: int) -> None:
    global selected_x, selected_y
    
    def on_mouse_down(event: tk.Event) -> None:
        global start_x, start_y, dragging
        start_x = event.x
        start_y = event.y
        dragging = True

    def on_mouse_move(event: tk.Event) -> None:
        global start_x, start_y, dragging, rect, canvas
        if dragging:
            dx = event.x - start_x
            dy = event.y - start_y
            canvas.move(rect, dx, dy)
            start_x, start_y = event.x, event.y

    def on_mouse_up(event: tk.Event) -> None:
        global dragging
        dragging = False
        
    def on_key_press(event: tk.Event) -> None:
        global root
        if event.char == 'y': 
            root.quit()

    global root, canvas, rect, start_x, start_y
    root = tk.Tk()
    root.bind("<Key>", on_key_press)
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.3)
    canvas = tk.Canvas(root)
    canvas.pack(fill="both", expand=True)
    initial_x, initial_y = 100, 100
    rect = canvas.create_rectangle(initial_x, initial_y, initial_x + screenshot_size, initial_y + screenshot_size, outline='red', width=2)
    
    instructions = canvas.create_text(root.winfo_screenwidth() // 2, 50, text="Drag the rectangle to select region. Press 'y' to confirm.", fill="white", font=("Arial", 24))
    
    dragging = False
    start_x, start_y = 0, 0
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    root.mainloop()
    
    if rect:
        x1, y1, x2, y2 = canvas.coords(rect)
        selected_x, selected_y = int((x1 + x2) / 2 - screenshot_size / 2), int((y1 + y2) / 2 - screenshot_size / 2)
    
    root.destroy()