import ctypes
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog
from ctypes import wintypes

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.spinner import Spinner
from rich.table import Table

import cv2
import keyboard
import sys
import subprocess
import colorama
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from playsound import playsound

import kornia.enhance as K
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Your new model definition
from model import create_model

def play_sound_async(sound_name):
    """Plays a sound from the 'sounds' folder in a non-blocking background thread."""
    sound_path = os.path.join("sounds/Felix", sound_name)
    if os.path.exists(sound_path):
        # daemon=True ensures the thread won't prevent the program from exiting
        threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

# --- NEW SIMPLIFIED CONFIGURATION ---
# These settings control the entire artistic pipeline, from AI generation to final output.

# --- Stage 0: Pre-Upscaling (Real-ESRGAN) ---
USE_UPSCALER = True             # Set to True to use Real-ESRGAN before processing.
UPSCALER_MODEL = 'realesrgan-x4plus-anime' # Optimized model for anime style.

# --- Stage 1: AI Art Generation ---
# These are the settings from our perfected Anime2Sketch script.
MODEL_TYPE = 'default'          # 'default' or 'improved'
LOAD_SIZE = 2048                # AI processing resolution. 2048 is crisp but slow. 1024 is a good balance.
CLAHE_CLIP_LIMIT = 4.0          # The "secret sauce." Enhances input contrast for thicker, more confident lines. (0 to disable)

# --- Stage 2: Post-Processing & Thresholding ---
# This pipeline cleans the AI's output and converts it to a final black & white image.
USE_POST_PROCESSING = True      # Set to False to disable all cleaning.

INTERACTIVE_TUNING = True       # POP-UP UI: Visually tune settings in real-time before drawing.

# DEFAULT VALUES (Used immediately if INTERACTIVE_TUNING is False, or as starting points if True)
CLAHE_CONTRAST = 20             # Local contrast boost (0-100). Forces faint gray details to become black.
BINARY_THRESHOLD = 240          # Fixed global cutoff.
MEDIAN_BLUR_SIZE = 3            # Smooths out tiny noise dots. Must be an odd number.

# --- Stage 4: Pixel-Level Cleanup ---
# This is now applied to BOTH the lineart and shading layers independently.
MINIMUM_PIXEL_AREA = 2 # Removes any blob smaller than this pixel count.

# --- Stage 5: Drawing Parameters ---
DRAWING_SPEED_PERCENT = 100     # Overall speed. 100 is fastest, 10 is 10% speed. Affects all drawing delays.
GLIDE_STEP_SIZE = 3            # For smooth pen-up moves in Innovative mode. Higher is faster/jumpier, 1 is pixel-by-pixel.
BATCH_SIZE = 7                 # Accumulates this many small moves into one driver command for speed.
INTER_STROKE_DELAY_SEC = 0.001  # The base "cool-down" between strokes, used at 100% speed.

# --- Constants and Proven Driver Interface Code ---

INVALID_HANDLE_VALUE = -1
DIGCF_PRESENT = 0x00000002
DIGCF_DEVICEINTERFACE = 0x00000010
FILE_ATTRIBUTE_NORMAL = 0x00000080
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
IOCTL_MOUSE = 0x88883020

PVOID = ctypes.c_void_p
HDEVINFO = PVOID


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.ULONG),
        ("Data2", wintypes.USHORT),
        ("Data3", wintypes.USHORT),
        ("Data4", ctypes.c_ubyte * 8)
    ]
LPCGUID = ctypes.POINTER(GUID)

class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("InterfaceClassGuid", GUID),
                ("Flags", wintypes.DWORD), ("Reserved", PVOID)]

class SP_DEVINFO_DATA(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("ClassGuid", GUID),
                ("DevInst", wintypes.DWORD), ("Reserved", PVOID)]

GUID_DEVINTERFACE_RZCONTROL = GUID(
    0xe3be005d, 0xd130, 0x4910,
    (ctypes.c_ubyte * 8)(0x88, 0xff, 0x09, 0xae, 0x02, 0xf6, 0x80, 0xe9)
)

setupapi = ctypes.windll.setupapi
kernel32 = ctypes.windll.kernel32

setupapi.SetupDiGetClassDevsW.restype = HDEVINFO
setupapi.SetupDiGetClassDevsW.argtypes = [LPCGUID, wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD]
setupapi.SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL
setupapi.SetupDiEnumDeviceInterfaces.argtypes = [HDEVINFO, PVOID, LPCGUID, wintypes.DWORD,
                                                 ctypes.POINTER(SP_DEVICE_INTERFACE_DATA)]
setupapi.SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL
setupapi.SetupDiGetDeviceInterfaceDetailW.argtypes = [HDEVINFO, ctypes.POINTER(SP_DEVICE_INTERFACE_DATA), PVOID,
                                                      wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                      ctypes.POINTER(SP_DEVINFO_DATA)]
setupapi.SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL
setupapi.SetupDiDestroyDeviceInfoList.argtypes = [HDEVINFO]
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, PVOID, wintypes.DWORD,
                                 wintypes.DWORD, wintypes.HANDLE]
kernel32.DeviceIoControl.restype = wintypes.BOOL
kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, PVOID, wintypes.DWORD, PVOID, wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD), PVOID]


# --- AI Model Helper Functions ---

def setup_device():
    """Checks for DirectML availability and sets the device accordingly."""
    try:
        import torch_directml
        dml_device = torch_directml.device(torch_directml.default_device())
        console.print(f"Using: {dml_device}")
        return dml_device
    except (ImportError, Exception):
        console.print("DirectML not found or failed. Falling back to CPU.")
        return torch.device("cpu")

def load_model(device):
    """Loads the Anime2Sketch model."""
    model_type = MODEL_TYPE
    net = create_model(model=model_type)
    net.to(device)
    net.eval()
    console.print("Model loaded successfully.")
    return net

def pad_to_square(image):
    """Pads a PIL image to a square aspect ratio without distortion."""
    width, height = image.size
    max_dim = max(width, height)
    # Use white padding to match the sketch background
    square_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    square_image.paste(image, (paste_x, paste_y))
    return square_image, paste_x, paste_y, max_dim

# --- High-Precision Timer for Ultra Speed ---

class PreciseSleeper:
    """
    A context manager to temporarily increase the Windows system timer resolution.
    This allows for much more accurate, short sleep durations.
    """
    def __init__(self, period_ms=1):
        self.period_ms = period_ms
        self.winmm = ctypes.WinDLL('winmm.dll')

    def __enter__(self):
        self.winmm.timeBeginPeriod(self.period_ms)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.winmm.timeEndPeriod(self.period_ms)


# --- Razer Driver and Drawing Classes ---

class RazerMouse:
    class RazerIOCtl(ctypes.Structure):
        _fields_ = [
            ("unknown0", ctypes.c_int32), ("command_id", ctypes.c_int32),
            ("max_val", ctypes.c_int32), ("click_state", ctypes.c_int32),
            ("unknown1", ctypes.c_int32), ("x_delta", ctypes.c_int32),
            ("y_delta", ctypes.c_int32), ("unknown2", ctypes.c_int32)
        ]

    def __init__(self):
        self.device_handle = None
        self._find_and_open_device()

    def _find_and_open_device(self):
        h_dev_info = setupapi.SetupDiGetClassDevsW(ctypes.byref(GUID_DEVINTERFACE_RZCONTROL), None, None,
                                                   DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
        if not h_dev_info or h_dev_info == INVALID_HANDLE_VALUE: raise IOError(
            "Failed to get device info set for Razer devices.")
        try:
            dev_interface_data = SP_DEVICE_INTERFACE_DATA()
            dev_interface_data.cbSize = ctypes.sizeof(dev_interface_data)
            dev_index = 0
            while setupapi.SetupDiEnumDeviceInterfaces(h_dev_info, None, ctypes.byref(GUID_DEVINTERFACE_RZCONTROL),
                                                       dev_index, ctypes.byref(dev_interface_data)):
                dev_index += 1
                required_size = wintypes.DWORD()
                setupapi.SetupDiGetDeviceInterfaceDetailW(h_dev_info, ctypes.byref(dev_interface_data), None, 0,
                                                          ctypes.byref(required_size), None)
                if required_size.value == 0: continue
                detail_data_buffer = ctypes.create_string_buffer(required_size.value)
                ctypes.cast(detail_data_buffer, ctypes.POINTER(wintypes.DWORD))[0] = 8
                dev_info_data = SP_DEVINFO_DATA()
                dev_info_data.cbSize = ctypes.sizeof(dev_info_data)
                if not setupapi.SetupDiGetDeviceInterfaceDetailW(h_dev_info, ctypes.byref(dev_interface_data),
                                                                 detail_data_buffer, required_size, None,
                                                                 ctypes.byref(dev_info_data)): continue
                device_path = ctypes.wstring_at(ctypes.addressof(detail_data_buffer) + 4)
                self.device_handle = kernel32.CreateFileW(device_path, wintypes.DWORD(0xC0000000),
                                                          wintypes.DWORD(FILE_SHARE_READ | FILE_SHARE_WRITE), None,
                                                          wintypes.DWORD(OPEN_EXISTING),
                                                          wintypes.DWORD(FILE_ATTRIBUTE_NORMAL), None)
                if self.device_handle != INVALID_HANDLE_VALUE:
                    return
        finally:
            setupapi.SetupDiDestroyDeviceInfoList(h_dev_info)
        raise IOError("Could not find or open a handle to a Razer device.")

    def _send_command(self, data):
        if self.device_handle is None or self.device_handle == INVALID_HANDLE_VALUE: return
        bytes_returned = wintypes.DWORD()
        kernel32.DeviceIoControl(self.device_handle, IOCTL_MOUSE, ctypes.byref(data), ctypes.sizeof(data), None, 0,
                                 ctypes.byref(bytes_returned), None)

    def move_relative(self, dx, dy):
        dx = max(-32767, min(dx, 32767));
        dy = max(-32767, min(dy, 32767))
        payload = self.RazerIOCtl(command_id=2, x_delta=dx, y_delta=dy)
        self._send_command(payload)

    def mouse_down(self):
        payload = self.RazerIOCtl(command_id=2, click_state=1)
        self._send_command(payload)

    def mouse_up(self):
        payload = self.RazerIOCtl(command_id=2, click_state=2)
        self._send_command(payload)

    def close(self):
        if self.device_handle and self.device_handle != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(self.device_handle)
            console.print("Razer device handle closed.")

class ImageSketcher:
    COMMAND_PACKET_SIZE = 32  # 8 fields * 4 bytes/field
    def __init__(self, razer_mouse):
        self.razer = razer_mouse
        self.canvas_top_left = None
        self.canvas_bottom_right = None
        self.image_shape = None
        self.panic_event = threading.Event()
        self._log_messages = []
        self.api_call_counter = 0
        self.current_pen_state = "IDLE"
        self.total_distance_traveled = 0.0

        # Calculate drawing delays based on the global speed percentage
        speed_modifier = DRAWING_SPEED_PERCENT / 100.0
        if speed_modifier <= 0: speed_modifier = 0.001 # Avoid division by zero and make it very slow
        self.inter_stroke_delay = INTER_STROKE_DELAY_SEC / speed_modifier
        # Use the same base delay for intra-stroke sleeps for simplicity
        self.intra_stroke_delay = INTER_STROKE_DELAY_SEC / speed_modifier

        # --- Load the AI Model ---
        self.device = setup_device()
        self.lineart_model = load_model(self.device)

    def _get_image_path(self):
        root = tk.Tk();
        root.withdraw()
        path = filedialog.askopenfilename(title="Select an Image to Sketch",
                                          filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            console.print("No image selected.")
            return None
        return path

    def _draw_path(self, path, scale, offset_x, offset_y, scaling_factor, is_shading=False):
        """
        Draws a path of vertices by interpolating lines between them.
        Includes Engine Stabilization delays to ensure Skribbl.io/Roblox register inputs.
        Dynamically adjusts speed: lightning fast for straight shading, ultra-smooth for sketches.
        """
        if len(path) < 1 or self.panic_event.is_set():
            return

        start_img_x, start_img_y = path[0]
        start_screen_x = int(self.canvas_top_left[0] + offset_x + (start_img_x * scale))
        start_screen_y = int(self.canvas_top_left[1] + offset_y + (start_img_y * scale))

        self._glide_to(start_screen_x, start_screen_y, scaling_factor, is_shading=is_shading)

        if self.panic_event.is_set(): return

        self.razer.mouse_down()
        time.sleep(0.015) # ENGINE STABILIZATION: Let game engine poll the click

        self.current_pen_state = "[yellow]DRAWING[/yellow]"

        batched_dx = 0
        batched_dy = 0
        last_screen_x, last_screen_y = start_screen_x, start_screen_y
        move_count = 0

        active_batch_size = BATCH_SIZE
        active_delay = self.intra_stroke_delay

        for i in range(1, len(path)):
            if self.panic_event.is_set(): break

            x1, y1 = path[i - 1]
            x2, y2 = path[i]

            dx_line = abs(x2 - x1)
            dy_line = -abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx_line + dy_line

            while True:
                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 >= dy_line:
                    err += dy_line
                    x1 += sx
                if e2 <= dx_line:
                    err += dx_line
                    y1 += sy

                screen_x = int(self.canvas_top_left[0] + offset_x + (x1 * scale))
                screen_y = int(self.canvas_top_left[1] + offset_y + (y1 * scale))

                dx_move = screen_x - last_screen_x
                dy_move = screen_y - last_screen_y

                if dx_move != 0 or dy_move != 0:
                    self.total_distance_traveled += (dx_move ** 2 + dy_move ** 2) ** 0.5
                    batched_dx += dx_move
                    batched_dy += dy_move
                    move_count += 1

                    if move_count >= active_batch_size:
                        self.razer.move_relative(int(batched_dx * scaling_factor), int(batched_dy * scaling_factor))
                        self.api_call_counter += 1
                        if active_delay > 0:
                            time.sleep(active_delay)
                        batched_dx, batched_dy, move_count = 0, 0, 0

                last_screen_x, last_screen_y = screen_x, screen_y

        if batched_dx != 0 or batched_dy != 0:
            self.razer.move_relative(int(batched_dx * scaling_factor), int(batched_dy * scaling_factor))
            self.api_call_counter += 1
            if active_delay > 0:
                time.sleep(active_delay)

        time.sleep(0.015) # ENGINE STABILIZATION: Let game engine poll the final position
        self.razer.mouse_up()

    def _calibrate_canvas(self, live, layout):
        self._log("Awaiting canvas calibration...")

        def make_panel(text, is_active=False):
            style = "cyan" if is_active else "dim"
            return Panel(text, title="[bold]Canvas Calibration", border_style=style)

        # Top-left
        layout["main"].update(make_panel(
            "1. Open MS Paint on your main monitor.\n2. Maximize the window.\n3. Move your mouse to the [bold]TOP-LEFT[/bold] corner of the white canvas.\n\n[bold blink]>>> PRESS ENTER TO CAPTURE <<<[/bold blink]",
            is_active=True))
        live.refresh()
        play_sound_async("top_left.mp3")
        keyboard.wait('enter')
        self.canvas_top_left = self._get_mouse_pos_from_user()
        self._log(f"Top-left captured: {self.canvas_top_left}")

        # Bottom-right
        layout["main"].update(make_panel(
            f"Top-left: {self.canvas_top_left}\n\nNow, move your mouse to the [bold]BOTTOM-RIGHT[/bold] corner of the canvas.\n\n[bold blink]>>> PRESS ENTER TO CAPTURE <<<[/bold blink]",
            is_active=True))
        live.refresh()
        play_sound_async("bottom_right.mp3")
        time.sleep(0.2)
        keyboard.wait('enter')
        self.canvas_bottom_right = self._get_mouse_pos_from_user()
        self._log(f"Bottom-right captured: {self.canvas_bottom_right}")
        return True

    def _get_mouse_pos_from_user(self):
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _glide_to(self, target_x, target_y, scaling_factor, is_shading=False):
        """Moves the mouse smoothly from its current position to a target with the pen up."""
        if self.panic_event.is_set(): return
        self.current_pen_state = "[bright_cyan]GLIDING[/bright_cyan]"
        start_x, start_y = self._get_mouse_pos_from_user()
        total_dx, total_dy = target_x - start_x, target_y - start_y
        distance = (total_dx ** 2 + total_dy ** 2) ** 0.5
        if distance < 1:
            self.total_distance_traveled += distance
            return

        active_glide_step = GLIDE_STEP_SIZE
        active_delay = self.intra_stroke_delay

        num_steps = int(distance / active_glide_step) + 1 if active_glide_step > 0 else 1
        last_int_x, last_int_y = start_x, start_y

        for i in range(1, num_steps + 1):
            if self.panic_event.is_set(): break
            progress = i / num_steps
            ideal_x, ideal_y = start_x + total_dx * progress, start_y + total_dy * progress
            move_dx, move_dy = int(ideal_x) - last_int_x, int(ideal_y) - last_int_y
            if move_dx != 0 or move_dy != 0:
                self.razer.move_relative(int(move_dx * scaling_factor), int(move_dy * scaling_factor))
                self.api_call_counter += 1
                last_int_x += move_dx
                last_int_y += move_dy
            if active_delay > 0:
                time.sleep(active_delay)

        if active_delay == 0:
            time.sleep(0.015)

        # Final correction to ensure it lands exactly on the target
        final_x, final_y = self._get_mouse_pos_from_user()
        correction_dx, correction_dy = target_x - final_x, target_y - final_y
        if abs(correction_dx) > 0 or abs(correction_dy) > 0:
            self.razer.move_relative(int(correction_dx * scaling_factor), int(correction_dy * scaling_factor))
            self.api_call_counter += 1

    def _process_image(self, path, live, layout):
        """
        The new, simplified Art Engine. It generates a high-quality grayscale lineart,
        cleans it, and converts it to a single, final binary image for drawing.
        """
        # --- STAGE 0: REAL-ESRGAN UPSCALING ---
        if USE_UPSCALER:
            upscaler_dir = os.path.abspath("upscaler")
            upscaler_exe = os.path.join(upscaler_dir, "realesrgan-ncnn-vulkan.exe")
            upscaled_temp = os.path.join(upscaler_dir, "temp_upscaled.png")

            if os.path.exists(upscaler_exe):
                self._log("Starting Real-ESRGAN upscaling...")
                layout["main"].update(
                    Panel(Spinner("earth", text="[bold]Upscaling image (4x)...[/bold]"),
                          title="[bold]Stage 0: Super-Resolution",
                          border_style="magenta"))
                live.refresh()

                try:
                    # Run upscaler without popping up a new window
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                    cmd = [upscaler_exe, "-i", path, "-o", upscaled_temp, "-n", UPSCALER_MODEL]
                    subprocess.run(cmd, cwd=upscaler_dir, check=True, capture_output=True, startupinfo=startupinfo)

                    path = upscaled_temp  # SWAP PATH: Subsequent stages now use the high-res image
                    self._log("Upscaling complete.")
                except Exception as e:
                    self._log(f"[red]Upscaling failed: {e}[/red]")
                    # We intentionally don't return here, we just fall back to the original 'path'
            else:
                self._log("[yellow]Upscaler executable not found. Skipping.[/yellow]")

        # --- STAGE 1: AI ART GENERATION ---
        layout["main"].update(
            Panel(Spinner("dots", text="[bold]Running AI Art Engine...[/bold]"), title="[bold]Stage 1: AI Processing",
                  border_style="cyan"))
        live.refresh()
        play_sound_async("processing.mp3")
        original_image = Image.open(path).convert("RGB")
        original_width, original_height = original_image.size

        # --- Stage 1: AI Generation ---
        padded_image, pad_x, pad_y, max_dim = pad_to_square(original_image)
        to_tensor = transforms.Compose([
            transforms.Resize((LOAD_SIZE, LOAD_SIZE)),
            transforms.ToTensor(),
        ])
        input_tensor = to_tensor(padded_image)
        if CLAHE_CLIP_LIMIT > 0:
            console.print(f"Applying CLAHE with clip limit: {CLAHE_CLIP_LIMIT}")
            input_tensor = K.equalize_clahe(input_tensor.unsqueeze(0), clip_limit=float(CLAHE_CLIP_LIMIT)).squeeze(0)

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        input_tensor = normalize(input_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.lineart_model(input_tensor)

        output_data = (output_tensor.squeeze(0).cpu().float().numpy().transpose(1, 2, 0) + 1.0) / 2.0 * 255.0
        lineart_square = Image.fromarray(output_data.astype(np.uint8).squeeze(), 'L')

        scale_ratio = LOAD_SIZE / max_dim
        crop_x = int(pad_x * scale_ratio)
        crop_y = int(pad_y * scale_ratio)
        crop_width = int(original_width * scale_ratio)
        crop_height = int(original_height * scale_ratio)
        grayscale_master_pil = lineart_square.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

        self.image_shape = (grayscale_master_pil.height, grayscale_master_pil.width)
        grayscale_master = np.array(grayscale_master_pil)

        # Local variables to track tuning states
        active_blur = MEDIAN_BLUR_SIZE
        active_contrast = CLAHE_CONTRAST
        fixed_thresh = BINARY_THRESHOLD

        if INTERACTIVE_TUNING:
            self._log("Opening Interactive Tuning Engine...")
            layout["main"].update(Panel(Align.center(
                "\n\n[bold cyan]INTERACTIVE TUNING ACTIVE[/bold cyan]\n\nPlease check the popup window to tune your sketch.\nPress[bold yellow]ENTER[/bold yellow] in the popup when finished."),
                                        border_style="cyan"))
            live.refresh()

            cv2.namedWindow("SOTA Art Tuner - Press ENTER to Accept", cv2.WINDOW_NORMAL)
            
            aspect_ratio = grayscale_master.shape[1] / grayscale_master.shape[0]  # width / height
            if aspect_ratio >= 1:
                win_w, win_h = 800, int(800 / aspect_ratio)
            else:
                win_h, win_w = 800, int(800 * aspect_ratio)
            cv2.resizeWindow("SOTA Art Tuner - Press ENTER to Accept", win_w, win_h)

            # Create trackbars (Threshold slider removed, fixed at 240)
            cv2.createTrackbar("1. Contrast (Darken Lines)", "SOTA Art Tuner - Press ENTER to Accept", active_contrast,
                               100, lambda x: None)
            cv2.createTrackbar("2. Blur Level", "SOTA Art Tuner - Press ENTER to Accept", active_blur, 9,
                               lambda x: None)

            while True:
                # Read GUI slider positions
                active_contrast = cv2.getTrackbarPos("1. Contrast (Darken Lines)",
                                                     "SOTA Art Tuner - Press ENTER to Accept")
                raw_blur = cv2.getTrackbarPos("2. Blur Level", "SOTA Art Tuner - Press ENTER to Accept")

                # Sanitize constraints
                active_blur = raw_blur if raw_blur % 2 != 0 else raw_blur + 1

                # 1. Apply Live Contrast Enhancement (CLAHE)
                temp_gray = grayscale_master.copy()
                if active_contrast > 0:
                    clip_limit = active_contrast / 10.0
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                    temp_gray = clahe.apply(temp_gray)

                # 2. Apply Live Blur
                if active_blur > 1:
                    temp_gray = cv2.medianBlur(temp_gray, active_blur)

                # 3. Apply Fixed Global Threshold
                _, live_binary = cv2.threshold(temp_gray, fixed_thresh, 255, cv2.THRESH_BINARY_INV)

                cv2.imshow("SOTA Art Tuner - Press ENTER to Accept", live_binary)

                # Wait for ENTER (13) or ESC (27)
                key = cv2.waitKey(33) & 0xFF
                if key == 13 or key == 27:
                    break

            cv2.destroyAllWindows()
            for _ in range(4): cv2.waitKey(1)
            self._log(f"Tuning Locked: Contrast={active_contrast / 10.0}, Blur={active_blur}, Thresh={fixed_thresh}")

        # --- Stage 2 & 3: Apply the Selected Parameters ---
        processed_grayscale = grayscale_master.copy()
        if active_contrast > 0:
            clip_limit = active_contrast / 10.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            processed_grayscale = clahe.apply(processed_grayscale)

        if USE_POST_PROCESSING and active_blur > 1:
            processed_grayscale = cv2.medianBlur(processed_grayscale, active_blur)

        _, binary_image = cv2.threshold(processed_grayscale, fixed_thresh, 255, cv2.THRESH_BINARY_INV)

        # --- Stage 4: Final Pixel Cleanup ---
        final_image = binary_image
        if USE_POST_PROCESSING and MINIMUM_PIXEL_AREA > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, 4, cv2.CV_32S)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < MINIMUM_PIXEL_AREA:
                    final_image[labels == i] = 0

        # --- DEBUG STEP ---
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "0_grayscale_master.png"), grayscale_master)
        cv2.imwrite(os.path.join(debug_dir, "1_binary_final.png"), final_image)
        console.print(f"DEBUG: Saved processing stages to '{debug_dir}'")

        console.print("Artistic pipeline complete.")
        return final_image

    def _extend_path(self, point_to_extend, segments, endpoints, distance):
        """Helper function to find and consume the next segment to extend a path."""
        for y in range(point_to_extend[1] - distance, point_to_extend[1] + distance + 1):
            for x in range(point_to_extend[0] - distance, point_to_extend[0] + distance + 1):
                search_pt = (x, y)
                if search_pt in endpoints:
                    for seg_start, seg_end in endpoints[search_pt]:
                        seg = (seg_start, seg_end)
                        if seg in segments:
                            segments.remove(seg)
                            other_end = seg_end if seg_start == search_pt else seg_start
                            return True, segments, other_end
        return False, segments, None

    def _panic_listen(self):
        keyboard.wait('esc')
        self.panic_event.set()
        play_sound_async("halted.mp3")
        console.print("\n!!! PANIC KEY PRESSED! Halting drawing. !!!")

    def _render_stroke_to_braille(self, stroke, panel_width, panel_height):
        """Renders a stroke to a high-resolution braille character canvas."""
        if not stroke or len(stroke) < 2:
            return ""

        # Target canvas dimensions in "pixels" (2x4 grid per character)
        canvas_width = panel_width * 2
        canvas_height = panel_height * 4

        # Find the bounding box of the stroke
        min_x = min(p[0] for p in stroke)
        max_x = max(p[0] for p in stroke)
        min_y = min(p[1] for p in stroke)
        max_y = max(p[1] for p in stroke)

        stroke_width = max_x - min_x
        stroke_height = max_y - min_y

        if stroke_width == 0 and stroke_height == 0:
            return ""  # Cannot render a single point

        # Determine the scale to fit the stroke in the canvas, preserving aspect ratio
        scale = 1.0
        if stroke_width > 0:
            scale = min(scale, (canvas_width - 1) / stroke_width)
        if stroke_height > 0:
            scale = min(scale, (canvas_height - 1) / stroke_height)

        # Create a boolean grid (our pixel canvas)
        grid = [[False for _ in range(canvas_width)] for _ in range(canvas_height)]

        # Function to draw a line on the grid (Bresenham's algorithm)
        def draw_line(x1, y1, x2, y2):
            dx = abs(x2 - x1)
            dy = -abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx + dy
            while True:
                if 0 <= x1 < canvas_width and 0 <= y1 < canvas_height:
                    grid[y1][x1] = True
                if x1 == x2 and y1 == y2: break
                e2 = 2 * err
                if e2 >= dy: err += dy; x1 += sx
                if e2 <= dx: err += dx; y1 += sy

        # Translate and scale stroke points, then draw lines between them
        scaled_points = []
        for x, y in stroke:
            # Translate to origin, scale, then center it
            tx = (x - min_x) * scale
            ty = (y - min_y) * scale
            centered_x = int(tx + (canvas_width - stroke_width * scale) / 2)
            centered_y = int(ty + (canvas_height - stroke_height * scale) / 2)
            scaled_points.append((centered_x, centered_y))

        for i in range(len(scaled_points) - 1):
            draw_line(*scaled_points[i], *scaled_points[i + 1])

        # Convert the boolean grid to braille characters
        braille_map = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
        output_lines = []
        for r in range(0, canvas_height, 4):
            line = []
            for c in range(0, canvas_width, 2):
                char_code = 0x2800
                for i in range(4):  # y-offset
                    for j in range(2):  # x-offset
                        if (r + i < canvas_height and c + j < canvas_width and grid[r + i][c + j]):
                            char_code |= braille_map[i][j]
                line.append(chr(char_code))
            output_lines.append("".join(line))
        return "\n".join(output_lines)

    def _calculate_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _scan_mask_at_angle(self, mask, angle, spacing, min_length=3):
        """Scan a binary mask along parallel lines at a given angle.
        Returns a set of segments as ((x1,y1), (x2,y2)) tuples.
        Supported angles: 0 (horizontal), 45 (diagonal), 90 (vertical), 135 (diagonal)
        """
        h, w = mask.shape
        segments = set()

        def _extract_runs(points):
            """From ordered adjacent points, extract contiguous runs as segments."""
            if len(points) < 2:
                return
            start = points[0]
            prev = points[0]
            for pt in points[1:]:
                if abs(pt[0] - prev[0]) <= 1 and abs(pt[1] - prev[1]) <= 1:
                    prev = pt
                else:
                    dist = ((start[0] - prev[0]) ** 2 + (start[1] - prev[1]) ** 2) ** 0.5
                    if dist >= min_length:
                        segments.add((start, prev))
                    start = pt
                    prev = pt
            dist = ((start[0] - prev[0]) ** 2 + (start[1] - prev[1]) ** 2) ** 0.5
            if dist >= min_length:
                segments.add((start, prev))

        if angle == 0:  # Horizontal
            for y in range(0, h, spacing):
                xs = np.where(mask[y, :] > 0)[0]
                points = [(int(x), y) for x in xs]
                _extract_runs(points)

        elif angle == 90:  # Vertical
            for x in range(0, w, spacing):
                ys = np.where(mask[:, x] > 0)[0]
                points = [(x, int(y)) for y in ys]
                _extract_runs(points)

        elif angle == 45:  # Diagonal lines where x + y = c
            c_step = max(1, int(spacing * 1.414))
            for c in range(0, w + h - 1, c_step):
                x_start = max(0, c - (h - 1))
                x_end = min(w - 1, c)
                xs = np.arange(x_start, x_end + 1)
                ys = c - xs
                valid = (ys >= 0) & (ys < h)
                xs, ys = xs[valid], ys[valid]
                active = mask[ys, xs] > 0
                idxs = np.where(active)[0]
                points = [(int(xs[i]), int(ys[i])) for i in idxs]
                _extract_runs(points)

        elif angle == 135:  # Diagonal lines where y - x = c
            c_step = max(1, int(spacing * 1.414))
            for c in range(-(w - 1), h, c_step):
                x_start = max(0, -c)
                x_end = min(w - 1, h - 1 - c)
                xs = np.arange(x_start, x_end + 1)
                ys = xs + c
                valid = (ys >= 0) & (ys < h)
                xs, ys = xs[valid], ys[valid]
                active = mask[ys, xs] > 0
                idxs = np.where(active)[0]
                points = [(int(xs[i]), int(ys[i])) for i in idxs]
                _extract_runs(points)

        return segments

    def _log(self, message):
        """Adds a timestamped message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_messages.append(f"[[dim]{timestamp}[/dim]] {message}")
        # Keep the log from getting too long
        if len(self._log_messages) > 10:
            self._log_messages.pop(0)

    def _format_time(self, seconds):
        """Formats seconds into a MM:SS string."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _format_time_summary(self, seconds):
        """Formats seconds into a M m S s string for the final summary."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes}m {seconds}s"

    def _format_data_rate(self, bps):
        """Formats a bytes-per-second rate into a readable string (B/s, KB/s, etc.)."""
        if bps < 1024:
            return f"{bps:.1f} B/s"
        elif bps < 1024 ** 2:
            return f"{bps / 1024:.2f} KB/s"
        else:
            return f"{bps / (1024 ** 2):.2f} MB/s"

    def _format_distance(self, pixels):
        """Formats a pixel distance into a readable string (px, Kpx, Mpx)."""
        if pixels < 1000:
            return f"{pixels:.0f} px"
        elif pixels < 1_000_000:
            return f"{pixels / 1000:.2f} Kpx"
        else:
            return f"{pixels / 1_000_000:.2f} Mpx"

    def draw_innovative(self, image_layer, live, layout, image_path):
        """Draws using a Multi-Phase Architecture (Sketching -> Shading) with interactive brush switching."""
        # --- TUNABLE PARAMETERS ---
        ENABLE_SHADING = True
        # Pixel-perfect digital screentone bands (for a 1px brush)
        # Format: (pixel_threshold, angle_degrees, line_spacing_px)
        SHADING_TONE_BANDS = [
            (170, 45, 3),    # Light shadows: 33% density diagonal tone
            (120, 135, 3),   # Medium shadows: perpendicular cross-tone
            (70, 45, 1),     # Deep shadows: 100% solid fill (pixel-perfect)
        ]
        SHADING_LINE_MARGIN = 0    # Pixel-perfect flushness (0 = no white halos)
        SHADING_MIN_LINE_LEN = 2   # Fill into the tightest corners
        SHADING_STITCH_DIST = 5    
        STITCHING_DISTANCE = 3
        GRID_DIVISIONS = 2
        WINDOWS_SCALING_FACTOR = 2.0

        panic_thread = threading.Thread(target=self._panic_listen, daemon=True)
        panic_thread.start()

        def process_phase(layer, phase_title, step, min_len, stitch_dist, pre_segments=None):
            self._log(f"[{phase_title}] Generating line plan...")
            if pre_segments is not None:
                segments = set(pre_segments)
            else:
                ys, xs = np.where(layer > 0)
                rows = {}
                for y, x in zip(ys, xs):
                    if y % step != 0: continue
                    if y not in rows: rows[y] = []
                    rows[y].append(x)
                for y in rows: rows[y].sort()

                segments = set()
                for y in sorted(rows.keys()):
                    x_coords = rows[y]
                    if not x_coords: continue
                    start_x = x_coords[0]
                    for i in range(1, len(x_coords)):
                        if x_coords[i] != x_coords[i - 1] + 1:
                            if (x_coords[i - 1] - start_x) >= min_len:
                                segments.add(((start_x, y), (x_coords[i - 1], y)))
                            start_x = x_coords[i]
                    if (x_coords[-1] - start_x) >= min_len:
                        segments.add(((start_x, y), (x_coords[-1], y)))

            self._log(f"[{phase_title}] Found {len(segments)} raw segments.")
            self._log(f"[{phase_title}] Stitching segments into continuous strokes...")
            endpoints = {}
            for start_pt, end_pt in segments:
                endpoints.setdefault(start_pt, []).append((start_pt, end_pt))
                endpoints.setdefault(end_pt, []).append((start_pt, end_pt))

            super_strokes = []
            while segments:
                current_path = list(segments.pop())
                while True:
                    found, segments, new_end = self._extend_path(current_path[-1], segments, endpoints, stitch_dist)
                    if not found: break
                    current_path.append(new_end)
                while True:
                    found, segments, new_start = self._extend_path(current_path[0], segments, endpoints, stitch_dist)
                    if not found: break
                    current_path.insert(0, new_start)
                super_strokes.append(current_path)

            self._log(f"[{phase_title}] Consolidated into {len(super_strokes)} strokes.")
            self._log(f"[{phase_title}] Sorting strokes into {GRID_DIVISIONS}x{GRID_DIVISIONS} grid...")
            image_height, image_width = self.image_shape
            cell_width = image_width / GRID_DIVISIONS
            cell_height = image_height / GRID_DIVISIONS
            grid = [[] for _ in range(GRID_DIVISIONS * GRID_DIVISIONS)]
            for stroke in super_strokes:
                center_x = (stroke[0][0] + stroke[1][0]) / 2
                center_y = (stroke[0][1] + stroke[1][1]) / 2
                grid_x = min(int(center_x // cell_width), GRID_DIVISIONS - 1)
                grid_y = min(int(center_y // cell_height), GRID_DIVISIONS - 1)
                grid[grid_y * GRID_DIVISIONS + grid_x].append(stroke)

            self._log(f"[{phase_title}] Applying Nearest Neighbor path sort...")
            ordered_strokes = []
            last_pen_position = (0, 0)
            for cell_idx, cell in enumerate(grid):
                if not cell: continue
                undrawn_in_cell = list(cell)
                while undrawn_in_cell:
                    best_stroke_info = None
                    min_dist = float('inf')
                    for i, stroke in enumerate(undrawn_in_cell):
                        start_point, end_point = stroke[0], stroke[-1]
                        dist_to_start = self._calculate_distance(last_pen_position, start_point)
                        dist_to_end = self._calculate_distance(last_pen_position, end_point)
                        if dist_to_start < min_dist: min_dist = dist_to_start; best_stroke_info = (stroke, i, False)
                        if dist_to_end < min_dist: min_dist = dist_to_end; best_stroke_info = (stroke, i, True)
                    found_stroke, found_idx, should_reverse = best_stroke_info
                    if should_reverse: found_stroke.reverse()
                    last_pen_position = found_stroke[-1]
                    ordered_strokes.append(found_stroke)
                    undrawn_in_cell.pop(found_idx)
            return ordered_strokes

        def run_draw_loop(ordered_strokes, phase_title):
            if not ordered_strokes:
                return 0

            for i in range(2, 0, -1):
                layout["main"].update(Panel(Align.center(
                    f"\n\n[bold {'green' if phase_title == 'SKETCH' else 'cyan'}]{phase_title} PHASE[/]\nStarting in {i}..."),
                    border_style="green"))
                live.refresh()
                time.sleep(1)

            canvas_width = self.canvas_bottom_right[0] - self.canvas_top_left[0]
            canvas_height = self.canvas_bottom_right[1] - self.canvas_top_left[1]
            image_height, image_width = self.image_shape
            scale = min(canvas_width / image_width, canvas_height / image_height)
            offset_x = int((canvas_width - int(image_width * scale)) / 2)
            offset_y = int((canvas_height - int(image_height * scale)) / 2)
            total_strokes = len(ordered_strokes)
            strokes_drawn = 0

            progress = Progress(BarColumn(bar_width=None), TextColumn("[bold cyan]{task.percentage:>3.1f}%"))
            progress_task = progress.add_task("Progress", total=total_strokes)

            start_time = time.perf_counter()
            with PreciseSleeper(period_ms=1):
                for i, stroke in enumerate(ordered_strokes):
                    if self.panic_event.is_set(): break
                    strokes_drawn = i + 1

                    elapsed_time = time.perf_counter() - start_time
                    strokes_per_sec = strokes_drawn / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (elapsed_time / strokes_drawn) * (
                            total_strokes - strokes_drawn) if strokes_drawn > 0 else float('inf')
                    avg_stroke_time_ms = (elapsed_time / strokes_drawn) * 1000 if strokes_drawn > 0 else 0

                    center_x = (stroke[0][0] + stroke[1][0]) / 2
                    center_y = (stroke[0][1] + stroke[1][1]) / 2
                    cell_width = image_width / GRID_DIVISIONS
                    cell_height = image_height / GRID_DIVISIONS
                    grid_x = min(int(center_x // cell_width), GRID_DIVISIONS - 1)
                    grid_y = min(int(center_y // cell_height), GRID_DIVISIONS - 1)
                    current_grid_cell = (grid_y * GRID_DIVISIONS + grid_x) + 1

                    # --- Parameters Panel ---
                    canvas_w = self.canvas_bottom_right[0] - self.canvas_top_left[0]
                    canvas_h = self.canvas_bottom_right[1] - self.canvas_top_left[1]
                    param_table = Table(box=None, show_header=False, expand=True)
                    param_table.add_column(style="bold dim", width=12);
                    param_table.add_column(style="bright_white")
                    param_table.add_row("Phase:",
                                        f"[bold {'green' if phase_title == 'SKETCH' else 'cyan'}]{phase_title}[/]")
                    param_table.add_row("Speed:", f"{DRAWING_SPEED_PERCENT}%")
                    param_table.add_row("Strokes:", f"{total_strokes}")
                    param_table.add_row("Canvas:", f"{canvas_w}x{canvas_h}px")

                    if phase_title == "SHADING":
                        param_table.add_row("Method:", "Cross-Hatching")
                        param_table.add_row("Layers:", f"{len(SHADING_TONE_BANDS)}")
                        param_table.add_row("Stitching:", f"{SHADING_STITCH_DIST} px")
                    else:
                        param_table.add_row("Grid Size:", f"{GRID_DIVISIONS}x{GRID_DIVISIONS}")
                        param_table.add_row("Stitching:", f"{STITCHING_DISTANCE} px")

                    # --- Live Status Panel ---
                    data_rate_bps = (
                                                self.api_call_counter * self.COMMAND_PACKET_SIZE) / elapsed_time if elapsed_time > 0 else 0
                    formatted_rate = self._format_data_rate(data_rate_bps)
                    status_table = Table(box=None, show_header=False, expand=True)
                    status_table.add_column(style="bold dim", width=12)
                    status_table.add_column()
                    status_table.add_row("Progress:", progress)
                    time_str = f"[cyan]Elapsed:[/] [bright_white]{self._format_time(elapsed_time)}[/] [dim]|[/] [cyan]ETA:[/] [yellow]{self._format_time(eta_seconds) if eta_seconds != float('inf') else '...'}[/]"
                    status_table.add_row("Time:", time_str)
                    status_table.add_row("Pen State:", Spinner("dots",
                                                               text=f"{self.current_pen_state} {strokes_drawn}/{total_strokes}"))
                    status_table.add_row("Complexity:", f"{len(stroke)} vertices")
                    status_table.add_row("API Calls:", f"{self.api_call_counter:,}")
                    status_table.add_row("Data Rate:", formatted_rate)
                    status_table.add_row("Stroke Time:", f"{avg_stroke_time_ms:.0f}ms (avg)")
                    status_table.add_row("Speed:", f"{strokes_per_sec:.1f} strk/s")

                    if phase_title != "SHADING":
                        status_table.add_row("Location:", f"Grid {current_grid_cell} of {GRID_DIVISIONS ** 2}")

                    status_table.add_row("Distance:", self._format_distance(self.total_distance_traveled))

                    log_panel = Panel('\n'.join(self._log_messages), title="[bold]Event Log", border_style="dim green")

                    # --- Render Stroke Preview ---
                    preview_width, preview_height = 38, 10
                    braille_canvas = self._render_stroke_to_braille(stroke, preview_width, preview_height)
                    aligned_canvas = Align.center(braille_canvas, vertical="middle")
                    preview_panel = Panel(aligned_canvas, title="[bold]Live Preview", border_style="green",
                                          height=preview_height + 2, padding=0)

                    # --- Update Layout ---
                    layout["side"].split(
                        Panel(param_table, title="[bold]Parameters", border_style="green"),
                        preview_panel
                    )
                    layout["body"].split(
                        Panel(status_table, title=f"[bold]Live Status[{phase_title}]", border_style="green"),
                        log_panel
                    )

                    # --- Draw and Update ---
                    is_shade_phase = (phase_title == "SHADING")
                    self._draw_path(stroke, scale, offset_x, offset_y, WINDOWS_SCALING_FACTOR,
                                    is_shading=is_shade_phase)

                    if not is_shade_phase:
                        time.sleep(self.inter_stroke_delay)

                    progress.update(progress_task, advance=1)
                    live.refresh()
            return strokes_drawn

        try:
            # ================= PHASE 1: SKETCH =================
            sketch_strokes = process_phase(image_layer, "SKETCH", step=1, min_len=0, stitch_dist=STITCHING_DISTANCE)
            total_drawn = run_draw_loop(sketch_strokes, "SKETCH")
            total_possible = len(sketch_strokes)

            # ================= PHASE 2: SHADING (Cross-Hatching) =================
            if ENABLE_SHADING and not self.panic_event.is_set():
                play_sound_async("processing.mp3")
                layout["main"].update(Panel(
                    Align.center(
                        "\n\n[bold yellow]SKETCH COMPLETE![/bold yellow]\n\n"
                        "Please select a [bold cyan]1px brush[/bold cyan] for pixel-perfect tone shading.\n\n"
                        "[bold blink]>>> PRESS ENTER TO START SHADING <<<[/bold blink]"),
                    border_style="yellow"
                ))
                live.refresh()
                time.sleep(0.5)
                keyboard.wait('enter')

                if not self.panic_event.is_set():
                    self._log("[SHADING] Building cross-hatching topology from tone analysis...")
                    layout["main"].update(
                        Panel(Align.center("\n\n[bold cyan]Generating Cross-Hatch Topology...[/bold cyan]"),
                              border_style="cyan"))
                    live.refresh()

                    # Load and resize original image for tone analysis
                    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    img_h, img_w = self.image_shape
                    orig_img = cv2.resize(orig_img, (img_w, img_h), interpolation=cv2.INTER_AREA)

                    # Smooth to avoid noisy tone boundaries
                    orig_img = cv2.GaussianBlur(orig_img, (5, 5), 0)

                    # Build sketch exclusion mask (only dilate if margin > 0)
                    if SHADING_LINE_MARGIN > 0:
                        margin_k = SHADING_LINE_MARGIN * 2 + 1
                        line_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_k, margin_k))
                        dilated_lines = cv2.dilate(image_layer, line_kernel)
                    else:
                        dilated_lines = image_layer.copy()

                    # --- INTERACTIVE SHADING TUNER ---
                    active_light = SHADING_TONE_BANDS[0][0]
                    active_medium = SHADING_TONE_BANDS[1][0]
                    active_deep = SHADING_TONE_BANDS[2][0]

                    if getattr(self, 'interactive_tuning', True):  # Default True if not explicitly false
                        # Generate static screentone patterns for instant live compositing
                        pat_light = np.full((img_h, img_w), 255, dtype=np.uint8)
                        pat_medium = np.full((img_h, img_w), 255, dtype=np.uint8)
                        
                        # 45 deg, spacing 3
                        for c in range(0, img_w + img_h, 3 * 1):
                            xs = np.arange(max(0, c - img_h + 1), min(img_w, c + 1))
                            ys = c - xs
                            valid = (ys >= 0) & (ys < img_h)
                            pat_light[ys[valid], xs[valid]] = 0
                            
                        # 135 deg, spacing 3
                        for c in range(-img_w + 1, img_h, 3 * 1):
                            xs = np.arange(max(0, -c), min(img_w, img_h - c))
                            ys = xs + c
                            valid = (ys >= 0) & (ys < img_h)
                            pat_medium[ys[valid], xs[valid]] = 0

                        # Create windows
                        cv2.namedWindow("SOTA Shading Tuner - Press ENTER to Accept", cv2.WINDOW_NORMAL)
                        cv2.namedWindow("Original Reference", cv2.WINDOW_NORMAL)
                        
                        aspect_ratio = img_w / img_h
                        if aspect_ratio >= 1:
                            win_w, win_h = 800, int(800 / aspect_ratio)
                        else:
                            win_h, win_w = 800, int(800 * aspect_ratio)
                        
                        cv2.resizeWindow("SOTA Shading Tuner - Press ENTER to Accept", win_w, win_h)
                        cv2.resizeWindow("Original Reference", win_w, win_h)
                        
                        # Load full color rendering of reference
                        display_ref = cv2.imread(image_path)
                        if display_ref is not None:
                            cv2.imshow("Original Reference", display_ref)

                        cv2.createTrackbar("1. Light Shadows", "SOTA Shading Tuner - Press ENTER to Accept", active_light, 255, lambda x: None)
                        cv2.createTrackbar("2. Medium Shadows", "SOTA Shading Tuner - Press ENTER to Accept", active_medium, 255, lambda x: None)
                        cv2.createTrackbar("3. Deep Shadows", "SOTA Shading Tuner - Press ENTER to Accept", active_deep, 255, lambda x: None)

                        self._log("Opening Shading Tuner Engine...")
                        while True:
                            active_light = cv2.getTrackbarPos("1. Light Shadows", "SOTA Shading Tuner - Press ENTER to Accept")
                            active_medium = cv2.getTrackbarPos("2. Medium Shadows", "SOTA Shading Tuner - Press ENTER to Accept")
                            active_deep = cv2.getTrackbarPos("3. Deep Shadows", "SOTA Shading Tuner - Press ENTER to Accept")
                            
                            # Logically enforce nested thresholds
                            active_medium = min(active_medium, max(0, active_light - 1))
                            active_deep = min(active_deep, max(0, active_medium - 1))
                            
                            # Composite fast preview
                            preview = np.full((img_h, img_w), 255, dtype=np.uint8)
                            preview[dilated_lines > 0] = 0 # Add solid sketch
                            
                            mask_light = (orig_img < active_light) & (dilated_lines == 0)
                            mask_med = (orig_img < active_medium) & (dilated_lines == 0)
                            mask_deep = (orig_img < active_deep) & (dilated_lines == 0)
                            
                            preview = np.where(mask_light, pat_light, preview)
                            preview = np.where(mask_med, np.minimum(pat_light, pat_medium), preview)
                            preview[mask_deep] = 0
                            
                            cv2.imshow("SOTA Shading Tuner - Press ENTER to Accept", preview.astype(np.uint8))
                            
                            key = cv2.waitKey(33) & 0xFF
                            if key == 13 or key == 27:
                                break
                                
                        cv2.destroyAllWindows()
                        for _ in range(4): cv2.waitKey(1)
                        
                        SHADING_TONE_BANDS[0] = (active_light, SHADING_TONE_BANDS[0][1], SHADING_TONE_BANDS[0][2])
                        SHADING_TONE_BANDS[1] = (active_medium, SHADING_TONE_BANDS[1][1], SHADING_TONE_BANDS[1][2])
                        SHADING_TONE_BANDS[2] = (active_deep, SHADING_TONE_BANDS[2][1], SHADING_TONE_BANDS[2][2])
                        self._log(f"Locked Tones: Light(>{active_light}), Med(>{active_medium}), Deep(>{active_deep})")

                    # Generate cross-hatching segments for each tone band
                    all_segments = set()
                    for threshold, angle, spacing in SHADING_TONE_BANDS:
                        # Create mask: pixels darker than this threshold
                        tone_mask = np.zeros_like(orig_img)
                        tone_mask[orig_img < threshold] = 255

                        # Exclude already-drawn sketch pixels
                        tone_mask[dilated_lines > 0] = 0

                        # Clean small noise blobs
                        if MINIMUM_PIXEL_AREA > 0:
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                                tone_mask, 4, cv2.CV_32S)
                            for lbl in range(1, num_labels):
                                if stats[lbl, cv2.CC_STAT_AREA] < MINIMUM_PIXEL_AREA * 4:
                                    tone_mask[labels == lbl] = 0

                        # Generate hatching segments at this angle and spacing
                        band_segs = self._scan_mask_at_angle(
                            tone_mask, angle, spacing, SHADING_MIN_LINE_LEN)
                        self._log(f"[SHADING] Tone <{threshold} @ {angle}\u00b0: {len(band_segs)} segments")
                        all_segments.update(band_segs)

                    self._log(f"[SHADING] Total cross-hatch segments: {len(all_segments)}")

                    # Route through the stitching & TSP optimizer
                    shade_strokes = process_phase(
                        None,
                        "SHADING",
                        step=1,
                        min_len=SHADING_MIN_LINE_LEN,
                        stitch_dist=SHADING_STITCH_DIST,
                        pre_segments=all_segments
                    )

                    shade_drawn = run_draw_loop(shade_strokes, "SHADING")
                    total_drawn += shade_drawn
                    total_possible += len(shade_strokes)

        finally:
            try:
                keyboard.remove_hotkey('esc')
            except (KeyError, ValueError):
                pass

        return total_drawn, total_possible

    def run(self):
        """Main execution flow of the application, now managing the entire Rich UI lifecycle."""
        self.panic_event.clear()
        self.api_call_counter = 0
        self.total_distance_traveled = 0.0
        self._log_messages = []
        self._log("Application initialized.")

        # --- Define Layout Structure ---
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(ratio=1, name="main"),
            Layout(size=3, name="footer"),
        )
        layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2))

        # --- Define Static Components ---
        header_text = "[bold]RAZER HYDRA[/bold]"
        layout["header"].update(Panel(Align.center(header_text), style="bold green", border_style="green"))
        layout["footer"].update(Panel(Align.center("[dim]Press ESC to stop at any time[/dim]"), border_style="green"))

        # --- Get Image Path ---
        image_path = self._get_image_path()
        if not image_path:
            return "cancelled"  # Return a status

        # --- Main Live Context ---
        with Live(layout, console=console, screen=True, transient=True, vertical_overflow="visible") as live:
            theme_color = "cyan"
            layout["header"].update(Panel(Align.center(header_text), style=f"bold {theme_color}", border_style=theme_color))
            layout["footer"].update(Panel(Align.center("[dim]Press ESC to stop at any time[/dim]"), border_style=theme_color))

            # --- STAGE 1: CALIBRATION ---
            if not self._calibrate_canvas(live, layout):
                return "cancelled"

            # --- STAGE 2: AI PROCESSING ---
            self._log("Starting AI image processing...")
            live.refresh()
            final_image_to_draw = self._process_image(image_path, live, layout)

            # --- STAGE 3: DRAWING ---
            theme_color = "green"
            layout["header"].update(Panel(Align.center(header_text), style=f"bold {theme_color}", border_style=theme_color))
            layout["footer"].update(Panel(Align.center("[dim]Press ESC to stop at any time[/dim]"), border_style=theme_color))
            self._log("Image processing complete. Starting draw sequence.")
            live.refresh()

            start_time = time.perf_counter()
            strokes_drawn, total_strokes = self.draw_innovative(final_image_to_draw, live, layout, image_path)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # --- STAGE 4: AWAITING USER ---
            status = "HALTED BY USER" if self.panic_event.is_set() else "MISSION COMPLETE"
            theme_color = "bright_red" if self.panic_event.is_set() else "gold3"

            # Play the completion sound only on success
            if not self.panic_event.is_set():
                play_sound_async("complete.mp3")

            self._log(f"[bold {theme_color}]{status}[/bold {theme_color}]")
            live.refresh()

            # --- Non-Blocking Blinking Footer ---
            stop_blinking = threading.Event()

            def blink_footer():
                """Function to run in a background thread to blink the footer."""
                prompt_text = "[bold]>>> PRESS ENTER TO CONTINUE <<<[/bold]"
                is_visible = True
                while not stop_blinking.is_set():
                    display_text = prompt_text if is_visible else " "
                    layout["footer"].update(
                        Panel(Align.center(display_text), style=f"bold {theme_color}", border_style=theme_color)
                    )
                    live.refresh()
                    is_visible = not is_visible
                    time.sleep(0.5)

            # Start the blinking in the background
            blinker_thread = threading.Thread(target=blink_footer, daemon=True)
            blinker_thread.start()

            # Main thread waits for user input
            keyboard.wait('enter')

            # Signal the blinking to stop and clean up
            stop_blinking.set()
            blinker_thread.join(timeout=1)  # Wait briefly for the thread to finish

            return "finished"


if __name__ == "__main__":
    colorama.init()
    console = Console()
    play_sound_async("init.mp3")
    razer_controller = None
    try:
        razer_controller = RazerMouse()
        sketcher = ImageSketcher(razer_controller)

        while True:
            try:
                # The run method now manages its own UI and returns a status.
                status = sketcher.run()
                if status == "cancelled":
                    console.print("[yellow]Operation cancelled by user.[/yellow]")

            except FileNotFoundError as e:
                console.print("\n--- MODEL ERROR ---")
                console.print(f"Could not find the model weights file: {e.filename}")
                console.print("Please ensure you have a 'weights' folder in the same directory as this script,")
                console.print("and it contains the 'improved.bin' or 'netG.pth' file.")
                break  # A missing model is a fatal error, so we break the loop.
            except Exception as e:
                console.print(f"\nAn error occurred during the drawing process: {e}")
                console.print("The process for this image has been stopped.")

            # A robust input loop that only accepts 'y' or 'n'.
            answer = ''
            while answer not in ['y', 'n']:
                # The input prompt is now outside the Live display context.
                answer = console.input("\n[bold]Do you want to draw another image? (y/n):[/bold] ").lower().strip()
                if answer not in ['y', 'n']:
                    console.print("   [dim]Invalid input. Please enter 'y' or 'n'.[/dim]")

            if answer == 'n':
                break

    except Exception as e:
        console.print(f"\nAn critical error occurred (e.g., failed to connect to Razer device): {e}")
        console.print("Please ensure the script is run as an Administrator and Razer Synapse is running.")
    finally:
        if razer_controller:
            razer_controller.close()
        console.print("\nExiting program. Goodbye!")
        play_sound_async("exit.mp3")
        time.sleep(4)  # Give the exit sound a moment to play