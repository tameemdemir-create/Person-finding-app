import base64
import json
import os
import queue
import ssl
import threading
import time
import urllib.error
import urllib.request
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from server import start_server

LOCAL_SERVER_URL = "https://127.0.0.1:5000"
SERVER_URL = os.environ.get(
    "CAMERA_SERVER_URL", "https://person-finding-app.onrender.com"
).rstrip("/")
HTTP_CONTEXT = ssl._create_unverified_context()
PHONE_SITE_URL = "https://tameemdemir-create.github.io/Person-finding-app/"
BLACK = "#030704"
SURFACE = "#0a120c"
PANEL = "#0d190f"
GREEN = "#39f27a"
SOFT_GREEN = "#b5f7c8"
MUTED_GREEN = "#6eae80"
BORDER_GREEN = "#1c6736"


def api_request(path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        SERVER_URL + path, data=data, headers=headers
    )
    with urllib.request.urlopen(request, context=HTTP_CONTEXT, timeout=3) as response:
        return response.read()


class LaptopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("تطبيق البحث عن الشخص")
        self.root.geometry("1220x800")
        self.root.minsize(940, 680)
        self.root.configure(bg=BLACK)
        self.updates = queue.Queue()
        self.closed = False
        self.last_frame_id = 0
        self.displayed_frame_id = 0
        self.latest_live_update = None
        self.live_update_lock = threading.Lock()
        self.reference_image_path = None
        self.search_active = False

        self._build_ui()
        threading.Thread(target=self._poll_server, daemon=True).start()
        self.root.after(150, self._process_updates)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton", padding=10, background=PANEL, foreground=GREEN,
            bordercolor=BORDER_GREEN, lightcolor=BORDER_GREEN,
            darkcolor=BLACK, font=("Segoe UI", 10, "bold")
        )
        style.map(
            "TButton",
            background=[("active", BORDER_GREEN), ("disabled", SURFACE)],
            foreground=[("disabled", MUTED_GREEN)]
        )
        style.configure(
            "Accent.TButton", padding=12, background=GREEN, foreground=BLACK,
            bordercolor=GREEN, lightcolor=GREEN, darkcolor=BORDER_GREEN,
            font=("Segoe UI", 11, "bold")
        )
        style.map(
            "Accent.TButton",
            background=[("active", SOFT_GREEN), ("disabled", BORDER_GREEN)],
            foreground=[("disabled", MUTED_GREEN)]
        )
        style.configure(
            "TEntry", padding=9, fieldbackground=BLACK, foreground=SOFT_GREEN,
            insertcolor=GREEN, bordercolor=BORDER_GREEN
        )

        outer = tk.Frame(self.root, bg=BLACK, padx=22, pady=20)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        header = tk.Frame(outer, bg=BLACK, pady=8)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        tk.Label(
            header, text="نظام تتبّع الأشخاص", bg=BLACK, fg=GREEN,
            font=("Segoe UI", 23, "bold"), anchor="e"
        ).grid(row=0, column=1, sticky="e")
        tk.Label(
            header,
            text="رؤية مباشرة  /  كاميرا ذكية  /  بحث دقيق",
            bg=BLACK, fg=MUTED_GREEN, font=("Segoe UI", 10), anchor="e"
        ).grid(row=1, column=1, sticky="e", pady=(4, 0))
        tk.Label(
            header, text="●  LIVE SYSTEM", bg=SURFACE, fg=GREEN,
            font=("Segoe UI", 9, "bold"), padx=12, pady=7
        ).grid(row=0, column=0, rowspan=2, sticky="w")

        video_panel = tk.Frame(
            outer, bg=PANEL, padx=15, pady=15, highlightthickness=1,
            highlightbackground=BORDER_GREEN
        )
        video_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        video_panel.rowconfigure(1, weight=1)
        video_panel.columnconfigure(0, weight=1)
        tk.Label(
            video_panel, text="البث الحي", font=("Segoe UI", 15, "bold"),
            bg=PANEL, fg=SOFT_GREEN
        ).grid(row=0, column=0, sticky="e", pady=(0, 10))
        self.image_label = tk.Label(
            video_panel, text="بانتظار اتصال الهاتف...", bg=BLACK,
            fg=MUTED_GREEN, font=("Segoe UI", 13), compound="center"
        )
        self.image_label.grid(row=1, column=0, sticky="nsew")
        self.status_label = tk.Label(
            video_panel, text="الخادم يبدأ...", bg=PANEL, fg=GREEN,
            anchor="e", justify="right", font=("Segoe UI", 10)
        )
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        controls = tk.Frame(
            outer, bg=PANEL, padx=18, pady=18, highlightthickness=1,
            highlightbackground=BORDER_GREEN
        )
        controls.grid(row=1, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        tk.Label(
            controls, text="بحث ذكي عن شخص", font=("Segoe UI", 16, "bold"),
            bg=PANEL, fg=GREEN
        ).grid(row=0, column=0, sticky="e", pady=(0, 14))

        tk.Label(
            controls, text="اكتب المواصفات (مثال: قميص أحمر وبنطلون أسود)",
            bg=PANEL, fg=SOFT_GREEN, anchor="e", justify="right"
        ).grid(row=1, column=0, sticky="ew", pady=(5, 3))
        self.description_var = tk.StringVar()
        self.description_entry = ttk.Entry(
            controls, textvariable=self.description_var, justify="right"
        )
        self.description_entry.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        self.reference_label = tk.Label(
            controls, text="لم يتم اختيار صورة مرجعية", bg=PANEL,
            fg=MUTED_GREEN, anchor="e", justify="right", wraplength=330
        )
        self.reference_label.grid(row=3, column=0, sticky="ew", pady=(4, 4))
        ttk.Button(
            controls, text="اختيار صورة الشخص (اختياري)",
            command=self._choose_reference_image
        ).grid(row=4, column=0, sticky="ew")

        self.search_button = ttk.Button(
            controls, text="ابدأ البحث", command=self._search,
            style="Accent.TButton"
        )
        self.search_button.grid(row=5, column=0, sticky="ew", pady=(12, 8))
        self.result_label = tk.Label(
            controls, text="اكتب مواصفات واضحة أو اختر صورة للشخص.", bg=BLACK,
            fg=SOFT_GREEN, justify="right", anchor="ne", wraplength=320,
            padx=12, pady=12
        )
        self.result_label.grid(row=6, column=0, sticky="nsew", pady=(4, 12))
        controls.rowconfigure(6, weight=1)

        address = PHONE_SITE_URL
        tk.Label(
            controls, text="افتح موقع الهاتف:\n" + address,
            bg=PANEL, fg=GREEN, justify="right", wraplength=330
        ).grid(row=7, column=0, sticky="ew")
        ttk.Button(
            controls, text="نسخ رابط الهاتف",
            command=lambda: self._copy_address(address)
        ).grid(row=8, column=0, sticky="ew", pady=(8, 0))
        tk.Label(
            controls,
            text="تظهر إطارات خضراء حول الأشخاص المكتشفين، ويتحول إطار الشخص المطابق إلى الأحمر مع تكبيره.",
            bg=PANEL, fg=MUTED_GREEN, justify="right", wraplength=330
        ).grid(row=9, column=0, sticky="ew", pady=(14, 0))

    def _copy_address(self, address):
        self.root.clipboard_clear()
        self.root.clipboard_append(address)
        self.status_label.config(text="تم نسخ الرابط. افتحه على الهاتف.")

    def _choose_reference_image(self):
        image_path = filedialog.askopenfilename(
            title="اختر صورة الشخص",
            filetypes=[
                ("صور", "*.jpg *.jpeg *.png *.bmp"),
                ("كل الملفات", "*.*"),
            ],
        )
        if image_path:
            self.reference_image_path = image_path
            self.reference_label.config(
                text="الصورة المختارة: " + os.path.basename(image_path)
            )

    def _search(self):
        description = self.description_var.get().strip()
        if not description and not self.reference_image_path:
            self.result_label.config(text="اكتب مواصفات الشخص أو اختر صورة له أولًا.")
            return
        payload = {
            "description": description,
            "reference_path": self.reference_image_path,
        }
        self.search_button.config(state="disabled")
        threading.Thread(
            target=self._send_criteria, args=(payload,), daemon=True
        ).start()

    def _send_criteria(self, payload):
        try:
            request_payload = {
                "description": payload["description"],
                "reference_image": "",
            }
            if payload["reference_path"]:
                image_bytes = np.fromfile(payload["reference_path"], dtype=np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("تعذر قراءة الصورة المختارة.")
                height, width = image.shape[:2]
                scale = min(1.0, 640 / max(height, width))
                if scale < 1:
                    image = cv2.resize(
                        image,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                    )
                success, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not success:
                    raise ValueError("تعذر تجهيز الصورة المختارة.")
                request_payload["reference_image"] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii")
                )
            response = json.loads(api_request("/api/criteria", request_payload))
            self.updates.put(("criteria", response))
        except urllib.error.HTTPError as error:
            try:
                details = json.loads(error.read()).get("error", str(error))
            except (ValueError, UnicodeDecodeError):
                details = str(error)
            self.updates.put(("error", "تعذر بدء البحث: {}".format(details)))
        except (urllib.error.URLError, TimeoutError, ValueError, cv2.error) as error:
            self.updates.put(("error", "تعذر إرسال المواصفات: {}".format(error)))

    def _poll_server(self):
        while not self.closed:
            try:
                req = urllib.request.Request(
                    "{}{}?since={}".format(
                        SERVER_URL, "/api/latest.jpg", self.last_frame_id
                    )
                )
                with urllib.request.urlopen(
                    req, context=HTTP_CONTEXT, timeout=3
                ) as response:
                    frame_id = int(response.headers.get("X-Frame-Id", "0"))
                    status = {
                        "has_frame": frame_id > 0,
                        "frame_id": frame_id,
                        "people_count": int(
                            response.headers.get("X-People-Count", "0")
                        ),
                        "match_count": int(
                            response.headers.get("X-Match-Count", "0")
                        ),
                    }
                    frame = response.read() if response.status == 200 else None
                if frame is not None:
                    self.last_frame_id = frame_id
                with self.live_update_lock:
                    pending_update = self.latest_live_update
                    if (
                        frame is None
                        and pending_update is not None
                        and pending_update[0]["frame_id"] == frame_id
                    ):
                        frame = pending_update[1]
                    self.latest_live_update = (status, frame)
            except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError) as error:
                self.updates.put(("offline", str(error)))
            time.sleep(0.12)

    def _process_updates(self):
        try:
            while True:
                update = self.updates.get_nowait()
                kind = update[0]
                if kind == "criteria":
                    self.search_button.config(state="normal")
                    self._show_matches(update[1])
                elif kind == "error":
                    self.search_button.config(state="normal")
                    self.result_label.config(text=update[1])
                elif kind == "offline":
                    self.status_label.config(text="الخادم غير متصل: " + update[1])
        except queue.Empty:
            pass
        with self.live_update_lock:
            live_update = self.latest_live_update
            self.latest_live_update = None
        if live_update is not None:
            self._show_live(*live_update)
        if not self.closed:
            self.root.after(150, self._process_updates)

    def _show_live(self, status, frame):
        if frame is not None and status["frame_id"] != self.displayed_frame_id:
            try:
                decoded = cv2.imdecode(
                    np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None:
                    raise ValueError("تعذر فك ترميز صورة البث.")
                success, png = cv2.imencode(".png", decoded)
                if not success:
                    raise ValueError("تعذر فك ترميز صورة البث.")
                image = tk.PhotoImage(
                    data=base64.b64encode(png.tobytes()).decode("ascii")
                )
                self.image_label.config(image=image, text="")
                self.image_label.image = image
                self.displayed_frame_id = status["frame_id"]
            except (tk.TclError, ValueError, cv2.error) as error:
                self.status_label.config(text="تعذر عرض الصورة: " + str(error))
        if status["has_frame"]:
            count = status["people_count"]
            matches = status["match_count"]
            if matches:
                message = "تم العثور على {} شخص مطابق؛ تم تكبير الهدف.".format(matches)
            elif count and self.search_active:
                message = "تم اكتشاف {} أشخاص؛ لم يظهر تطابق بعد.".format(count)
            elif count:
                message = "تم اكتشاف أشخاص؛ أدخل المواصفات لبدء البحث."
            else:
                message = "جارٍ البحث عن الشخص في البث..."
            self.status_label.config(text=message)
        else:
            self.status_label.config(text="الخادم جاهز؛ افتح الموقع على الهاتف وابدأ الكاميرا.")

    def _show_matches(self, response):
        self.search_active = True
        self.result_label.config(
            text="بدأ البحث. الأشخاص بإطار أخضر، والهدف المطابق بإطار أحمر مع تكبيره."
        )

    def _close(self):
        self.closed = True
        self.root.destroy()


if __name__ == "__main__":
    if SERVER_URL == LOCAL_SERVER_URL:
        threading.Thread(target=start_server, daemon=True).start()
    root = tk.Tk()
    LaptopApp(root)
    root.mainloop()
