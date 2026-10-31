import base64
import json
import queue
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
import tkinter as tk
from tkinter import ttk

from server import start_server

SERVER_URL = "https://127.0.0.1:5000"
HTTP_CONTEXT = ssl._create_unverified_context()
COLORS = {
    "أي لون": "",
    "أسود": "black",
    "أبيض": "white",
    "أحمر": "red",
    "برتقالي": "orange",
    "أصفر": "yellow",
    "أخضر": "green",
    "أزرق": "blue",
    "بنفسجي": "purple",
    "رمادي": "gray",
}


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


def get_laptop_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class LaptopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("تطبيق البحث عن الشخص")
        self.root.geometry("1100x720")
        self.root.minsize(850, 600)
        self.root.configure(bg="#0f172a")
        self.updates = queue.Queue()
        self.closed = False
        self.last_frame_id = None

        self._build_ui()
        threading.Thread(target=self._poll_server, daemon=True).start()
        self.root.after(150, self._process_updates)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e293b", foreground="#f8fafc")
        style.configure("TButton", padding=8)
        style.configure("TEntry", padding=6)
        style.configure("TCombobox", padding=6)

        outer = tk.Frame(self.root, bg="#0f172a", padx=18, pady=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        video_panel = tk.Frame(outer, bg="#1e293b", padx=12, pady=12)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_panel.rowconfigure(1, weight=1)
        video_panel.columnconfigure(0, weight=1)
        tk.Label(
            video_panel, text="بث كاميرا الهاتف", font=("Segoe UI", 16, "bold"),
            bg="#1e293b", fg="white"
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.image_label = tk.Label(
            video_panel, text="بانتظار اتصال الهاتف...", bg="#020617",
            fg="#cbd5e1", font=("Segoe UI", 13), compound="center"
        )
        self.image_label.grid(row=1, column=0, sticky="nsew")
        self.status_label = tk.Label(
            video_panel, text="الخادم يبدأ...", bg="#1e293b", fg="#a7f3d0",
            anchor="w", justify="right"
        )
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        controls = tk.Frame(outer, bg="#1e293b", padx=16, pady=16)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        tk.Label(
            controls, text="مواصفات الشخص", font=("Segoe UI", 16, "bold"),
            bg="#1e293b", fg="white"
        ).grid(row=0, column=0, sticky="e", pady=(0, 14))

        self.color_var = tk.StringVar(value="أي لون")
        self.type_var = tk.StringVar()
        self.fabric_var = tk.StringVar()
        self.scale_var = tk.StringVar()
        self._add_field(controls, 1, "لون الملابس", ttk.Combobox(
            controls, textvariable=self.color_var, values=list(COLORS),
            state="readonly", justify="right"
        ))
        self._add_field(controls, 2, "نوع الملابس (وصف)", ttk.Entry(
            controls, textvariable=self.type_var, justify="right"
        ))
        self._add_field(controls, 3, "القماش (وصف)", ttk.Entry(
            controls, textvariable=self.fabric_var, justify="right"
        ))
        self._add_field(controls, 4, "بكسل لكل سم (اختياري)", ttk.Entry(
            controls, textvariable=self.scale_var, justify="right"
        ))

        self.search_button = ttk.Button(
            controls, text="ابدأ البحث بالمواصفات", command=self._search
        )
        self.search_button.grid(row=9, column=0, sticky="ew", pady=(12, 8))
        self.result_label = tk.Label(
            controls, text="أدخل اللون ثم ابدأ البحث.", bg="#111827",
            fg="#e2e8f0", justify="right", anchor="ne", wraplength=320,
            padx=12, pady=12
        )
        self.result_label.grid(row=10, column=0, sticky="nsew", pady=(4, 12))
        controls.rowconfigure(10, weight=1)

        address = "https://{}:5000".format(get_laptop_ip())
        tk.Label(
            controls, text="افتح هذا الرابط على الهاتف:\n" + address,
            bg="#1e293b", fg="#93c5fd", justify="right", wraplength=330
        ).grid(row=11, column=0, sticky="ew")
        ttk.Button(
            controls, text="نسخ رابط الهاتف",
            command=lambda: self._copy_address(address)
        ).grid(row=12, column=0, sticky="ew", pady=(8, 0))
        tk.Label(
            controls,
            text="يُطابق التطبيق اللون آليًا. نوع الملابس والقماش أوصاف لا يمكن تأكيدها بهذه النسخة. السنتيمتر يحتاج معايرة.",
            bg="#1e293b", fg="#cbd5e1", justify="right", wraplength=330
        ).grid(row=13, column=0, sticky="ew", pady=(14, 0))

    @staticmethod
    def _add_field(parent, row, label_text, widget):
        tk.Label(
            parent, text=label_text, bg="#1e293b", fg="#cbd5e1",
            anchor="e", justify="right"
        ).grid(row=row * 2 - 1, column=0, sticky="ew", pady=(5, 3))
        widget.grid(row=row * 2, column=0, sticky="ew", pady=(0, 6))

    def _copy_address(self, address):
        self.root.clipboard_clear()
        self.root.clipboard_append(address)
        self.status_label.config(text="تم نسخ الرابط. افتحه على الهاتف.")

    def _search(self):
        try:
            pixels_per_cm = self.scale_var.get().strip()
            pixels_per_cm = float(pixels_per_cm) if pixels_per_cm else None
            if pixels_per_cm is not None and not 0.1 <= pixels_per_cm <= 100:
                raise ValueError
        except ValueError:
            self.result_label.config(
                text="قيمة المعايرة غير صحيحة. أدخل قيمة بين 0.1 و100 أو اتركها فارغة."
            )
            return

        payload = {
            "color": COLORS[self.color_var.get()],
            "clothing_type": self.type_var.get().strip(),
            "fabric": self.fabric_var.get().strip(),
            "pixels_per_cm": pixels_per_cm,
        }
        self.search_button.config(state="disabled")
        threading.Thread(
            target=self._send_criteria, args=(payload,), daemon=True
        ).start()

    def _send_criteria(self, payload):
        try:
            response = json.loads(api_request("/api/criteria", payload))
            self.updates.put(("criteria", response))
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            self.updates.put(("error", "تعذر إرسال المواصفات: {}".format(error)))

    def _poll_server(self):
        while not self.closed:
            try:
                status = json.loads(api_request("/api/status"))
                frame = api_request("/api/latest.png") if status["has_frame"] else None
                self.updates.put(("live", status, frame))
            except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError) as error:
                self.updates.put(("offline", str(error)))
            time.sleep(0.35)

    def _process_updates(self):
        try:
            while True:
                update = self.updates.get_nowait()
                kind = update[0]
                if kind == "live":
                    self._show_live(update[1], update[2])
                elif kind == "criteria":
                    self.search_button.config(state="normal")
                    self._show_matches(update[1])
                elif kind == "error":
                    self.search_button.config(state="normal")
                    self.result_label.config(text=update[1])
                elif kind == "offline":
                    self.status_label.config(text="الخادم غير متصل: " + update[1])
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(150, self._process_updates)

    def _show_live(self, status, frame):
        if frame is not None and status["frame_id"] != self.last_frame_id:
            try:
                image = tk.PhotoImage(data=base64.b64encode(frame).decode("ascii"))
                self.image_label.config(image=image, text="")
                self.image_label.image = image
                self.last_frame_id = status["frame_id"]
            except tk.TclError as error:
                self.status_label.config(text="تعذر عرض الصورة: " + str(error))
        if status["has_frame"]:
            self.status_label.config(
                text="متصل | الأشخاص المكتشفة: {}".format(len(status["people"]))
            )
        else:
            self.status_label.config(text="الخادم جاهز؛ افتح الموقع على الهاتف وابدأ الكاميرا.")

    def _show_matches(self, response):
        matches = response["matches"]
        if not response["people"]:
            text = "لا يوجد شخص ظاهر في بث الهاتف الآن."
        elif matches:
            details = []
            for index, person in enumerate(matches, start=1):
                height = "{} بكسل".format(person["height_px"])
                if person["height_cm"] is not None:
                    height += " / {:.1f} سم تقريبًا".format(person["height_cm"])
                details.append(
                    "الشخص {}: اللون {}، الارتفاع {}".format(
                        index, person["color_ar"], height
                    )
                )
            text = "تم العثور على {} شخص مطابق للون:\n".format(len(matches))
            text += "\n".join(details)
        else:
            text = "تم اكتشاف أشخاص، لكن لون الملابس لا يطابق اللون المطلوب."
        text += "\n\nملاحظة: نوع الملابس والقماش وصف يدوي، ولا يتم التحقق منهما بصريًا."
        self.result_label.config(text=text)

    def _close(self):
        self.closed = True
        self.root.destroy()


if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    root = tk.Tk()
    LaptopApp(root)
    root.mainloop()
