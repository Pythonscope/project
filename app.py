# ---------- Tkinter Desktop GUI ----------------------------
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from well_log_model import WellLogInterpreter


class WellLogGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-Powered Well-Log Interpreter")
        self.interp = WellLogInterpreter()

        # top control bar
        bar = ttk.Frame(root); bar.pack(fill=tk.X, pady=4)
        for lbl, cmd in [
            ("Load CSV",       self.load_csv),
            ("Generate Targets", self.targets),
            ("Train",            self.train),
            ("Show Logs",        self.plot_logs),
            ("Importance",       self.features),
            ("Metrics",          self.metrics),
            ("Recommend",        self.recommend)
        ]:
            ttk.Button(bar, text=lbl, command=cmd, width=15).pack(side=tk.LEFT, padx=2)

        # scrolling console
        self.console = tk.Text(root, height=14, bg="#111", fg="#0f0",
                               font=("Consolas", 9), wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True)

    # ------------- helpers ---------------------------------
    def log(self, txt, clear=False):
        if clear:
            self.console.delete(1.0, tk.END)
        self.console.insert(tk.END, txt + "\n")
        self.console.see(tk.END)

    def need_data(self):
        if self.interp.data is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return True
        return False

    # ------------- callbacks -------------------------------
    def load_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if not fp: return
        try:
            self.interp.preprocess_data(fp)
            self.log(f"Loaded {os.path.basename(fp)}", True)
        except Exception as e:
            messagebox.showerror("Error", e)

    def targets(self):
        if self.need_data(): return
        self.interp.generate_targets()
        self.log("Synthetic targets generated.")
        self.log(str(self.interp.data['LITHOLOGY'].value_counts()))

    def train(self):
        if self.need_data(): return
        self.interp.train_models()
        self.log("Models trained.")
        for k,v in self.interp.metrics.items():
            self.log(f"{k}: {v:.4f}")

    def plot_logs(self):
        if self.need_data(): return
        fig = self.interp.make_plot()
        win = tk.Toplevel(self.root); win.title("Well Logs")
        FigureCanvasTkAgg(fig, win).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def features(self):
        if not self.interp.lithology_model:
            messagebox.showinfo("Info","Train first."); return
        feats = self.interp.feature_columns + self.interp.extra_features
        imps  = self.interp.lithology_model.feature_importances_
        msg = "\n".join(f"{f}: {v:.4f}" for f,v in zip(feats,imps))
        self.log("Feature Importance:\n"+msg)

    def metrics(self):
        if not self.interp.metrics:
            messagebox.showinfo("Info","Train first."); return
        for k,v in self.interp.metrics.items():
            self.log(f"{k}: {v:.4f}")

    def recommend(self):
        if not self.interp.lithology_model:
            messagebox.showinfo("Info","Train first."); return
        self.log(self.interp.generate_recommendations())


# ------------- run -----------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    WellLogGUI(root)
    root.geometry("1100x700")
    root.mainloop()
