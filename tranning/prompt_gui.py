import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_learn import save_candidate  # noqa: E402


def main():
    root = tk.Tk()
    root.title("Prompt GUI")
    root.geometry("500x260")

    tk.Label(root, text="Prompt:").pack(pady=(15, 0))
    prompt_entry = tk.Entry(root, width=60)
    prompt_entry.pack(pady=5)

    tk.Label(root, text="Reply:").pack(pady=(10, 0))
    reply_entry = tk.Entry(root, width=60)
    reply_entry.pack(pady=5)

    status = tk.Label(root, text="", fg="green")
    status.pack(pady=10)

    def save(event=None):
        prompt, reply = prompt_entry.get().strip(), reply_entry.get().strip()
        if not prompt or not reply:
            status.config(text="請同時輸入 prompt 與 reply", fg="red")
            return
        save_candidate(prompt, reply, topic=prompt, source="manual_gui")
        status.config(text=f"已新增到 auto_learn_candidates.json：{prompt}", fg="green")
        prompt_entry.delete(0, tk.END)
        reply_entry.delete(0, tk.END)
        prompt_entry.focus_set()

    prompt_entry.bind("<Return>", lambda e: reply_entry.focus_set())
    reply_entry.bind("<Return>", save)
    tk.Button(root, text="Save", command=save).pack(pady=10)

    prompt_entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    main()
